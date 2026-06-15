import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)
logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('git').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)

import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
import wandb
from tqdm.auto import tqdm 

# Accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler,
)
from models import (
    Generator,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from load_encoder import load_encoder
from scheduler import TriStageLRScheduler

torch.backends.cudnn.benchmark = True

def get_scheduler(accelerator, optimizer, hps, global_step):
    """Stage-based Tri-Stage Learning Rate Scheduler"""
    
    total_steps = hps.train.total_steps * accelerator.num_processes
    warmup_steps = int(total_steps * hps.train.warmup_ratio)
    hold_steps = int(total_steps * hps.train.hold_ratio)
    decay_steps = total_steps - warmup_steps - hold_steps
    
    scheduler = TriStageLRScheduler(
        optimizer,
        hps.train.init_lr,
        hps.train.peak_lr,
        hps.train.final_lr,
        warmup_steps,
        hold_steps,
        decay_steps,
        total_steps,
    )

    scheduler._step_count = global_step * accelerator.num_processes

    return scheduler

def main():
    hps = utils.get_hparams()
    
    # ---------------------------------------------------------
    # 1. Accelerate Init
    # ---------------------------------------------------------
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    config = ProjectConfiguration(project_dir=hps.experiment_dir, logging_dir=hps.experiment_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=hps.train.mixed_precision, 
        log_with="wandb",
        project_config=config,
        kwargs_handlers=[ddp_kwargs]
    )
    
    set_seed(hps.train.seed)

    if accelerator.is_main_process:
        logger = utils.get_logger(hps.experiment_dir)
        logger.info(hps)
        accelerator.init_trackers(
            project_name="URBERT (Interspeech2026)",
            config=dict(hps),
            init_kwargs={"wandb": {
                "name": os.path.basename(hps.experiment_dir),
                "reinit": True,
                "dir": hps.experiment_dir
            }}
        )
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)

    # ---------------------------------------------------------
    # 2. Model & Optimizer
    # ---------------------------------------------------------
    backbone, tokenizer = load_encoder(hps.bert_type, hps.bert_model_dir)
    
    net_g = Generator(
        backbone,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    
    freeze_step_threshold = int(hps.train.total_steps * hps.train.encoder_freeze_ratio)
    if hasattr(net_g, 'enc_p') and hasattr(net_g.enc_p, 'backbone') and hps.bert_type != 'vits':
        for param in net_g.enc_p.backbone.parameters():
            param.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Initial State: Backbone Encoder is FROZEN.")
    else:
        if accelerator.is_main_process:
            logger.info("Initial State: Backbone Encoder is UNFROZEN.")
    
    total_g, trainable_g, frozen_g = utils.count_parameters(net_g)
    total_d, trainable_d, frozen_d = utils.count_parameters(net_d)
    if accelerator.is_main_process:
        logger.info(f"Generator: Total Parameters: {total_g}, Trainable Parameters: {trainable_g}, Frozen Parameters: {frozen_g}")
        logger.info(f"Discriminator: Total Parameters: {total_d}, Trainable Parameters: {trainable_d}, Frozen Parameters: {frozen_d}")

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.init_lr,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.init_lr,
        betas=hps.train.betas,
        eps=hps.train.eps)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    collate_fn = TextAudioSpeakerCollate(pad_token_id=tokenizer.pad_token_id)
    
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, tokenizer)
    
    boundaries = [32, 100, 200, 300, 400, 500, 700, 900, 1100, 1500]
    
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        boundaries,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        num_workers=4, 
        shuffle=False, 
        pin_memory=True,
        drop_last=False, 
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True
    )
    
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, tokenizer)     
    eval_loader = DataLoader(
        eval_dataset, 
        num_workers=4, 
        shuffle=False,
        batch_size=hps.train.batch_size, 
        pin_memory=True,
        drop_last=False, 
        collate_fn=collate_fn,
        persistent_workers=True
    )

    # ---------------------------------------------------------
    # 4. Checkpoint Loading
    # ---------------------------------------------------------
    global_step = 0
    try:
        g_path = utils.latest_checkpoint_path(hps.experiment_dir, "G_*.pth")
        d_path = utils.latest_checkpoint_path(hps.experiment_dir, "D_*.pth")
        
        if g_path and d_path:
            step_str = os.path.basename(g_path).split('_')[1].split('.')[0]
            global_step = int(step_str)
            utils.load_checkpoint(g_path, net_g, optim_g)
            utils.load_checkpoint(d_path, net_d, optim_d)
            if accelerator.is_main_process:
                logger.info(f"Loaded checkpoint from step {global_step}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.info(f"Start from scratch: {e}")
            global_step = 0

    scheduler_g = get_scheduler(accelerator, optim_g, hps, global_step)
    scheduler_d = get_scheduler(accelerator, optim_d, hps, global_step)

    if accelerator.is_main_process:
        logger.info(f"Scheduler (Generator): Total Steps: {scheduler_g.total_steps} | Warmup Steps: {scheduler_g.warmup_steps} | Hold Steps: {scheduler_g.hold_steps} | Decay Steps: {scheduler_g.decay_steps}")
        logger.info(f"Scheduler (Discriminator): Total Steps: {scheduler_d.total_steps} | Warmup Steps: {scheduler_d.warmup_steps} | Hold Steps: {scheduler_d.hold_steps} | Decay Steps: {scheduler_d.decay_steps}")

    # ---------------------------------------------------------
    # 5. Prepare (Accelerate)
    # ---------------------------------------------------------
    net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, eval_loader = accelerator.prepare(
        net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, eval_loader
    )

    # ---------------------------------------------------------
    # 6. Training Loop
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        logger.info(f"Starting training loop. Backbone Freeze Limit: Step {freeze_step_threshold}")
        progress_bar = tqdm(total=hps.train.total_steps, initial=global_step, desc="Training", unit="step")
    
    is_bert_frozen = True
    if global_step >= freeze_step_threshold:
        is_bert_frozen = False
        unwrapped_net_g = accelerator.unwrap_model(net_g)
        if hasattr(unwrapped_net_g, 'enc_p') and hasattr(unwrapped_net_g.enc_p, 'backbone'):
            for param in unwrapped_net_g.enc_p.backbone.parameters():
                param.requires_grad = True
                
        if accelerator.is_main_process:
            total_g, trainable_g, frozen_g = utils.count_parameters(net_g)
            total_d, trainable_d, frozen_d = utils.count_parameters(net_d)
            logger.info(f"Generator: Total Parameters: {total_g}, Trainable Parameters: {trainable_g}, Frozen Parameters: {frozen_g}")
            logger.info(f"Discriminator: Total Parameters: {total_d}, Trainable Parameters: {trainable_d}, Frozen Parameters: {frozen_d}")

    epoch = 0
    
    while global_step < hps.train.total_steps:
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()

        for batch in train_loader:
            if global_step >= hps.train.total_steps:
                break
            
            # 1. BERT Freezing Check
            should_freeze = global_step < freeze_step_threshold
            
            if should_freeze != is_bert_frozen:
                unwrapped_net_g = accelerator.unwrap_model(net_g)
                if hasattr(unwrapped_net_g, 'enc_p') and hasattr(unwrapped_net_g.enc_p, 'backbone'):
                    for param in unwrapped_net_g.enc_p.backbone.parameters():
                        param.requires_grad = not should_freeze
                    
                    if accelerator.is_main_process:
                        status = "FROZEN" if should_freeze else "UNFROZEN"
                        logger.info(f"Step {global_step}: Backbone Encoder is now {status}!")
                        
                        total_g, trainable_g, frozen_g = utils.count_parameters(net_g)
                        total_d, trainable_d, frozen_d = utils.count_parameters(net_d)
                        logger.info(f"Generator: Total Parameters: {total_g}, Trainable Parameters: {trainable_g}, Frozen Parameters: {frozen_g}")
                        logger.info(f"Discriminator: Total Parameters: {total_d}, Trainable Parameters: {trainable_d}, Frozen Parameters: {frozen_d}")
                
                is_bert_frozen = should_freeze

            # 2. Train Step
            loss_dict = train_step(accelerator, batch, net_g, net_d, optim_g, optim_d, hps) # No Gradient Accumulation
                
            scheduler_g.step()
            scheduler_d.step()
            
            progress_bar.update(1)
            global_step += 1
            
            # 3. Logging
            if accelerator.is_main_process:
                
                current_lr_g = optim_g.param_groups[0]['lr']
                current_lr_d = optim_d.param_groups[0]['lr']
                
                progress_bar.set_postfix(
                    loss_g=f"{loss_dict['loss_gen_all'].item():.3f}", 
                    loss_d=f"{loss_dict['loss_disc'].item():.3f}",
                    lr=f"{current_lr_g:.1e}"
                )

                if global_step % hps.train.log_interval == 0:
                    log_metrics(accelerator, logger, loss_dict, global_step, current_lr_g, current_lr_d)

            # 4. Evaluation & Save
            if global_step % hps.train.eval_interval == 0 and global_step != 0:
                if accelerator.is_main_process:
                    logger.info(f"Evaluating at step {global_step}...")
                
                evaluate(hps, accelerator.unwrap_model(net_g), eval_loader, accelerator, global_step)
                
                if accelerator.is_main_process:
                    save_checkpoint(hps, accelerator, net_g, optim_g, net_d, optim_d, global_step)
            
        epoch += 1
            
    # Final Save
    if accelerator.is_main_process:
        progress_bar.close()
        logger.info("Training Completed!")
        save_checkpoint(hps, accelerator, net_g, optim_g, net_d, optim_d, global_step)
        accelerator.end_training()

def train_step(accelerator, batch, net_g, net_d, optim_g, optim_d, hps):
    # Unpack Batch (Already on Device)
    x, attention_mask, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

    # ==========================
    # 1. Discriminator Update
    # ==========================
    
    # Generator Forward for Disc input
    y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, attention_mask, spec, spec_lengths, sid=speakers)

    mel = spec_to_mel_torch(
        spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate,
        hps.data.mel_fmin, hps.data.mel_fmax)
    y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate,
        hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax
    )
    y_sliced = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

    # Disc Forward
    y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())
    
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    
    # Optimizer Step
    optim_d.zero_grad()
    accelerator.backward(loss_disc)
    grad_norm_d = accelerator.clip_grad_norm_(net_d.parameters(), float('inf')) 
    optim_d.step()

    # ==========================
    # 2. Generator Update
    # ==========================
    
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
    
    loss_dur = torch.sum(l_length.float())
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

    # Optimizer Step
    optim_g.zero_grad()
    accelerator.backward(loss_gen_all)
    grad_norm_g = accelerator.clip_grad_norm_(net_g.parameters(), float('inf')) 
    optim_g.step()

    return {
        "loss_disc": loss_disc, "loss_gen_all": loss_gen_all, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g,
        "loss_fm": loss_fm, "loss_mel": loss_mel, "loss_dur": loss_dur, "loss_kl": loss_kl,
        "mel_org": y_mel, "mel_gen": y_hat_mel, "mel_full": mel, "attn": attn
    }


def log_metrics(accelerator, logger, metrics, step, lr_g, lr_d):
    scalar_dict = {
        # Loss
        "loss/g/total": metrics["loss_gen_all"].item(),
        "loss/d/total": metrics["loss_disc"].item(),
        "loss/g/fm": metrics["loss_fm"].item(),
        "loss/g/mel": metrics["loss_mel"].item(),
        "loss/g/dur": metrics["loss_dur"].item(),
        "loss/g/kl": metrics["loss_kl"].item(),
        
        # Hyperparams (LR & Step)
        "common/learning_rate_g": lr_g,
        "common/learning_rate_d": lr_d,
        "common/grad_norm_d": metrics["grad_norm_d"].item(),
        "common/grad_norm_g": metrics["grad_norm_g"].item(),
        "common/global_step": step,
    }
    
    if logger is not None:
        logger.info(
            f"[Step {step}] "
            f"Loss G: {metrics['loss_gen_all'].item():.4f} | "
            f"Loss D: {metrics['loss_disc'].item():.4f} | "
            f"LR: {lr_g:.2e}"
        )

    wandb_dict = scalar_dict.copy()
    
    mel_org = utils.plot_spectrogram_to_numpy(metrics["mel_org"][0].data.cpu().numpy())
    mel_gen = utils.plot_spectrogram_to_numpy(metrics["mel_gen"][0].data.cpu().numpy())
    mel_full = utils.plot_spectrogram_to_numpy(metrics["mel_full"][0].data.cpu().numpy())
    attn_img = utils.plot_alignment_to_numpy(metrics["attn"][0, 0].data.cpu().numpy())

    wandb_dict.update({
        "slice/mel_org": wandb.Image(mel_org, caption="Original Mel Slice"),
        "slice/mel_gen": wandb.Image(mel_gen, caption="Generated Mel Slice"),
        "all/mel": wandb.Image(mel_full, caption="Full Mel"),
        "all/attn": wandb.Image(attn_img, caption="Alignment")
    })
    
    accelerator.get_tracker("wandb").log(wandb_dict, step=step)


def save_checkpoint(hps, accelerator, net_g, optim_g, net_d, optim_d, step):
    save_path_g = os.path.join(hps.experiment_dir, f"G_{step}.pth")
    save_path_d = os.path.join(hps.experiment_dir, f"D_{step}.pth")
    utils.save_checkpoint(accelerator.unwrap_model(net_g), optim_g, step, save_path_g)
    utils.save_checkpoint(accelerator.unwrap_model(net_d), optim_d, step, save_path_d)


def evaluate(hps, generator, eval_loader, accelerator, step):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx > 0: break
            
            x, attention_mask, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
            
            batch_size = x.size(0)
            target_list = [(0, "sample_1")]
            if batch_size > 1: target_list.append((batch_size // 2, "sample_mid"))

            for idx, tag in target_list:
                x_curr = x[idx:idx+1]
                mask_curr = attention_mask[idx:idx+1]
                spec_curr = spec[idx:idx+1]
                spec_len_curr = spec_lengths[idx:idx+1]
                y_curr = y[idx:idx+1]
                y_len_curr = y_lengths[idx:idx+1]
                sid_curr = speakers[idx:idx+1]

                y_hat, attn, mask, *_ = generator.infer(x_curr, mask_curr, sid=sid_curr, max_len=hps.inference_max_len)
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel_curr = spec_to_mel_torch(
                    spec_curr, hps.data.filter_length, hps.data.n_mel_channels, 
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels, 
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, 
                    hps.data.mel_fmin, hps.data.mel_fmax
                )

                gen_mel_img = utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
                gt_mel_img = utils.plot_spectrogram_to_numpy(mel_curr[..., :spec_len_curr[0].item()][0].cpu().numpy())
                
                gen_audio = y_hat[0, :, :y_hat_lengths[0]].cpu().float().numpy().squeeze()
                gt_audio = y_curr[0, :, :y_len_curr[0]].cpu().float().numpy().squeeze()

                log_dict = {
                    f"gen/{tag}/mel": wandb.Image(gen_mel_img, caption=f"Gen Mel {tag}"),
                    f"gt/{tag}/mel": wandb.Image(gt_mel_img, caption=f"GT Mel {tag}"),
                    f"gen/{tag}/audio": wandb.Audio(gen_audio, sample_rate=hps.data.sampling_rate, caption=f"Gen Audio {tag}"),
                    f"gt/{tag}/audio": wandb.Audio(gt_audio, sample_rate=hps.data.sampling_rate, caption=f"GT Audio {tag}")
                }
                
                accelerator.get_tracker("wandb").log(log_dict, step=step)

    generator.train()

if __name__ == "__main__":
    main()