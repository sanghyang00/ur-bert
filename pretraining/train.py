# train.py

import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from transformers import BertConfig, BertModel
from scheduler import TriStageLRScheduler
from torch.optim import AdamW
import wandb
import pandas as pd
from tqdm import tqdm
import math

from model import MultiTaskBert
from dataloader import URbertDataset, custom_collator
from tokenizer import UromanCharTokenizer
from urbert_utils import (
    get_hparams,
    HParams,
    save_checkpoint,
    load_checkpoint,
    latest_checkpoint_path,
    get_logger,
    get_combined_df_from_split,
    preprocess_acoustic_id
)

def train_and_evaluate(
    model,
    dataloader,
    optimizer,
    scheduler,
    accelerator,
    hps: HParams,
    model_dir: str,
    global_step: int = 0,
    logger=None,
    val_dataloader=None,
    val_dataloader_balanced=None
):
    model.train()
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    val_enabled = val_dataloader is not None
    
    accum_loss_total = 0.0
    accum_loss_text = 0.0
    accum_loss_audio = 0.0
    accum_count = 0
    
    progress_bar = tqdm(
        range(global_step, hps.num_steps),
        desc="Training",
        disable=not accelerator.is_main_process,
        initial=global_step,
        total=hps.num_steps
    )

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    
    train_iterator = iter(dataloader)

    while global_step < hps.num_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            batch = next(train_iterator)

        with accelerator.accumulate(model):
            text_mlm_logits, audio_distill_logits = model(
                uroman_ids=batch["uroman_ids"],
                attn_mask=batch["attn_mask"],
                text_mlm_on=hps.text_mlm_on,
                audio_distill_on=hps.audio_distill_on
            )

            # --- Loss Calculation ---
            loss_text_mlm = torch.tensor(0.0, device=accelerator.device)
            if hps.text_mlm_on and text_mlm_logits is not None:
                loss_text_mlm = loss_fct(
                    text_mlm_logits.view(-1, text_mlm_logits.size(-1)),
                    batch["uroman_labels"].view(-1)
                )

            loss_audio_distill = torch.tensor(0.0, device=accelerator.device)
            if hps.audio_distill_on and audio_distill_logits is not None:
                loss_audio_distill = loss_fct(
                    audio_distill_logits.view(-1, audio_distill_logits.size(-1)),
                    batch["audio_ids"].view(-1)
                )

            loss_total = (
                hps.optimizer.w_text * loss_text_mlm + 
                hps.optimizer.w_audio * loss_audio_distill
            )

            # --- Backward ---
            accelerator.backward(loss_total)
            
            accum_loss_total += loss_total.item()
            accum_loss_text += (hps.optimizer.w_text * loss_text_mlm).item()
            accum_loss_audio += (hps.optimizer.w_audio * loss_audio_distill).item()
            accum_count += 1 

            # --- Step Update (Sync Gradients) ---
            if accelerator.sync_gradients:
                
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                
                # Logging
                if global_step % hps.log_interval == 0:
                    avg_loss = accum_loss_total / accum_count
                    avg_text = accum_loss_text / accum_count
                    avg_audio = accum_loss_audio / accum_count
                
                    if accelerator.is_main_process:
                        if logger:
                            logger.info(
                                f"Step {global_step} | Total: {avg_loss:.4f} | "
                                f"Text: {avg_text:.4f} | Audio: {avg_audio:.4f}| "
                                f"LR: {scheduler.get_lr():.6f}"
                            )
                        wandb.log({
                            "train/loss_total": avg_loss,
                            "train/loss_text": avg_text,
                            "train/loss_audio": avg_audio,
                            "train/lr": scheduler.get_lr(),
                            "step": global_step
                        })
                    
                    accum_loss_total = 0.0
                    accum_loss_text = 0.0
                    accum_loss_audio = 0.0
                    accum_count = 0

                # Validation
                if val_enabled and global_step % hps.validation.interval == 0:
                    accelerator.wait_for_everyone()
                    val_loss_balanced, val_loss_all = evaluate(
                        model, val_dataloader, val_dataloader_balanced, accelerator, hps, logger
                    )
                    
                    if accelerator.is_main_process:
                        wandb.log({
                            "val/loss_balanced": val_loss_balanced,
                            "val/loss_all": val_loss_all,
                            "step": global_step
                        })
                    model.train() 

                # Checkpoint Saving
                if global_step % hps.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(unwrapped_model, optimizer, scheduler, global_step, checkpoint_dir)
                        if logger:
                            logger.info(f"Checkpoint saved at step {global_step}")

                if global_step >= hps.num_steps:
                    break
    
    progress_bar.close()

    # Final Validation
    if val_enabled:
        accelerator.wait_for_everyone()
        val_loss_balanced, val_loss_all = evaluate(
            model, val_dataloader, val_dataloader_balanced, accelerator, hps, logger
        )
        if accelerator.is_main_process:
            wandb.log({
                "val/final_loss_balanced": val_loss_balanced,
                "val/final_loss_all": val_loss_all,
                "step": global_step
            })

    if accelerator.is_main_process:
        wandb.finish()
        if logger:
            logger.info("Training finished successfully!")

def evaluate(model, dataloader_all, dataloader_balanced, accelerator, hps, logger=None):
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def _calc_loss(loader, desc):
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(
            loader, 
            desc=desc, 
            disable=not accelerator.is_main_process, 
            leave=False
        )
        
        with torch.no_grad():
            for batch in iterator:
                text_mlm_logits, audio_distill_logits = model(
                    uroman_ids=batch["uroman_ids"],
                    attn_mask=batch["attn_mask"],
                    text_mlm_on=hps.text_mlm_on,
                    audio_distill_on=hps.audio_distill_on
                )

                loss_text_mlm = torch.tensor(0.0, device=accelerator.device)
                if hps.text_mlm_on and text_mlm_logits is not None:
                    loss_text_mlm = loss_fct(
                        text_mlm_logits.view(-1, text_mlm_logits.size(-1)),
                        batch["uroman_labels"].view(-1)
                    )

                loss_audio_distill = torch.tensor(0.0, device=accelerator.device)
                if hps.audio_distill_on and audio_distill_logits is not None:
                    loss_audio_distill = loss_fct(
                        audio_distill_logits.view(-1, audio_distill_logits.size(-1)),
                        batch["audio_ids"].view(-1)
                    )

                batch_loss = loss_text_mlm + loss_audio_distill
                total_loss += batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

    loss_balanced = _calc_loss(dataloader_balanced, "Eval (Balanced)")
    loss_all = _calc_loss(dataloader_all, "Eval (All)")

    if accelerator.is_main_process and logger:
        logger.info(f"Eval Result | Balanced: {loss_balanced:.4f} | All: {loss_all:.4f}")

    return loss_balanced, loss_all

def training_function(args):
    # 1. Config & Params
    model_dir = os.path.join("logs", args.exp_name)
    hps = get_hparams(args.config, model_dir=model_dir)

    # 2. Accelerator Init
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision=hps.mixed_precision,
        gradient_accumulation_steps=hps.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    set_seed(42)

    # 3. Logger & WandB
    logger = None
    if accelerator.is_main_process:
        logger = get_logger(model_dir)
        logger.info(f"Training started on {accelerator.num_processes} GPUs")
        logger.info(f"Config: {args.config} | Exp: {args.exp_name}")
        
        wandb.init(
            project=hps.wandb.project,
            entity=hps.wandb.entity,
            name=args.exp_name,
            config=hps.__dict__,
            dir=model_dir,
            reinit=True
        )
    
    accelerator.wait_for_everyone()

    # 4. Model & Tokenizer
    tokenizer = UromanCharTokenizer()
    bert_config = BertConfig.from_pretrained(hps.model.base_model)
    bert_config.pad_token_id = tokenizer.pad_token_id # Fix the pad_token_id to our token index
    bert = BertModel(bert_config)
    bert.resize_token_embeddings(hps.model.vocab_size) # Keypoint: Further considerations to avoid shape mismatch on finetuning
    assert bert_config.pad_token_id == tokenizer.pad_token_id
    assert bert.embeddings.word_embeddings.weight.shape[0] == hps.model.vocab_size
    
    if accelerator.is_main_process and logger:
        logger.info(f"BERT Config: {bert_config}")
        logger.info(f"BERT pad_token_id: {bert_config.pad_token_id} | Tokenizer pad_token_id: {tokenizer.pad_token_id}")
        logger.info(f"BERT word_embeddings shape: {bert.embeddings.word_embeddings.weight.shape} | Tokenizer vocab_size: {len(tokenizer.vocab)}")
        logger.info(f"BERT word_embeddings weight of the first token (spacing): {bert.embeddings.word_embeddings.weight[0,:10]}")
        logger.info(f"BERT word_embeddings weight of the padding token: {bert.embeddings.word_embeddings.weight[bert_config.pad_token_id,:10]}")
    
    model = MultiTaskBert(
        bert,
        uroman_vocab_size=hps.model.vocab_size,
        acoustic_vocab_size=hps.model.acoustic_vocab_size
    )

    # 5. Data Preparation
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(project_root, hps.dataset.data_dir))

    train_df = get_combined_df_from_split(
        data_dir=data_dir,
        split='train',
        logger=logger if accelerator.is_main_process else None,
        required_cols=['romanized transcription', 'acoustic_id', 'language']
    )
    if train_df is None:
        raise FileNotFoundError("Train data not found")

    dataset = URbertDataset(
        df=train_df,
        tokenizer=tokenizer,
        text_mlm_on=hps.text_mlm_on,
        audio_distill_on=hps.audio_distill_on,
        max_seq_length=hps.dataset.max_seq_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=hps.batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collator(batch, tokenizer),
        num_workers=4,
        pin_memory=True
    )

    # Validation
    val_enabled = hps.validation.enabled if hasattr(hps, 'validation') else False
    val_dataloader = None
    val_dataloader_balanced = None

    if val_enabled:
        val_df = get_combined_df_from_split(
            data_dir=data_dir,
            split='dev',
            logger=None,
            required_cols=['romanized transcription', 'acoustic_id', 'language']
        )
        if val_df is not None:
            val_df_balanced = (
                val_df.sample(frac=1, random_state=42)
                .groupby('language')
                .head(1000)
                .reset_index(drop=True)
            )
            
            val_dataset = URbertDataset(
                df=val_df,
                tokenizer=tokenizer,
                text_mlm_on=hps.text_mlm_on,
                audio_distill_on=hps.audio_distill_on,
                max_seq_length=hps.dataset.max_seq_length
            )
            val_dataset_balanced = URbertDataset(
                df=val_df_balanced,
                tokenizer=tokenizer,
                text_mlm_on=hps.text_mlm_on,
                audio_distill_on=hps.audio_distill_on,
                max_seq_length=hps.dataset.max_seq_length
            )
            
            val_batch_size = hps.validation.batch_size if hasattr(hps.validation, 'batch_size') else hps.batch_size
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=lambda batch: custom_collator(batch, tokenizer),
                num_workers=4,
                pin_memory=True
            )
            val_dataloader_balanced = DataLoader(
                val_dataset_balanced,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=lambda batch: custom_collator(batch, tokenizer),
                num_workers=4,
                pin_memory=True
            )
            if accelerator.is_main_process and logger:
                logger.info("Validation Dataloaders Ready")

    # 6. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=hps.optimizer.init_learning_rate,
        weight_decay=hps.optimizer.weight_decay,
    )
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    
    # Modified to tri-stage scheduler
    # Steps were scaled to match the step of the optimizer
    # In accelerate, the step of the scheduler is multiplied by the number of GPUs
    # https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates
    total_steps = hps.num_steps * accelerator.num_processes
    warmup_steps = int(hps.optimizer.warmup_ratio * total_steps)
    hold_steps = int(hps.optimizer.hold_ratio * total_steps)
    decay_steps = (total_steps - warmup_steps - hold_steps)
    scheduler = TriStageLRScheduler(
        optimizer,
        init_lr=hps.optimizer.init_learning_rate,
        peak_lr=hps.optimizer.peak_learning_rate,
        final_lr=hps.optimizer.final_learning_rate,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        decay_steps=decay_steps,
        total_steps=total_steps
    )
    
    # To check the step scaling for scheduler
    if accelerator.is_main_process:
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Number of steps on config: {hps.num_steps}")
        print(f"-" * 30)
        print(f"Scaled warmup steps: {warmup_steps}")
        print(f"Scaled hold steps: {hold_steps}")
        print(f"Scaled decay steps: {decay_steps}")
        print(f"Scaled total steps: {total_steps}")  
        print(f"-" * 30)

    # 7. Accelerate Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    if val_enabled:
        val_dataloader, val_dataloader_balanced = accelerator.prepare(
            val_dataloader, val_dataloader_balanced
        )

    # 8. Resume Handling
    global_step = 0
    if args.resume:
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        if args.resume == 'latest':
            ckpt_path = latest_checkpoint_path(checkpoint_dir)
        else:
            ckpt_path = args.resume
        
        if ckpt_path:
            if accelerator.is_main_process and logger:
                logger.info(f"Loading checkpoint: {ckpt_path}")
            
            unwrapped_model = accelerator.unwrap_model(model)
            global_step = load_checkpoint(ckpt_path, unwrapped_model, optimizer, scheduler)
        else:
            if accelerator.is_main_process and logger:
                logger.warning("No checkpoint found. Starting from 0.")

    # 9. Run Training Loop
    train_and_evaluate(
        model, dataloader, optimizer, scheduler, accelerator, hps, 
        model_dir, global_step, logger, val_dataloader, val_dataloader_balanced
    )

def main():
    parser = argparse.ArgumentParser(description="URBERT Multi-Task Training")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint path or 'latest'")
    
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"\n[Launcher] Detected {num_gpus} GPUs.")
    
    if num_gpus > 1:
        print(f"[Launcher] Starting distributed training on {num_gpus} GPUs...")
        notebook_launcher(training_function, (args,), num_processes=num_gpus)
    else:
        print("[Launcher] Starting single-process training...")
        training_function(args)

if __name__ == "__main__":
    main()
