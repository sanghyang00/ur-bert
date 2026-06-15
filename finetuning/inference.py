import os
import re
import json
import math
import torch
import torchaudio
import unicodedata
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import commons
import utils
from load_encoder import load_encoder
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import Generator
from tqdm import tqdm
import argparse

import torch
import random
import os

import warnings
warnings.filterwarnings("ignore")

CKPT_STEP_PER_LANG = {
    'en': 300000,
    'de': 300000,
    'zh': 300000,
    'af': 100000,
    'tn': 100000,
    'xh': 100000,
    'si': 100000,
    'jv': 100000,
    'su': 100000,
    'km': 100000,
    'np': 100000,
}

class TestDataset(Dataset):
    def __init__(self, audiopaths_sid_text, hparams, tokenizer):
        self.audiopaths_sid_text = utils.load_filepaths_and_text(audiopaths_sid_text)
        
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 500)
        self.tokenizer = tokenizer
        
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        filtered_data = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                filtered_data.append([audiopath, sid, text])
        self.audiopaths_sid_text = filtered_data

    def get_text(self, text):
        tokenized = self.tokenizer(text)
        input_ids = torch.LongTensor(tokenized['input_ids'])
        attention_mask = torch.LongTensor(tokenized['attention_mask'])
        return input_ids, attention_mask

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def __getitem__(self, index):
        audiopath, sid_raw, text = self.audiopaths_sid_text[index]
        
        input_ids, attention_mask = self.get_text(text)
        sid = self.get_sid(sid_raw)
        
        return audiopath, input_ids, attention_mask, sid

    def __len__(self):
        return len(self.audiopaths_sid_text)
    
def normalize_text(text):
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    return text

def main():
    
    parser = argparse.ArgumentParser(description="Multi-Speaker TTS Inference")
    parser.add_argument('--logs_dir', type=str, required=True, help="Path to logs directory (Consists of config.json and checkpoint.pth)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for WAV files")
    parser.add_argument('--target_experiments', type=str, nargs='+', default=None, help="Target experiment(s) to process")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for inference")
    parser.add_argument('--predefined_training_step', action='store_true', default=False, help="Fixed training step to use for inference")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.target_experiments is None:
        print("No target experiment specified. Processing all experiments in the logs directory.")
        target_experiments = sorted(os.listdir(args.logs_dir))
    else:
        print(f"Processing target experiment(s): {args.target_experiments}")
        target_experiments = args.target_experiments
        
    for exp in tqdm(target_experiments, total=len(target_experiments), desc="Processing experiments", leave=True):
        
        processing_dir = os.path.join(args.logs_dir, exp)
        
        print(f"Building model for experiment: {exp}")
        config_path = os.path.join(processing_dir, "config.json")
        
        hps = utils.get_hparams_from_file(config_path)
        
        lang = exp.split("-")[-1].lower()
        
        if args.predefined_training_step:
            checkpoint_paths = [os.path.join(processing_dir, f"G_{CKPT_STEP_PER_LANG[lang]}.pth")]
        else:
            checkpoint_paths = [
            os.path.join(processing_dir, f)
            for f in os.listdir(processing_dir)
            if f.startswith("G_") and f.endswith(".pth")
            ]
        
        bert_type = exp.split("-")[0].lower()
        backbone, tokenizer = load_encoder(bert_type, hps.bert)
        
        tts_model = Generator(
            backbone,
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model 
        ).to(args.device)
        tts_model.eval()
        
        print(f"Building dataloader for experiment: {exp}")
        
        test_dataset = TestDataset(hps.data.validation_files, hps.data, tokenizer)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False, # To ensure the same order of the metadata
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        
        for checkpoint_path in checkpoint_paths:
            
            checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
            save_dir = os.path.join(args.output_dir, exp, checkpoint_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"Loading checkpoint: {checkpoint_path}")
            utils.load_checkpoint(checkpoint_path, tts_model, None)
            print(f"Checkpoint loaded: {checkpoint_path}")
        
            print(f"Creating output directories for experiment: {exp}")
            for folder_name in ['generated_wav', 'attention_map']:
                os.makedirs(os.path.join(save_dir, folder_name), exist_ok=True)
            
            print(f"Starting inference for experiment: {exp}")
            for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference", leave=False):
                # x, attention_mask, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
                audiopath, x, attention_mask, speakers = batch
                
                audiopath = audiopath[0] # Assume inference with batch size 1
                sid = speakers.squeeze(0) # Assume inference with batch size 1
                x, attention_mask, sid = x.to(args.device), attention_mask.to(args.device), sid.to(args.device)
                
                with torch.no_grad():
                    y_hat, attention_map, _, _ = tts_model.infer(x, attention_mask, sid=sid)
                
                generated_wav = y_hat.squeeze().cpu()
                attention_map = attention_map.squeeze().cpu()
                
                audioname = os.path.basename(audiopath).split('.')[0]
                generated_wav_path = os.path.join(save_dir, 'generated_wav', f'{audioname}.wav')
                attention_map_path = os.path.join(save_dir, 'attention_map', f'{audioname}.pt')
                
                torchaudio.save(generated_wav_path, generated_wav, hps.data.sampling_rate, 
                                encoding="PCM_S", bits_per_sample=16) # Assert 16bit PCM audio format
                torch.save(attention_map, attention_map_path)
            
if __name__ == "__main__":
    main()
    