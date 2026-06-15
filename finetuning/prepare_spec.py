import os
import json
import argparse
import torch
import torchaudio
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial

from mel_processing import spectrogram_torch

import warnings
warnings.filterwarnings("ignore")

def process_one_file(wav_path, spec_dir, config_params):
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    spec_save_path = os.path.join(spec_dir, f"{file_name}.spec.pt")

    if os.path.exists(spec_save_path):
        return

    try:
        sr_target = config_params['sampling_rate']
        filter_length = config_params['filter_length']
        hop_length = config_params['hop_length']
        win_length = config_params['win_length']

        audio, sr = torchaudio.load(wav_path)
        
        assert sr == sr_target, f"Sampling rate mismatch: {sr} != {sr_target}"
            
        audio = audio.squeeze(0)
        
        audio_norm = audio # For low-resource setting
        audio_norm = audio_norm.unsqueeze(0)

        spec = spectrogram_torch(
            audio_norm, 
            filter_length,
            sr_target, 
            hop_length, 
            win_length,
            center=False
        )
        spec = torch.squeeze(spec, 0)

        torch.save(spec, spec_save_path)
        
    except Exception as e:
        print(f" ! Error processing {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Precompute spectrograms with Multiprocessing")
    parser.add_argument("--wav_dir", type=str, default="/ssd2/sangmin/datasets/tts_db/trimmed_wavs", help="Path to wav files (e.g., .../wavs)")
    parser.add_argument("--config_path", type=str, default="/ssd2/sangmin/urbert_baselines/unified_framework/configs/urbert/default.json", help="Path to config.json")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of CPU cores to use")

    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    data_cfg = config['data']
    spec_dir_name = data_cfg['spec_dir_name']
    
    parent_dir = os.path.dirname(os.path.abspath(args.wav_dir.rstrip("/")))
    spec_dir = os.path.join(parent_dir, spec_dir_name)
    os.makedirs(spec_dir, exist_ok=True)
    
    wav_files = glob(os.path.join(args.wav_dir, "*.wav"))
    print(f" > Found {len(wav_files)} files in {args.wav_dir}")
    print(f" > Saving spectrograms to: {spec_dir}")
    print(f" > Using {args.num_workers} workers for parallel processing")

    config_params = {
        'sampling_rate': data_cfg['sampling_rate'],
        'filter_length': data_cfg['filter_length'],
        'hop_length': data_cfg['hop_length'],
        'win_length': data_cfg['win_length']
    }
    
    worker_fn = partial(process_one_file, spec_dir=spec_dir, config_params=config_params)

    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap(worker_fn, wav_files), total=len(wav_files), desc="Parallel Processing"))

    print("Done!")

if __name__ == "__main__":
    main()
