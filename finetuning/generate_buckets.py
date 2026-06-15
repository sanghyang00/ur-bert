import os
import torch
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Analyze spectrogram lengths and estimate buckets per language.")
    parser.add_argument('--spec_dir', type=str, default='/ssd2/sangmin/datasets/tts_db/trimmed_spec', 
                        help='Directory containing .spec.pt files')
    parser.add_argument('--metadata_dir', type=str, default='/ssd2/sangmin/datasets/tts_db/tts_metadata', 
                        help='Directory containing metadata csv files')
    parser.add_argument('--plot_dir', type=str, default='./plots', 
                        help='Directory to save distribution plots')
    parser.add_argument('--output_json', type=str, default='estimated_buckets.json', 
                        help='Output JSON file path for buckets')
    parser.add_argument('--samples_per_bucket', type=int, default=150, 
                        help='Target minimum samples per bucket for efficiency')
    return parser.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.plot_dir, exist_ok=True)
    
    langs = ['en', 'zh', 'de', 'af', 'jv', 'km', 'np', 'si', 'st', 'su', 'tn', 'xh']
    all_buckets = {}

    for lang in langs:
        metadata_path = os.path.join(args.metadata_dir, f'{lang}_train.csv')
        if not os.path.exists(metadata_path):
            print(f"⚠️ Skip: {metadata_path} not found.")
            continue

        print(f"\n[Processing '{lang}']...")
        df = pd.read_csv(metadata_path, sep='|')
        file_ids = df['id'].tolist()
        
        spec_lengths = []
        for f_id in tqdm(file_ids, desc=f"Loading {lang}"):
            spec_path = os.path.join(args.spec_dir, f"{f_id}.spec.pt")
            if os.path.exists(spec_path):
                spec = torch.load(spec_path, map_location='cpu', weights_only=True)
                spec_lengths.append(spec.shape[-1])
        
        if not spec_lengths:
            continue
            
        spec_lengths = np.array(spec_lengths)
        
        num_buckets = max(len(spec_lengths) // args.samples_per_bucket, 1)
        percentiles = np.linspace(0, 100, num_buckets + 1)
        
        raw_boundaries = np.percentile(spec_lengths, percentiles)
        bucket_boundaries = sorted(list(set([int(b) for b in raw_boundaries])))
        
        final_buckets = [b for b in bucket_boundaries if b > 32]
        max_val = int(np.max(spec_lengths))
        
        final_buckets.append(max_val + 100)
            
        all_buckets[lang] = [int(b) for b in final_buckets]
        
        plt.figure(figsize=(12, 6))
        plt.hist(spec_lengths, bins=100, color='#4A90E2', alpha=0.7, label='Data Distribution')
        for b in all_buckets[lang]:
            plt.axvline(b, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.title(f"Spectrogram Length Distribution & Buckets: {lang}")
        plt.xlabel("Frames")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.2)
        
        plot_path = os.path.join(args.plot_dir, f"distribution_{lang}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ {lang} Done. Buckets: {len(all_buckets[lang])}, Max: {max_val}")

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(all_buckets, f, indent=4)
    
    print(f"\n🚀 All process finished! Buckets saved to '{args.output_json}'")

if __name__ == "__main__":
    main()