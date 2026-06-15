import os, json, torch, torchaudio, argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchaudio.pipelines import MMS_FA as bundle
from tqdm import tqdm
from dataset import AlignmentDataset

import warnings
warnings.filterwarnings('ignore')

MODEL = bundle.get_model()
TOKENIZER = bundle.get_tokenizer()
ALIGNER = bundle.get_aligner()
ID_TO_CHAR = {i: c for c, i in TOKENIZER.dictionary.items()}

def compute_alignments(waveform, transcript):
    with torch.inference_mode():
        emission, _ = MODEL(waveform)
        token_spans = ALIGNER(emission[0], TOKENIZER(transcript))
    return emission, token_spans

def get_alignment_map(waveform, transcripts):
    emission, token_spans = compute_alignments(waveform, transcripts)
    
    alignment_map = []
    for i, word_spans in enumerate(token_spans):
        for span in word_spans:
            char = ID_TO_CHAR[span.token]
            alignment_map.append({char: (span.start, span.end)})
        
        if i < len(token_spans) - 1:
            alignment_map.append({'_': (word_spans[-1].end, token_spans[i + 1][0].start)})
    
    return alignment_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--save_dir", type=str, default='alignment_maps')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--flush_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()  
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Preparing Alignment Model...")
    MODEL.to(args.device)
    MODEL.eval()
    
    save_path = os.path.join(args.save_dir, f'{args.split}.jsonl')
    
    dataset_list = [os.path.join(args.data_dir, p) for p in os.listdir(args.data_dir)]
    full_dataset = AlignmentDataset(dataset_list, args.split, duration_limit=30.0)
    
    start_idx = 0
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            start_idx = sum(1 for line in f if line.strip())
    
    if start_idx > 0:
        print(f"Resuming from index {start_idx} ({start_idx}/{len(full_dataset)} already processed)")
        dataset = Subset(full_dataset, range(start_idx, len(full_dataset)))
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=args.num_workers, 
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    with open(save_path, 'a', encoding='utf-8') as f:
        for i, (fp, y, roman) in tqdm(enumerate(dataloader), total=len(dataset)):
    
            fp, roman = fp[0], roman[0]
            
            try:
                transcripts = roman.split()
                alignment_map = get_alignment_map(y.to(args.device), transcripts)

                record = {
                    'file path': fp,
                    'transcription': roman,
                    'alignment': alignment_map
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                if i % args.flush_interval == 0:
                    f.flush()
                
            except Exception as e:
                print(f"Error processing {fp}: {e}")
                continue
    
    print(f"\nProcessing complete.")

if __name__ == "__main__":
    main()

