import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_token_chunk(chunk):
    processed_ids = []
    for _, row in chunk.iterrows():
        text = str(row['romanized transcription'])
        tokens = str(row['acoustic_id']).split()
        chars = list(text)
        
        ln = min(len(chars), len(tokens))
        assert ln == len(chars) == len(tokens)
        
        new_row_tokens = [
            "0" if chars[i] == ' ' else str(int(tokens[i]) + 1) 
            for i in range(ln)
        ]
        processed_ids.append(",".join(new_row_tokens))
    
    chunk['acoustic_id'] = processed_ids
    return chunk[['file path', 'acoustic_id']]

def parallel_process_tokens(df, num_workers=None):  
    if num_workers is None:
        num_workers = cpu_count()
        
    chunks = np.array_split(df, num_workers)
    
    print(f"Starting parallel processing with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_token_chunk, chunks), total=num_workers, desc="Processing Token Mapping"))
    
    return pd.concat(results)

def main():
    token_files = {
        'train': '/ssd2/sangmin/urbert/preprocess_tokens/audio_tokens/token_mapping_L16_C256.txt',
        'dev': '/ssd2/sangmin/urbert/preprocess_tokens/audio_tokens_dev/token_mapping_L16_C256.txt'
    }
    
    csvs_root = '/ssd2/sangmin/urbert/urbert_training/csvs'
    save_root = '/ssd2/sangmin/urbert/urbert_training/data'
    
    subdirs = ['fleurs', 'commonvoice', 'omnilingual_asr_corpus']

    for split in ['train', 'dev']:
        print(f"\n{'#'*70}")
        print(f" PHASE: Processing {split.upper()} Data")
        print(f"{'#'*70}")

        if not os.path.exists(token_files[split]):
            print(f"Skip: {token_files[split]} not found.")
            continue
            
        print(f"Loading {split} token mapping file...")
        raw_token_df = pd.read_csv(
            token_files[split], sep='|', 
            names=['file path', 'romanized transcription', 'acoustic_id'],
            low_memory=False
        )
        
        processed_mapping = parallel_process_tokens(raw_token_df)
        del raw_token_df 

        for subdir in subdirs:
            input_csv_path = os.path.join(csvs_root, subdir, f'{split}.csv')
            
            output_dir = os.path.join(save_root, subdir)
            os.makedirs(output_dir, exist_ok=True)
            output_csv_path = os.path.join(output_dir, f'{split}.csv')

            if not os.path.exists(input_csv_path):
                continue

            print(f"\n--- Processing {subdir}/{split}.csv ---")
            original_df = pd.read_csv(input_csv_path)
            orig_count = len(original_df)

            if 'acoustic_id' in original_df.columns:
                original_df = original_df.drop(columns=['acoustic_id'])

            merged_df = pd.merge(original_df, processed_mapping, on='file path', how='inner')
            final_count = len(merged_df)
            pruned_count = orig_count - final_count

            print(f"  > Original Rows: {orig_count:,}")
            print(f"  > Final Rows:    {final_count:,}")
            if pruned_count > 0:
                print(f"  > Pruned Rows:   {pruned_count:,} ({(pruned_count/orig_count)*100:.2f}% removed)")

            merged_df.to_csv(output_csv_path, index=False, quoting=1)
            print(f"  > Saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
