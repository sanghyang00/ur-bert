import pandas as pd
from pathlib import Path

def fix_paths(csv_path: str, backup: bool = True):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    if 'file path' not in df.columns:
        print(f"Warning: 'file path' column is missing -> {csv_path}")
        return
    
    if backup:
        backup_path = csv_path.with_suffix('.bak.csv')
        df.to_csv(backup_path, index=False)
        print(f"Backup created: {backup_path}")
    
    df['file path'] = df['file path'].str.replace(
        '/home/intern/datasets',
        '/workspace/datasets',
        regex=False
    )
    
    df.to_csv(csv_path, index=False)
    print(f"Updated successfully: {csv_path}")
    print(f"First updated path example: {df['file path'].iloc[0]}\n")

files = [
    #"/workspace/datasets/csvs/commonvoice/train.csv",
    #"/workspace/datasets/csvs/fleurs/train.csv",
    #"/workspace/datasets/csvs/omnilingual_asr_corpus/train.csv",
    
     "/workspace/datasets/csvs/commonvoice/dev.csv",
     "/workspace/datasets/csvs/fleurs/dev.csv",
     "/workspace/datasets/csvs/omnilingual_asr_corpus/dev.csv",
]

for f in files:
    if Path(f).exists():
        fix_paths(f, backup=True)
    else:
        print(f"File not found: {f}")
