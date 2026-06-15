# utils.py
import os
import torch
import glob
import yaml
import logging
from pathlib import Path
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_logger(model_dir: str, filename="train.log"):
    
    logger = logging.getLogger("urbert_train")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)5s | %(message)s")
    
    log_file = os.path.join(model_dir, filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return str(self.__dict__)

    def update(self, other):
        if isinstance(other, dict):
            other = HParams(**other)
        for k, v in other.__dict__.items():
            if isinstance(v, HParams):
                if k in self.__dict__:
                    self.__dict__[k].update(v)
                else:
                    self.__dict__[k] = v
            else:
                self.__dict__[k] = v


def get_hparams(config_path: str, model_dir: Optional[str] = None) -> HParams:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    hps = HParams(**config_dict)

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        config_save_path = model_dir / "config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Config copied to {config_save_path}")

    return hps

def get_combined_df_from_split(
    data_dir: str,
    split: str = 'train',
    logger=None,
    filename_pattern: str = None,
    required_cols=None
):
    if filename_pattern is None:
        filename_pattern = f"{split}.csv"
    else:
        filename_pattern = filename_pattern.format(split=split)

    abs_data_dir = os.path.abspath(data_dir)
    if logger:
        logger.info(f"Loading '{filename_pattern}' files from {abs_data_dir}")

    all_dfs = []
    loaded_files = []

    usecols = required_cols if required_cols else None

    for root, dirs, files in os.walk(abs_data_dir):
        for file in files:
            if file.lower() == filename_pattern.lower():
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(
                        csv_path,
                        usecols=usecols,
                        on_bad_lines='warn'
                    )
                    corpus_name = os.path.basename(root.rstrip('/'))
                    df['corpus'] = corpus_name
                    all_dfs.append(df)
                    loaded_files.append(csv_path)
                    if logger:
                        logger.info(f"  Loaded {len(df):,} rows from {csv_path} (corpus: {corpus_name})")
                except Exception as e:
                    if logger:
                        logger.warning(f"  Failed to load {csv_path}: {e}")

    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['romanized transcription']).reset_index(drop=True)
    
    return combined_df

def preprocess_acoustic_id(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    
    if 'acoustic_id' not in df.columns:
        if logger:
            logger.warning("No 'acoustic_id' column found. Skipping preprocessing.")
        return df

    def parse_acoustic_id(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [int(i) for i in x if str(i).strip().isdigit()]
        if isinstance(x, str):
            try:
                return [int(i.strip()) for i in x.split(',') if i.strip().isdigit()]
            except (ValueError, TypeError) as e:
                if logger:
                    logger.warning(f"Failed to parse acoustic_id: {x[:50]}... | Error: {e}")
                return []
        return []

    original_rows = len(df)
    df['acoustic_id'] = df['acoustic_id'].apply(parse_acoustic_id)
    
    empty_after = df['acoustic_id'].apply(len).eq(0).sum()
    if logger and empty_after > 0:
        logger.info(f"Preprocessed acoustic_id | {empty_after:,} rows became empty list "
                    f"({empty_after/original_rows*100:.1f}% of total)")

    return df


def save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, prefix="urbert"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pth") # urbert_latest.pth
    
    state = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
    }
    
    torch.save(state, checkpoint_path)
    logger.info(f"Updated latest checkpoint at step {global_step}: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None
):
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0
    
    state = torch.load(checkpoint_path, map_location='cpu')
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(state['model'])
    else:
        model.load_state_dict(state['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    
    global_step = state.get('global_step', 0)
    logger.info(f"Loaded checkpoint from {checkpoint_path} at step {global_step}")
    return global_step

def latest_checkpoint_path(checkpoint_dir, prefix="urbert"):
    candidate = os.path.join(checkpoint_dir, f"{prefix}_latest.pth")
    if os.path.exists(candidate):
        return candidate
    return None
