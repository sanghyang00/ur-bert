# dataloader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional
from tokenizer import UromanCharTokenizer


def URbertDataset(
    df: Optional[pd.DataFrame] = None,           
    csv_path: Optional[str] = None,               
    tokenizer: UromanCharTokenizer = None,
    text_mlm_on: bool = True,
    audio_distill_on: bool = True,
    mlm_probability: float = 0.15,
    mlm_mask_ratio: float = 0.8,
    mlm_random_ratio: float = 0.1,
    max_seq_length: int = 512
):

    if df is None and csv_path is None:
        raise ValueError("Either 'df' or 'csv_path' must be provided")

    if df is not None:
        data_df = df
    else:
        required_cols = ['language']
        if text_mlm_on:
            required_cols.append("romanized transcription")
        if audio_distill_on:
            required_cols.append("acoustic_id")

        data_df = pd.read_csv(csv_path, usecols=required_cols)

    data_df = data_df.dropna(subset=["romanized transcription"]).reset_index(drop=True)

    data = data_df.to_dict("records")

    class DatasetImpl(Dataset):
        def __init__(self):
            self.tokenizer = tokenizer
            self.data = data
            self.text_mlm_on = text_mlm_on
            self.audio_distill_on = audio_distill_on
            self.mlm_probability = mlm_probability
            self.mlm_mask_ratio = mlm_mask_ratio
            self.mlm_random_ratio = mlm_random_ratio
            self.max_seq_length = max_seq_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx) -> Dict:
            item = self.data[idx]
            result = {}

            if self.text_mlm_on:
                text = item["romanized transcription"]
                tokens = self.tokenizer.encode(text)
                input_ids = torch.tensor(tokens, dtype=torch.long)

                labels = input_ids.clone()

                prob_matrix = torch.full(labels.shape, self.mlm_probability)
                masked_indices = torch.bernoulli(prob_matrix).bool()

                labels[~masked_indices] = -100

                indices_replaced = (
                    torch.bernoulli(torch.full(labels.shape, self.mlm_mask_ratio)).bool()
                    & masked_indices
                )
                input_ids[indices_replaced] = self.tokenizer.mask_token_id

                indices_random = (
                    torch.bernoulli(torch.full(labels.shape, self.mlm_random_ratio / (1 - self.mlm_mask_ratio)))
                    .bool()
                    & masked_indices
                    & ~indices_replaced
                )
                random_words = torch.randint(
                    len(self.tokenizer.vocab), labels.shape, dtype=torch.long
                )
                input_ids[indices_random] = random_words[indices_random]

                result["uroman_ids"] = input_ids
                result["uroman_labels"] = labels
                result["seq_len"] = len(tokens)
                
                if result["seq_len"] > self.max_seq_length:
                    result["uroman_ids"] = result["uroman_ids"][:self.max_seq_length]
                    result["uroman_labels"] = result["uroman_labels"][:self.max_seq_length]
                    result["seq_len"] = self.max_seq_length

            if self.audio_distill_on:
                    audio_str = item.get("acoustic_id")
                    
                    seq_len = result.get("seq_len", 0)
                    default_audio = torch.full((seq_len,), -100, dtype=torch.long) if seq_len > 0 else torch.empty((0,), dtype=torch.long)

                    if not (audio_str and isinstance(audio_str, str) and audio_str.strip()):
                        result["audio_ids"] = default_audio
                        return result

                    try:
                        audio_ids = [int(t.strip()) for t in audio_str.split(',') if t.strip().isdigit()]
                        
                        if not audio_ids:
                            result["audio_ids"] = default_audio
                            return result

                        audio_tensor = torch.tensor(audio_ids, dtype=torch.long)
                        if len(audio_tensor) > self.max_seq_length:
                            audio_tensor = audio_tensor[:self.max_seq_length]
                            
                        if self.text_mlm_on and seq_len > 0:
                            if seq_len != len(audio_tensor):
                                print(f"Length mismatch at {idx}: text={seq_len}, audio={len(audio_tensor)}")
                                result["audio_ids"] = default_audio
                            else:
                                result["audio_ids"] = audio_tensor
                        else:
                            result["audio_ids"] = audio_tensor

                    except Exception as e:
                        print(f"Parse failed at {idx}: {e}")
                        result["audio_ids"] = default_audio

            return result

    return DatasetImpl()


def custom_collator(batch: List[Dict], tokenizer) -> Dict:
    if not batch:
        return {}

    seq_lens = [item.get("seq_len", 0) for item in batch]
    max_len = max(seq_lens) if seq_lens else 0

    batch_size = len(batch)

    uroman_ids = None
    uroman_labels = None
    attn_mask = None
    audio_ids = None

    if max_len > 0:
        uroman_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        uroman_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    has_audio = any("audio_ids" in item for item in batch)
    if has_audio:
        audio_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, sample in enumerate(batch):
        if "uroman_ids" in sample:
            length = sample["seq_len"]
            uroman_ids[i, :length] = sample["uroman_ids"]
            uroman_labels[i, :length] = sample["uroman_labels"]
            attn_mask[i, :length] = 1

        if has_audio and "audio_ids" in sample:
            length = len(sample["audio_ids"])
            audio_ids[i, :length] = sample["audio_ids"]

    return {
        "uroman_ids": uroman_ids,
        "attn_mask": attn_mask,   
        "uroman_labels": uroman_labels,
        "audio_ids": audio_ids,
    }
