# tokenizer.py

from typing import List
import torch

class UromanCharTokenizer:
    def __init__(self):
        base_chars = " abcdefghijklmnopqrstuvwxyz"
        
        self.vocab = {c: i for i, c in enumerate(base_chars)}
        
        self.special_tokens = ["[PAD]", "[UNK]", "[MASK]"]
        
        offset = len(self.vocab)
        for i, t in enumerate(self.special_tokens):
            self.vocab[t] = offset + i
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]
        self.mask_token_id = self.vocab["[MASK]"]


    def encode(self, text: str) -> List[int]:
        tokens = []
        for c in text:
            if c in self.vocab:
                tokens.append(self.vocab[c])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def __call__(self, text: str, **kwargs) -> dict:
    
        input_ids = self.encode(text)
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask) 
        }

    
def decode(self, ids: List[int], skip_special: bool = True) -> str:
    
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "[UNK]")
            if skip_special and token in {"[PAD]", "[UNK]", "[MASK]"}:
                continue
            tokens.append(token)
        return "".join(tokens)
