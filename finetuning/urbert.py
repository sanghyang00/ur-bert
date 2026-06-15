import torch
from torch import nn
from typing import Optional, Tuple

class MultiTaskBert(nn.Module):
    def __init__(self, model, uroman_vocab_size: int, acoustic_vocab_size: int):
        super().__init__()
        
        self.bert = model
        
        # Head 1: TextMLM
        self.text_mlm_head = nn.Linear(self.bert.config.hidden_size, uroman_vocab_size)
        
        # Head 2: AudioDistill
        self.audio_distill_head = nn.Linear(self.bert.config.hidden_size, acoustic_vocab_size)

    def forward(
        self,
        uroman_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        text_mlm_on: bool = True,
        audio_distill_on: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        outputs = self.bert(input_ids=uroman_ids, attention_mask=attn_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, hidden_size)
        # sequence_output = self.dropout(sequence_output)

        text_mlm_logits = None
        if text_mlm_on:
            text_mlm_logits = self.text_mlm_head(sequence_output)  # (B, L, uroman_vocab_size)

        audio_distill_logits = None
        if audio_distill_on:
            audio_distill_logits = self.audio_distill_head(sequence_output)  # (B, L, acoustic_vocab_size)

        return text_mlm_logits, audio_distill_logits