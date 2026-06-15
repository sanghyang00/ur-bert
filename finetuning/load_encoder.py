import os
import yaml
import torch
import logging
from transformers import VitsConfig, AlbertConfig, AlbertModel, BertConfig, BertModel, AutoModel, AutoTokenizer

from vits_hf import PlainVitsEncoder
from tokenizer import VITSTokenizer, PLBERTTokenizer, XPhoneBERTTokenizer, UromanCharTokenizer
from urbert import MultiTaskBert
from utils import read_yaml_config        

# Plain VITS
def load_vits(config_path):
    
    tokenizer = VITSTokenizer()
    
    config = VitsConfig.from_pretrained(config_path)
    encoder = PlainVitsEncoder(config)
    
    return encoder, tokenizer

# PLBERT
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state

def load_plbert(model_dir):
    
    tokenizer = PLBERTTokenizer()
    
    config_path = os.path.join(model_dir, 'config.yml')
    ckpt_path = os.path.join(model_dir, 'step_1100000.t7')
    
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    model = CustomAlbert(albert_base_configuration)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    try:
        del new_state_dict["embeddings.position_ids"]
    except KeyError:
        pass
    model.load_state_dict(new_state_dict, strict=True)
    
    return model, tokenizer

# XPhoneBERT
def load_xphonebert(model_dir):
    
    tokenizer = XPhoneBERTTokenizer()
    model = AutoModel.from_pretrained(model_dir)

    return model, tokenizer

# URBERT
def load_urbert(
    model_dir: str,
    strict_load: bool = False
) -> torch.nn.Module:
    
    tokenizer = UromanCharTokenizer()
    
    config_path = os.path.join(model_dir, "config.yaml")
    weights_path = os.path.join(model_dir, "checkpoints", "urbert_latest.pth")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    hps = read_yaml_config(config_path)

    bert_config = BertConfig.from_pretrained(hps.model.base_model)
    bert_config.pad_token_id = tokenizer.pad_token_id
    assert bert_config.pad_token_id == tokenizer.pad_token_id
    
    bert = BertModel(bert_config)
    bert.resize_token_embeddings(hps.model.vocab_size) # Keypoint: Further considerations to avoid shape mismatch on finetuning
    
    full_model = MultiTaskBert(
        model=bert,
        uroman_vocab_size=hps.model.vocab_size,
        acoustic_vocab_size=hps.model.acoustic_vocab_size
    )

    loaded = torch.load(weights_path, map_location='cpu')

    state_dict = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded

    new_state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    full_model.load_state_dict(new_state_dict, strict=True)
    model = full_model.bert
    
    assert model.embeddings.word_embeddings.weight[tokenizer.pad_token_id].sum().item() == 0, \
    "Pad token embedding should be zero: Might be caused by configuration mismatch or vocab size mismatch"

    return model, tokenizer

def load_encoder(model_type, model_dir):
    if model_type == 'vits':
        return load_vits(model_dir)
    elif model_type == 'plbert':
        return load_plbert(model_dir)
    elif model_type == 'xphonebert':
        return load_xphonebert(model_dir)
    elif model_type == 'urbert':
        return load_urbert(model_dir)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
