import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state

def load_plbert(bert):
    config_path = os.path.join(bert, 'config.yml')
    ckpt_path = os.path.join(bert, 'step_1100000.t7')
    
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

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
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
