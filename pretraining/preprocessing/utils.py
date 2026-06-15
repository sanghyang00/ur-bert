import json, torch

def load_dictionary(dict_path):
    with open(dict_path, 'r') as file:
        dictionary = json.load(file)
        
    return dictionary

def flip_dictionary(dictionary):
    
    flipped_dictionary = {v: k for k, v in dictionary.items()}
    
    return flipped_dictionary


def compute_output_length(length, kernel_sizes=[10,3,3,3,3,2,2], strides=[5,2,2,2,2,2,2]):
    for k, s in zip(kernel_sizes, strides):
        length = torch.div(length - k, s, rounding_mode="floor") + 1
        length = torch.max(torch.zeros_like(length), length)
    return length

def normalize_speech(x):
    max_value = torch.max(torch.abs(x))

    normalized_x = x / max_value
    
    return normalized_x

def load_json(path, reverse=False):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data
    
def len_to_mask(tensors, lengths):
    return (torch.arange(tensors.size(1), device=tensors.device)[None, :] < lengths[:, None]).long()
