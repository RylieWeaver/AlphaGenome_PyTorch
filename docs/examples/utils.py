# General
import random
from typing import Optional, Union

# Torch
import torch
import torch.nn.functional as F

# AlphaGenome


    
def resolve_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    # NOTE: Only allow CPU training if not doing any type of parallelism
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def move_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to(v, device) for v in data]
    else:
        return data
    
def bert_mlm(
    sequences: list[str],
    mask_token: str = 'N',
    select_prob: float = 0.15,
    mask_prob: float = 0.8,
    random_prob = 0.1,
    keep_prob = 0.1
) -> tuple[list[str], list[str]]:
    """Applies BERT-style masked language modeling to the input DNA sequences."""
    masked_sequences = []
    labels = []
    for seq in sequences:
        masked_seq = list(seq)
        label_seq = [mask_token] * len(seq)
        for i in range(len(seq)):
            if random.random() < select_prob:
                rand_val = random.random()
                if rand_val < mask_prob:
                    # Mask the token
                    masked_seq[i] = mask_token
                elif rand_val < mask_prob + random_prob:
                    # Replace with a random token
                    masked_seq[i] = random.choice('ACGT')
                else:
                    # Keep the original token (but still predict it)
                    pass
                label_seq[i] = seq[i]  # Set the label to the original token
        masked_sequences.append("".join(masked_seq))
        labels.append("".join(label_seq))
    return masked_sequences, labels

def make_means(num_tracks: list[int], low=3.0, high=10.0):
    all_means = []
    N = max(num_tracks)
    for n in num_tracks:
        means = torch.empty(n).uniform_(low, high)
        means = F.pad(means, (0, N - n), value=1.0)
        all_means.append(means)
    return torch.stack(all_means, dim=0)
