# External
import torch

# Internal
from alphagenome_pt import DNAOneHotEncoder


def test_dna_one_hot_encoder_returns_tensors():
    encoder = DNAOneHotEncoder()
    encoded = encoder.encode("aCgTn")
    assert isinstance(encoded, torch.Tensor)
