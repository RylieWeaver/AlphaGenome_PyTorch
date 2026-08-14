# External
import pytest
import torch

# Internal
from alphagenome_pt import HeadName, small_alphagenome, synthetic_metadata

from .helpers import DNA_SEQUENCE


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = small_alphagenome(
        synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,)),
        max_seq_len=len(DNA_SEQUENCE),
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
    )
    model.eval()
    return model
