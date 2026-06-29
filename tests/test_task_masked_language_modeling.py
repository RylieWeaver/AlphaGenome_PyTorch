# External
import torch

from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_scalars


def test_masked_language_modeling_head():
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "masked_language_modeling" in predictions
    assert predictions["masked_language_modeling"]["predictions"].shape == (
        batch.mlm.shape[0],
        batch.mlm.shape[1],
        5,
    )
