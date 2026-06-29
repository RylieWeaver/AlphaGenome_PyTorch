# External
import torch

from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_scalars


def test_splice_site_usage_head():
    metadata = synthetic_metadata((HeadName.SPLICE_SITES_USAGE,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "splice_sites_usage" in predictions
    assert predictions["splice_sites_usage"]["predictions"].shape == batch.splice_site_usage.shape
