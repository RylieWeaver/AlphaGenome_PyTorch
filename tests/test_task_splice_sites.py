# External
import torch

# Internal
from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_metric_tree


def test_splice_sites_head():
    metadata = synthetic_metadata((HeadName.SPLICE_SITES_CLASSIFICATION,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert "splice_sites_classification" in result.predictions
    assert (
        result.predictions["splice_sites_classification"]["predictions"].shape
        == batch.splice_sites.shape
    )
