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


def test_masked_language_modeling_head():
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert "masked_language_modeling" in result.predictions
    assert result.predictions["masked_language_modeling"]["predictions"].shape == (
        batch.mlm.shape[0],
        batch.mlm.shape[1],
        5,
    )
