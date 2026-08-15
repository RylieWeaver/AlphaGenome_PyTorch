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


def test_contact_maps_head():
    metadata = synthetic_metadata((HeadName.CONTACT_MAPS,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert "contact_maps" in result.predictions
    assert (
        result.predictions["contact_maps"]["predictions"].shape
        == batch.contact_maps.shape
    )
