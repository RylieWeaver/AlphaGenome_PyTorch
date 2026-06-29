# External
import torch

from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_scalars


def test_contact_maps_head():
    metadata = synthetic_metadata((HeadName.CONTACT_MAPS,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "contact_maps" in predictions
    assert (
        predictions["contact_maps"]["predictions"].shape
        == batch.contact_maps.shape
    )
