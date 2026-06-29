# External
import pytest
import torch

# Internal
from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_scalars


@pytest.mark.parametrize(
    ("head", "resolution"),
    [
        (HeadName.RNA_SEQ, 1),
        (HeadName.CAGE, 1),
        (HeadName.ATAC, 1),
        (HeadName.DNASE, 1),
        (HeadName.PROCAP, 1),
        (HeadName.CHIP_TF, 128),
        (HeadName.CHIP_HISTONE, 128),
    ],
)
def test_individual_genome_track_head(head: HeadName, resolution: int):
    head_name = head.value
    metadata = synthetic_metadata((head,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
    
    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert head_name in predictions
    assert (
        predictions[head_name][f"scaled_predictions_{resolution}bp"].shape
        == getattr(batch, head_name).shape
    )


def test_min_zero_multinomial_loss_config_reaches_genome_track_head():
    metadata = synthetic_metadata(
        (HeadName.RNA_SEQ,),
        num_tracks=2,
    )
    model = small_alphagenome(metadata, min_zero_multinomial_loss=False)

    assert model._heads["rna_seq"]._min_zero_multinomial_loss is False
