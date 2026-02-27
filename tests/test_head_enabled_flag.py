import torch

from alphagenome_pytorch.schemas import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_means,
    make_organism_index,
    make_poisson_tracks,
    random_dna_batch,
)


def test_enabled_flag_skips_disabled_heads():
    heads = {
        "rna_seq": {
            "num_tracks": [6, 4],
            "means": make_means([6, 4]),
            "enabled": False,
        },
        "atac": {
            "num_tracks": [5, 5],
            "means": make_means([5, 5]),
            "enabled": True,
        },
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.input_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)

    atac = make_poisson_tracks(heads["atac"]["means"], organism_index, seq_len)
    atac_mask = metadata.metadata["heads"]["atac"]["track_mask"][organism_index].unsqueeze(1)

    # rna_seq is intentionally omitted from batch because this head is disabled.
    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        atac=atac,
        atac_mask=atac_mask,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "atac" in predictions
    assert "rna_seq" not in predictions
    assert all(not key.startswith("rna_seq_") for key in scalars)

