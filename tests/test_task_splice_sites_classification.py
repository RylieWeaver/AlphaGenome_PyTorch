import torch

from alphagenome_pt import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_organism_index,
    make_splice_sites,
    random_dna_batch,
)


def test_splice_sites_classification_head():
    heads = {
        "splice_sites_classification": {
            "num_tracks": [5, 5],
        }
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.max_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)
    num_classes = max(heads["splice_sites_classification"]["num_tracks"])

    splice_sites = make_splice_sites(batch_size, seq_len, num_classes)
    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        splice_sites=splice_sites,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "splice_sites_classification" in predictions
    assert predictions["splice_sites_classification"]["predictions"].shape == splice_sites.shape
