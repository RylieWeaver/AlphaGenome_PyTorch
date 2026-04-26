import torch

from alphagenome_pt import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_organism_index,
    random_dna_batch,
)


def test_splice_sites_usage_head():
    heads = {
        "splice_sites_usage": {
            "num_tracks": [8, 4],
        }
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.max_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)
    num_tracks = max(heads["splice_sites_usage"]["num_tracks"])

    splice_site_usage = torch.rand(batch_size, seq_len, num_tracks)
    usage_mask = metadata.metadata["heads"]["splice_sites_usage"]["track_mask"][organism_index].unsqueeze(1)

    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        splice_site_usage=splice_site_usage,
        splice_site_usage_mask=usage_mask,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "splice_sites_usage" in predictions
    assert predictions["splice_sites_usage"]["predictions"].shape == splice_site_usage.shape
