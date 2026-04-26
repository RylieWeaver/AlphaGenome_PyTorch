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


def test_splice_sites_junction_head():
    heads = {
        "splice_sites_classification": {
            "num_tracks": [5, 5],
        },
        "splice_sites_junction": {
            "num_tissues": [3, 2],
        },
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.max_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)
    num_classes = max(heads["splice_sites_classification"]["num_tracks"])
    num_tissues = max(heads["splice_sites_junction"]["num_tissues"])
    num_splice_sites = model.num_splice_sites

    splice_sites = make_splice_sites(batch_size, seq_len, num_classes)
    splice_site_junction = torch.poisson(
        4.0 * torch.ones(batch_size, num_splice_sites, num_splice_sites, 2 * num_tissues)
    ).to(torch.float32)

    # This head expects an organism-indexed mask in the current implementation.
    splice_site_junction_mask = metadata.metadata["heads"]["splice_sites_junction"]["tissue_mask"][organism_index].view(batch_size, 1, 1, num_tissues)

    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        splice_sites=splice_sites,
        splice_site_junction=splice_site_junction,
        splice_site_junction_mask=splice_site_junction_mask,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "splice_sites_junction" in predictions
    assert predictions["splice_sites_junction"]["predictions"].shape == splice_site_junction.shape
