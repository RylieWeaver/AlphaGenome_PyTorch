import torch

from alphagenome_pytorch.schemas import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_organism_index,
    random_dna_batch,
)


def test_contact_maps_head():
    heads = {
        "contact_maps": {
            "num_tracks": [3, 2],
        }
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.input_seq_len
    pair_len = seq_len // 2048
    organism_index = make_organism_index(batch_size, num_organisms=2)

    num_tracks = max(heads["contact_maps"]["num_tracks"])
    contact_maps = torch.rand(batch_size, pair_len, pair_len, num_tracks)
    contact_maps_mask = metadata.metadata["heads"]["contact_maps"]["track_mask"][organism_index]
    contact_maps_mask = contact_maps_mask.view(batch_size, 1, 1, num_tracks)

    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        contact_maps=contact_maps,
        contact_maps_mask=contact_maps_mask,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "contact_maps" in predictions
    assert predictions["contact_maps"]["predictions"].shape == contact_maps.shape

