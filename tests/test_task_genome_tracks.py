import pytest
import torch

from alphagenome_pt import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_means,
    make_organism_index,
    make_poisson_tracks,
    random_dna_batch,
)


@pytest.mark.parametrize(
    ("head_name", "num_tracks", "resolution"),
    [
        ("rna_seq", [8, 5], 1),
        ("cage", [6, 0], 1),
        ("atac", [0, 6], 1),
        ("dnase", [5, 3], 1),
        ("procap", [4, 2], 1),
        ("chip_tf", [6, 4], 128),
        ("chip_histone", [5, 2], 128),
    ],
)
def test_individual_genome_track_head(head_name: str, num_tracks: list[int], resolution: int):
    heads = {
        head_name: {
            "num_tracks": num_tracks,
            "means": make_means(num_tracks),
        }
    }
    metadata = build_metadata(heads)
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.max_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)

    target_len = seq_len if resolution == 1 else seq_len // resolution
    means = metadata.metadata["heads"][head_name]["means"]
    tracks = make_poisson_tracks(means, organism_index, target_len)
    track_mask = metadata.metadata["heads"][head_name]["track_mask"][organism_index].unsqueeze(1)

    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        **{
            head_name: tracks,
            f"{head_name}_mask": track_mask,
        },
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert head_name in predictions
    assert predictions[head_name][f"scaled_predictions_{resolution}bp"].shape == tracks.shape


def test_min_zero_multinomial_loss_config_reaches_genome_track_head():
    metadata = build_metadata(
        {
            "rna_seq": {
                "num_tracks": [2, 2],
                "means": make_means([2, 2]),
            }
        }
    )
    model = build_small_model(metadata, min_zero_multinomial_loss=False)

    assert model._heads["rna_seq"]._min_zero_multinomial_loss is False
