import torch

from alphagenome_pt.schemas import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_organism_index,
    random_dna_batch,
)


def test_masked_language_modeling_head():
    metadata = build_metadata({"masked_language_modeling": {}})
    model = build_small_model(metadata)

    batch_size = 2
    seq_len = model.input_seq_len
    organism_index = make_organism_index(batch_size, num_organisms=2)
    labels = torch.randint(0, 5, (batch_size, seq_len), dtype=torch.long)

    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        mlm=labels,
    )

    total_loss, scalars, predictions = model.loss(batch)

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert "masked_language_modeling" in predictions
    assert predictions["masked_language_modeling"]["predictions"].shape == (batch_size, seq_len, 5)

