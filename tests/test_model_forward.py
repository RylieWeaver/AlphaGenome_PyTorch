# External
import torch

from alphagenome_pt import (
    DataBatch,
    HeadName,
    small_alphagenome,
    synthetic_dna_sequence,
    synthetic_metadata,
)


def test_forward_uses_batch_organism_index():
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(metadata)
    model.eval()

    batch_size = 2
    seq_len = model.max_seq_len
    dna_sequence = synthetic_dna_sequence(batch_size, seq_len)

    batch_mixed = DataBatch(
        dna_sequence=dna_sequence,
        organism_index=torch.tensor([0, 1], dtype=torch.long),
    )
    batch_human = DataBatch(
        dna_sequence=dna_sequence,
        organism_index=torch.tensor([0, 0], dtype=torch.long),
    )

    with torch.no_grad():
        predictions, embeddings_mixed = model(batch_mixed)
        _, embeddings_human = model(batch_human)

    assert predictions["masked_language_modeling"]["predictions"].shape[:2] == (
        batch_size,
        seq_len,
    )
    assert embeddings_mixed.embeddings_1bp.shape[0] == batch_size
    assert embeddings_mixed.embeddings_128bp.shape[0] == batch_size
    assert embeddings_mixed.embeddings_pair.shape[0] == batch_size
    assert not torch.allclose(
        embeddings_mixed.embeddings_128bp[1],
        embeddings_human.embeddings_128bp[1],
    )


def test_forward_defaults_missing_organism_index_to_zero():
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(metadata)
    model.eval()

    batch_size = 2
    seq_len = model.max_seq_len
    dna_sequence = synthetic_dna_sequence(batch_size, seq_len)

    batch_missing = DataBatch(dna_sequence=dna_sequence)
    batch_zero = DataBatch(
        dna_sequence=dna_sequence,
        organism_index=torch.zeros(batch_size, dtype=torch.long),
    )

    with torch.no_grad():
        _, embeddings_missing = model(batch_missing)
        _, embeddings_zero = model(batch_zero)

    assert torch.allclose(
        embeddings_missing.embeddings_1bp,
        embeddings_zero.embeddings_1bp,
    )
    assert torch.allclose(
        embeddings_missing.embeddings_128bp,
        embeddings_zero.embeddings_128bp,
    )
    assert torch.allclose(
        embeddings_missing.embeddings_pair,
        embeddings_zero.embeddings_pair,
    )
