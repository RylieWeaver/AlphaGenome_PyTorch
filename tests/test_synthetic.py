# External
import torch

# Internal
from alphagenome_pt import (
    synthetic_dna_sequence,
    synthetic_dna_sequence_one_hot,
)


def test_synthetic_dna_sequence_returns_string_batches():
    sequences = synthetic_dna_sequence(batch_size=2, seq_len=10)

    assert len(sequences) == 2
    assert all(len(sequence) == 10 for sequence in sequences)
    assert all(set(sequence) <= set("ACGTN") for sequence in sequences)


def test_synthetic_dna_one_hot_encodes_unknown_bases_as_zero(monkeypatch):
    monkeypatch.setattr(
        "alphagenome_pt.synthetic.synthetic_dna_sequence",
        lambda batch_size, seq_len: ["ACGTN"] * batch_size,
    )

    encoded = synthetic_dna_sequence_one_hot(batch_size=2, seq_len=5)

    expected = torch.cat((torch.eye(4), torch.zeros(1, 4))).expand(2, -1, -1)
    torch.testing.assert_close(encoded, expected)
