# External
from copy import deepcopy

import pytest
import torch

# Internal
from alphagenome_pt import DataBatch, synthetic_batch

from .helpers import DNA_SEQUENCE



##### DNA SEQUENCE ######
def test_as_data_batch_encodes_dna_sequence(model):
    sequence = "aCgTné"

    batch = model.as_data_batch(sequence)

    assert batch.dna_sequence == [sequence]
    assert batch.dna_sequence_one_hot is not None
    expected = torch.cat((torch.eye(4), torch.zeros(2, 4)))
    torch.testing.assert_close(batch.dna_sequence_one_hot[0], expected)


@pytest.mark.parametrize(
    "sequences",
    (
        ["acgt", "TGCA"],
        ("acgt", "TGCA"),
    ),
)
def test_as_data_batch_encodes_dna_sequence_batch(model, sequences):
    batch = model.as_data_batch(
        sequences,
        organism_index=[0, 1],
    )

    assert batch.dna_sequence == ["acgt", "TGCA"]
    assert batch.dna_sequence_one_hot is not None
    assert batch.organism_index is not None
    assert batch.organism_index.tolist() == [0, 1]
    acgt = torch.eye(4)
    expected = torch.stack((acgt, torch.flip(acgt, dims=(0,))))
    torch.testing.assert_close(batch.dna_sequence_one_hot, expected)


def test_as_data_batch_accepts_one_hot_dna_sequence(model):
    one_hot = torch.eye(4)

    batch = model.as_data_batch(one_hot)

    assert batch.dna_sequence is None
    assert batch.dna_sequence_one_hot is not None
    torch.testing.assert_close(
        batch.dna_sequence_one_hot,
        one_hot.unsqueeze(0),
    )


def test_as_data_batch_accepts_one_hot_dna_batch(model):
    one_hot = torch.stack((torch.eye(4), torch.flip(torch.eye(4), dims=(0,))))

    batch = model.as_data_batch(one_hot, organism_index=[0, 1])

    assert batch.dna_sequence is None
    assert batch.dna_sequence_one_hot is not None
    assert batch.organism_index is not None
    assert batch.organism_index.tolist() == [0, 1]
    torch.testing.assert_close(batch.dna_sequence_one_hot, one_hot)


@pytest.mark.parametrize(
    "data",
    [
        [],
        ["ACGT", "ACG"],
        torch.zeros(4),
        torch.zeros(0, 4, 4),
        torch.zeros(1, 4, 3),
        object(),
    ],
)
def test_as_data_batch_rejects_invalid_dna_inputs(model, data):
    with pytest.raises((TypeError, ValueError)):
        model.as_data_batch(data)



##### ORGANISM INDEX ######
@pytest.mark.parametrize(
    ("organism_index", "expected"),
    [
        ([0, 1], [0, 1]),
        ((0, 1), [0, 1]),
        (torch.tensor([0, 1]), [0, 1]),
        (torch.tensor([[0], [1]]), [0, 1]),
    ],
)
def test_as_data_batch_accepts_organism_index_input_types(
    model,
    organism_index,
    expected,
):
    normalized = model.as_data_batch(
        [DNA_SEQUENCE, DNA_SEQUENCE],
        organism_index=organism_index,
    )

    assert normalized.organism_index is not None
    assert normalized.organism_index.tolist() == expected


@pytest.mark.parametrize(
    ("organism_index", "error"),
    [
        (torch.tensor([0, 1]), ValueError),     # wrong length for one sequence
        (torch.tensor([[[0]]]), ValueError),    # wrong rank (3D or more)
        (torch.tensor([2]), ValueError),        # outside the valid range [0, 1]
        (torch.tensor([-1]), ValueError),       # negative index
        (1.2, TypeError),                       # floating-point scalar
        (True, TypeError),                      # boolean scalar
        (torch.tensor(1.2), TypeError),         # floating-point tensor scalar
        (torch.tensor(True), TypeError),        # boolean tensor scalar
        (torch.tensor([1.2]), TypeError),       # floating-point index
        (torch.tensor([True]), TypeError),      # boolean index
    ],
)
def test_as_data_batch_rejects_invalid_organism_indices(
    model,
    organism_index,
    error,
):
    with pytest.raises(error):
        model.as_data_batch(DNA_SEQUENCE, organism_index=organism_index)


@pytest.mark.parametrize("organism_index", [1, torch.tensor(1)])
def test_as_data_batch_repeats_scalar_organism_index_to_dna_batch_size(
    model,
    organism_index,
):
    normalized = model.as_data_batch(
        [DNA_SEQUENCE, DNA_SEQUENCE],
        organism_index=organism_index,
    )

    assert normalized.organism_index is not None
    assert normalized.organism_index.tolist() == [1, 1]


def test_as_data_batch_accepts_matching_organism_indices(model):
    batch = DataBatch(
        dna_sequence=[DNA_SEQUENCE, DNA_SEQUENCE],
        organism_index=torch.tensor([0, 1]),
    )

    normalized = model.as_data_batch(
        batch,
        organism_index=torch.tensor([[0], [1]]),
    )

    assert normalized.organism_index is not None
    assert normalized.organism_index.tolist() == [0, 1]


def test_as_data_batch_rejects_conflicting_organism_indices(model):
    batch = DataBatch(
        dna_sequence=[DNA_SEQUENCE],
        organism_index=torch.tensor([1]),
    )

    with pytest.raises(ValueError, match="must be equal"):
        model.as_data_batch(batch, organism_index=[0])


def test_as_data_batch_defaults_organism_index_to_zero(model):
    data = DataBatch(dna_sequence=[DNA_SEQUENCE, DNA_SEQUENCE])

    default = model.as_data_batch(data)
    explicit = model.as_data_batch(data, organism_index=[0, 0])

    assert default.organism_index is not None
    assert explicit.organism_index is not None
    torch.testing.assert_close(
        default.organism_index,
        torch.zeros(2, dtype=torch.long),
    )
    torch.testing.assert_close(default.organism_index, explicit.organism_index)



##### OTHER #####
def test_as_data_batch_validates_raw_and_one_hot_dna_shapes(model):
    batch = DataBatch(
        dna_sequence=["ACGT"],
        dna_sequence_one_hot=torch.zeros(1, 3, 4),
    )

    with pytest.raises(ValueError, match="disagree"):
        model.as_data_batch(batch)


def test_as_data_batch_requires_dna(model):
    data = DataBatch(
        mlm=torch.zeros(2, len(DNA_SEQUENCE), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="DNA input is required"):
        model.as_data_batch(data)


def test_as_data_batch_returns_new_batch_and_preserves_all_fields(model):
    data = synthetic_batch(batch_size=1, seq_len=len(DNA_SEQUENCE))
    data.dna_sequence = [DNA_SEQUENCE]
    data.dna_sequence_one_hot = (
        torch.eye(4).repeat(len(DNA_SEQUENCE) // 4, 1).unsqueeze(0)
    )
    data.organism_index = None
    before = deepcopy(data)
    assert all(
        value is not None
        for name, value in vars(before).items()
        if name != "organism_index"
    )

    normalized = model.as_data_batch(data)

    assert normalized is not data
    for name, expected in vars(before).items():
        source = getattr(data, name)
        result = getattr(normalized, name)
        if name == "organism_index":
            assert source is None
            torch.testing.assert_close(result, torch.zeros(1, dtype=torch.long))
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(source, expected)
            torch.testing.assert_close(result, expected)
        else:
            assert source == result == expected
