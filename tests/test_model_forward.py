# External
from unittest import mock
import pytest
import torch

# Internal
from alphagenome_pt import DataBatch, HeadName
from alphagenome_pt.embeddings import Embeddings

from .helpers import DNA_SEQUENCE, assert_predictions_close



def test_supported_dna_input_forms_are_numerically_equivalent(model):
    encoded_3d = model.as_data_batch(DNA_SEQUENCE).dna_sequence_one_hot
    assert encoded_3d is not None
    encoded_2d = encoded_3d[0]
    inputs = [
        DNA_SEQUENCE,
        [DNA_SEQUENCE],
        encoded_2d,
        encoded_3d,
        DataBatch(dna_sequence=DNA_SEQUENCE),
        DataBatch(dna_sequence=[DNA_SEQUENCE]),
        DataBatch(dna_sequence_one_hot=encoded_2d.clone()),
        DataBatch(dna_sequence_one_hot=encoded_3d.clone()),
    ]

    with torch.inference_mode():
        expected = model(DNA_SEQUENCE)
        for data in inputs:
            assert_predictions_close(model(data), expected)


def test_supported_organism_input_forms_are_numerically_equivalent(model):
    dna_sequences = [DNA_SEQUENCE, DNA_SEQUENCE]
    organism_indices = [
        1,
        torch.tensor(1),
        [1, 1],
        (1, 1),
        torch.tensor([1, 1]),
        torch.tensor([[1], [1]]),
    ]

    with torch.inference_mode():
        expected = model(dna_sequences, organism_index=[1, 1])
        for organism_index in organism_indices:
            assert_predictions_close(
                model(dna_sequences, organism_index=organism_index),
                expected,
            )
            batch = DataBatch(
                dna_sequence=dna_sequences,
                organism_index=organism_index,
            )
            assert_predictions_close(model(batch), expected)


def test_forward_uses_per_example_organism_indices_from_batch(model):
    dna_one_hot = model.as_data_batch(
        [DNA_SEQUENCE, DNA_SEQUENCE]
    ).dna_sequence_one_hot
    assert dna_one_hot is not None
    mixed_batch = DataBatch(
        dna_sequence_one_hot=dna_one_hot,
        organism_index=torch.tensor([0, 1]),
    )
    all_zero_batch = DataBatch(
        dna_sequence_one_hot=dna_one_hot,
        organism_index=torch.tensor([0, 0]),
    )

    with torch.inference_mode():
        _, mixed = model(mixed_batch, return_embeddings=True)
        _, all_zero = model(all_zero_batch, return_embeddings=True)

    torch.testing.assert_close(mixed.embeddings_1bp[0], all_zero.embeddings_1bp[0])
    torch.testing.assert_close(mixed.embeddings_128bp[0], all_zero.embeddings_128bp[0])
    torch.testing.assert_close(mixed.embeddings_pair[0], all_zero.embeddings_pair[0])
    assert not torch.allclose(mixed.embeddings_1bp[1], all_zero.embeddings_1bp[1])
    assert not torch.allclose(mixed.embeddings_128bp[1], all_zero.embeddings_128bp[1])
    assert not torch.allclose(mixed.embeddings_pair[1], all_zero.embeddings_pair[1])


def test_forward_uses_enabled_heads_not_available_targets(model):
    batch = DataBatch(
        dna_sequence=[DNA_SEQUENCE],
        atac=torch.zeros(1, len(DNA_SEQUENCE), 2),
    )

    with torch.inference_mode():
        predictions = model(batch)

    assert set(predictions) == {
        HeadName.MASKED_LANGUAGE_MODELING.value,
    }


def test_embed_does_not_run_prediction_heads(model):
    head = model._heads[HeadName.MASKED_LANGUAGE_MODELING.value]

    with mock.patch.object(
        head,
        "forward",
        side_effect=AssertionError("embed() ran a prediction head"),
    ):
        with torch.inference_mode():
            embeddings = model.embed(DNA_SEQUENCE)

    assert isinstance(embeddings, Embeddings)
    assert embeddings.embeddings_1bp is not None
    assert embeddings.embeddings_128bp is not None
    assert embeddings.embeddings_pair is not None


def test_embed_matches_forward_modes(model):
    with torch.inference_mode():
        embedded = model.embed(DNA_SEQUENCE)
        embedded_by_forward = model(DNA_SEQUENCE, mode="embed")
        _, prediction_embeddings = model(
            DNA_SEQUENCE,
            return_embeddings=True,
        )

    assert isinstance(embedded_by_forward, Embeddings)
    for actual in (embedded_by_forward, prediction_embeddings):
        torch.testing.assert_close(
            actual.embeddings_1bp,
            embedded.embeddings_1bp,
        )
        torch.testing.assert_close(
            actual.embeddings_128bp,
            embedded.embeddings_128bp,
        )
        torch.testing.assert_close(
            actual.embeddings_pair,
            embedded.embeddings_pair,
        )


def test_forward_and_predict_return_contracts(model):
    with torch.inference_mode():
        called = model(DNA_SEQUENCE)
        forwarded = model(DNA_SEQUENCE, mode="predict")
        predicted = model.predict(DNA_SEQUENCE)
        called_with_embeddings = model(DNA_SEQUENCE, return_embeddings=True)
        predicted_with_embeddings = model.predict(
            DNA_SEQUENCE,
            return_embeddings=True,
        )

    assert isinstance(called, dict)
    assert isinstance(forwarded, dict)
    assert isinstance(predicted, dict)
    assert_predictions_close(forwarded, called)
    assert_predictions_close(predicted, called)

    for result in (called_with_embeddings, predicted_with_embeddings):
        assert isinstance(result, tuple)
        assert len(result) == 2
        predictions, embeddings = result
        assert_predictions_close(predictions, called)
        assert isinstance(embeddings, Embeddings)


def test_forward_rejects_invalid_mode(model):
    with pytest.raises(ValueError, match="embed.*predict.*loss"):
        model(DNA_SEQUENCE, mode="invalid")


@pytest.mark.parametrize("value", (False, True))
@pytest.mark.parametrize(
    ("mode", "option"),
    (
        ("embed", "return_predictions"),
        ("embed", "return_embeddings"),
        ("predict", "return_predictions"),
    ),
)
def test_forward_rejects_options_unused_by_mode(model, mode, option, value):
    with pytest.raises(ValueError, match=f"{option}.*mode='{mode}'"):
        model(DNA_SEQUENCE, mode=mode, **{option: value})


def test_predict_preserves_training_states_and_disables_grad(model):
    head = model._heads[HeadName.MASKED_LANGUAGE_MODELING.value]
    original_states = [(module, module.training) for module in model.modules()]

    try:
        # NOTE: Setting some parts to training and some to eval just
        # to make sure that there are different states being tested.
        model.train()
        head.eval()

        expected_states = [
            (module, module.training)
            for module in model.modules()
        ]

        predictions = model.predict(DNA_SEQUENCE)

        assert all(
            module.training == expected
            for module, expected in expected_states
        )
        assert all(
            not output.requires_grad
            for head_outputs in predictions.values()
            for output in head_outputs.values()
        )
    finally:
        # Changing back
        for module, training in original_states:
            module.training = training
