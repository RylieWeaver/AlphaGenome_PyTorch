# External
import pytest
import torch

# Internal
from alphagenome_pt import (
    LossOutput,
    MetricTree,
    synthetic_mlm,
)

from .helpers import DNA_SEQUENCE



@pytest.mark.parametrize(
    "loss_method",
    ("forward", "loss"),
)
def test_loss_methods_default_to_differentiable_output(model, loss_method):
    batch = model.as_data_batch(DNA_SEQUENCE)
    batch.mlm = synthetic_mlm(batch_size=1, seq_len=len(DNA_SEQUENCE))
    model.zero_grad(set_to_none=True)

    if loss_method == "forward":
        output = model(batch, mode="loss")
    elif loss_method == "loss":
        output = model.loss(batch)
    else:
        raise ValueError(f"Unexpected loss_method: {loss_method}")

    output.total.backward()

    assert isinstance(output, LossOutput)
    assert output.total.ndim == 0
    assert torch.isfinite(output.total)
    assert torch.allclose(output.total, output.tree.total_loss())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_loss_requires_batch_with_targets(model):
    data = DNA_SEQUENCE
    with pytest.raises(TypeError, match="DataBatch"):
        model.loss(data)
    with pytest.raises(TypeError, match="DataBatch"):
        model(data, mode="loss")

    batch = model.as_data_batch(DNA_SEQUENCE)
    with pytest.raises(ValueError, match="target not in batch"):
        model.loss(batch)
    with pytest.raises(ValueError, match="target not in batch"):
        model(batch, mode="loss")


@pytest.mark.parametrize("loss_method", ("forward", "loss"))
@pytest.mark.parametrize("return_predictions", (False, True))
@pytest.mark.parametrize("return_embeddings", (False, True))
def test_loss_return_options_produce_loss_output(
    model,
    loss_method,
    return_predictions,
    return_embeddings,
):
    batch = model.as_data_batch(DNA_SEQUENCE)
    batch.mlm = synthetic_mlm(batch_size=1, seq_len=len(DNA_SEQUENCE))

    kwargs = {
        "return_predictions": return_predictions,
        "return_embeddings": return_embeddings,
    }
    if loss_method == "forward":
        result = model(batch, mode="loss", **kwargs)
    elif loss_method == "loss":
        result = model.loss(batch, **kwargs)
    else:
        raise ValueError(f"Unexpected loss_method: {loss_method}")

    assert isinstance(result, LossOutput)
    assert isinstance(result.tree, MetricTree)
    assert (result.predictions is not None) == return_predictions
    assert (result.embeddings is not None) == return_embeddings
    assert result.total.ndim == 0
    assert torch.allclose(result.total, result.tree.total_loss())


def test_metric_tree_from_predictions_returns_metric_tree(model):
    batch = model.as_data_batch(DNA_SEQUENCE)
    batch.mlm = synthetic_mlm(batch_size=1, seq_len=len(DNA_SEQUENCE))
    predictions = model(batch)

    tree = model.metric_tree_from_predictions(predictions, batch)

    assert isinstance(tree, MetricTree)
    assert torch.isfinite(tree.total_loss())


def test_metric_tree_from_predictions_requires_predictions_and_targets_for_enabled_heads(
    model,
):
    batch = model.as_data_batch(DNA_SEQUENCE)
    batch.mlm = synthetic_mlm(batch_size=1, seq_len=len(DNA_SEQUENCE))
    predictions = model(batch)

    with pytest.raises(ValueError, match="Predictions for enabled head"):
        model.metric_tree_from_predictions({}, batch)

    batch.mlm = None
    with pytest.raises(ValueError, match="target not in batch"):
        model.metric_tree_from_predictions(predictions, batch)
