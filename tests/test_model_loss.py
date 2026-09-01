# Full-model backward coverage in this file is adapted from
# genomicsxai/alphagenome-pytorch, tests/integration/test_backward.py
# (Apache-2.0). The existing loss-contract tests are original work.

# External
import pytest
import torch

# Internal
from alphagenome_pt import (
    LossOutput,
    MetricTree,
    synthetic_mlm,
)

from .helpers import ALL_HEADS, DNA_SEQUENCE, build_small_model_with_batch



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


class TestFullModelBackward:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_every_parameter_receives_a_gradient(self):
        model, batch = build_small_model_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch).total.backward()

        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is None or parameter.grad.norm() == 0
        ]
        assert not missing, (
            f"{len(missing)} parameters received no gradient:\n"
            + "\n".join(missing[:20])
        )

    def test_no_nan_or_inf_in_any_gradient(self):
        model, batch = build_small_model_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch).total.backward()
        invalid = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        assert not invalid, invalid

    def test_optimizer_step_changes_the_weights(self):
        model, batch = build_small_model_with_batch(ALL_HEADS)
        model.train()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.loss(batch).total.backward()
        optimizer.step()
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters())
        )

    def test_all_heads_can_be_trained_together(self):
        model, batch = build_small_model_with_batch(ALL_HEADS)
        model.train()
        output = model.loss(batch, return_predictions=True)
        assert torch.isfinite(output.total)
        assert output.predictions is not None
        assert len(output.predictions) == len(ALL_HEADS)
        output.total.backward()
