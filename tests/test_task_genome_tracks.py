# Track-scaling tests are adapted from genomicsxai/alphagenome-pytorch,
# tests/unit/test_heads.py (Apache-2.0), specifically TestScalingFunctions and
# TestPredictionsScaling. Task-level tests in this file are original work.

# External
from importlib import metadata

import pytest
import torch

# Internal
from alphagenome_pt import (
    DataBatch,
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from alphagenome_pt.heads import predictions_scaling, targets_scaling
from .helpers import assert_finite_metric_tree


@pytest.mark.parametrize(
    ("head", "resolution"),
    [
        (HeadName.RNA_SEQ, 1),
        (HeadName.CAGE, 1),
        (HeadName.ATAC, 1),
        (HeadName.DNASE, 1),
        (HeadName.PROCAP, 1),
        (HeadName.CHIP_TF, 128),
        (HeadName.CHIP_HISTONE, 128),
    ],
)
def test_individual_genome_track_head(head: HeadName, resolution: int):
    head_name = head.value
    metadata = synthetic_metadata((head,))
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
    setattr(batch, f"{head_name}_mask", None)
    
    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert head_name in result.predictions
    assert (
        result.predictions[head_name][f"scaled_predictions_{resolution}bp"].shape
        == getattr(batch, head_name).shape
    )


def test_genome_track_loss_combines_metadata_and_batch_masks():
    head_name = HeadName.CHIP_TF.value
    metadata = synthetic_metadata(
        (HeadName.CHIP_TF,),
        num_organisms=1,
        num_tracks=3,
    )
    head_metadata = metadata.metadata["heads"][head_name]
    head_metadata["means"] = torch.ones((1, 3))
    head_metadata["track_mask"] = torch.tensor([[True, False, True]])
    model = small_alphagenome(metadata, max_seq_len=2_048)

    targets = torch.ones(1, model.max_seq_len // 128, 3)
    batch = DataBatch(
        organism_index=torch.tensor([0]),
        chip_tf=targets,
        chip_tf_mask=torch.tensor([[[False, True, True]]]),
    )

    def loss_for(updated_predictions):
        return model.metric_tree_from_predictions(
            {
                head_name: {
                    "scaled_predictions_128bp": updated_predictions,
                },
            },
            batch,
        ).total_loss(head_name)

    predictions = torch.full_like(targets, 2.0)
    baseline_loss = loss_for(predictions)

    # Different predictions for the masked tracks should not change the loss.
    masked_predictions = predictions.clone()
    masked_predictions[..., :2] = 100.0
    torch.testing.assert_close(loss_for(masked_predictions), baseline_loss)

    # Different predictions for the unmasked tracks should change the loss.
    included_predictions = predictions.clone()
    included_predictions[..., 2] = 100.0
    assert not torch.isclose(
        loss_for(included_predictions),
        baseline_loss,
    ).item()


@pytest.mark.parametrize(
    ("min_zero", "positional_key"),
    (
        (True, "zero_loss_positional"),
        (False, "loss_positional"),
    ),
)
def test_genome_track_loss_tree_combines_loss_components(
    min_zero: bool,
    positional_key: str,
    monkeypatch,
):
    head_name = HeadName.CHIP_TF.value
    metadata = synthetic_metadata((HeadName.CHIP_TF,))
    model = small_alphagenome(
        metadata,
        min_zero_multinomial_loss=min_zero,
    )
    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
    head = model._heads[head_name]

    # NOTE: Just set the component losses so that we can avoid
    # mask-handling to call _compute_loss().
    components = {
        "loss_total": torch.tensor(2.0),
        "loss_positional": torch.tensor(4.0),
        "zero_loss_positional": torch.tensor(3.0),
    }
    monkeypatch.setattr(head, "_compute_loss", lambda **_: components)

    result = model.loss(batch)

    expected = (
        components["loss_total"]
        # NOTE: Change the 5.0 multiplier if we want
        # that to be configurable in the future.
        + 5.0 * components[positional_key]
    )
    expected = model.dtype_policy.cast_output(expected)
    torch.testing.assert_close(
        result.tree.total_loss(head_name),
        expected,
    )


def test_min_zero_multinomial_loss_config_reaches_genome_track_head():
    metadata = synthetic_metadata(
        (HeadName.RNA_SEQ,),
        num_tracks=2,
    )
    model = small_alphagenome(metadata, min_zero_multinomial_loss=False)

    assert model._heads["rna_seq"]._min_zero_multinomial_loss is False


def _scaling_input(batch=2, seq=1024, tracks=8):
    return torch.randn(batch, seq, tracks).abs() * 50


class TestTrackScalingIsInvertible:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    @pytest.mark.parametrize("apply_squashing", [True, False])
    def test_targets_then_predictions_is_identity(self, apply_squashing):
        x = _scaling_input()
        means = torch.ones(2, 8)
        scaled = targets_scaling(x, means, 128, apply_squashing)
        back = predictions_scaling(scaled, means, 128, apply_squashing)
        torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("apply_squashing", [True, False])
    def test_predictions_then_targets_is_identity(self, apply_squashing):
        x = _scaling_input()
        means = torch.ones(2, 8)
        unscaled = predictions_scaling(x, means, 128, apply_squashing)
        back = targets_scaling(unscaled, means, 128, apply_squashing)
        torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)

    def test_identity_with_nonuniform_track_means(self):
        x = _scaling_input()
        means = torch.rand(2, 8) * 9 + 1
        scaled = targets_scaling(x, means, 128, True)
        back = predictions_scaling(scaled, means, 128, True)
        torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("resolution", [1, 128])
    def test_identity_across_resolutions(self, resolution):
        x = _scaling_input()
        means = torch.ones(2, 8)
        scaled = targets_scaling(x, means, resolution, False)
        back = predictions_scaling(scaled, means, resolution, False)
        torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)


class TestTrackScalingBehaviour:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_below_soft_clip_is_linear_in_resolution(self):
        x = torch.full((1, 4, 1), 0.5)
        means = torch.ones(1, 1)
        out = targets_scaling(x, means, 2, False)
        torch.testing.assert_close(out, x / 2, rtol=1e-6, atol=1e-6)

    def test_soft_clip_compresses_large_values(self):
        means = torch.ones(1, 1)
        small = targets_scaling(torch.full((1, 1, 1), 10.0), means, 1, False)
        large = targets_scaling(
            torch.full((1, 1, 1), 1000.0), means, 1, False
        )
        assert large.item() < 1000.0
        assert large.item() > small.item()
        assert (large / small).item() < 100.0

    def test_squashing_changes_the_result(self):
        x = _scaling_input()
        means = torch.ones(2, 8)
        assert not torch.allclose(
            targets_scaling(x, means, 128, False),
            targets_scaling(x, means, 128, True),
        )

    def test_gradient_flows_through_scaling(self):
        x = _scaling_input().requires_grad_(True)
        targets_scaling(x, torch.ones(2, 8), 128, True).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
