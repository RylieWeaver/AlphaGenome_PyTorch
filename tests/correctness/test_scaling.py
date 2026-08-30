# Adapted from genomicsxai/alphagenome-pytorch, tests/unit/test_heads.py (Apache-2.0),
# specifically the TestScalingFunctions and TestPredictionsScaling classes.
# No golden numbers. Every assertion is a property, so nothing was carried over
# from the source except the choice of what to assert.
#
# Deviations from the source file:
#   - split into its own file; genomicsxai/alphagenome-pytorch mixed these in with head shape tests
#   - alphagenome_pt exposes predictions_scaling/targets_scaling at module level in heads.py

import pytest
import torch

from alphagenome_pt.heads import predictions_scaling, targets_scaling

pytestmark = pytest.mark.unit


def _x(batch=2, seq=1024, tracks=8):
    # Positive values with a wide range, to reach the soft-clip region.
    return torch.randn(batch, seq, tracks).abs() * 50


class TestScalingIsInvertible:
    # This is the highest-leverage test in the tier. If these two functions are
    # not exact inverses, every training target is silently wrong: the loss stays
    # finite, training still converges, and it converges to the wrong thing.

    @pytest.mark.parametrize("apply_squashing", [False, True])
    def test_targets_then_predictions_is_identity(self, apply_squashing):
        x = _x()
        means = torch.ones(2, 8)
        scaled = targets_scaling(x, means, 128, apply_squashing)
        back = predictions_scaling(scaled, means, 128, apply_squashing)
        torch.testing.assert_close(x, back, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("apply_squashing", [False, True])
    def test_predictions_then_targets_is_identity(self, apply_squashing):
        x = _x()
        means = torch.ones(2, 8)
        unscaled = predictions_scaling(x, means, 128, apply_squashing)
        back = targets_scaling(unscaled, means, 128, apply_squashing)
        torch.testing.assert_close(x, back, rtol=1e-4, atol=1e-4)

    def test_identity_with_nonuniform_track_means(self):
        # A per-track mean is a per-track divisor. Getting the broadcast axis
        # wrong here would still produce finite numbers.
        x = _x()
        means = torch.rand(2, 8) * 9 + 1
        scaled = targets_scaling(x, means, 128, True)
        back = predictions_scaling(scaled, means, 128, True)
        torch.testing.assert_close(x, back, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("resolution", [1, 128])
    def test_identity_across_resolutions(self, resolution):
        x = _x()
        means = torch.ones(2, 8)
        scaled = targets_scaling(x, means, resolution, False)
        back = predictions_scaling(scaled, means, resolution, False)
        torch.testing.assert_close(x, back, rtol=1e-4, atol=1e-4)


class TestScalingBehaviour:
    def test_below_soft_clip_is_linear_in_resolution(self):
        # Small values bypass the soft clip, so scaling is just a division.
        x = torch.full((1, 4, 1), 0.5)
        means = torch.ones(1, 1)
        out = targets_scaling(x, means, 2, False)
        torch.testing.assert_close(out, x / 2, rtol=1e-6, atol=1e-6)

    def test_soft_clip_compresses_large_values(self):
        # Above the clip threshold the transform must be sublinear.
        means = torch.ones(1, 1)
        small = targets_scaling(torch.full((1, 1, 1), 10.0), means, 1, False)
        large = targets_scaling(torch.full((1, 1, 1), 1000.0), means, 1, False)
        assert large.item() < 1000.0
        assert large.item() > small.item()
        assert (large / small).item() < 100.0

    def test_squashing_changes_the_result(self):
        # If apply_squashing were a no-op, RNA-seq would train on the wrong scale.
        x = _x()
        means = torch.ones(2, 8)
        assert not torch.allclose(
            targets_scaling(x, means, 128, False),
            targets_scaling(x, means, 128, True),
        )

    def test_gradient_flows_through_scaling(self):
        x = _x().requires_grad_(True)
        targets_scaling(x, torch.ones(2, 8), 128, True).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
