# Adapted from genomicsxai/alphagenome-pytorch, tests/unit/test_losses.py (Apache-2.0).
# The golden values originate from the reference JAX implementation
# (google-deepmind/alphagenome); that suite recorded them, and each one was
# re-verified against alphagenome_pt.losses before being pinned here. See info.md.
#
# Deviations from the source file:
#   - dropped TestCountWeight; multinomial_loss here has no count_weight argument
#   - dropped max_sum_preds assertions; not a key in this implementation's return dict
#   - added TestMultinomialGoldenBothBranches to cover the min_zero switch, which
#     has no counterpart in the reference JAX implementation (google-deepmind/alphagenome)

import numpy as np
import pytest
import torch

from alphagenome_pt import losses

pytestmark = pytest.mark.unit


class TestSafeMaskedMean:
    def test_no_mask(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.isclose(losses._safe_masked_mean(x), torch.tensor(2.5))

    def test_with_mask(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        assert torch.isclose(losses._safe_masked_mean(x, mask), torch.tensor(1.5))

    def test_all_masked_is_zero_not_nan(self):
        # A fully masked input must give 0.0, not a 0/0 NaN.
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        out = losses._safe_masked_mean(x, mask)
        assert torch.isfinite(out)
        assert torch.isclose(out, torch.tensor(0.0))


class TestPoissonLoss:
    def test_perfect_prediction_is_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        m = torch.ones(3, dtype=torch.bool)
        assert losses.poisson_loss(y_true=y, y_pred=y, mask=m).item() < 1e-5

    def test_wrong_prediction_is_positive(self):
        yt = torch.tensor([1.0, 2.0, 3.0])
        yp = torch.tensor([3.0, 1.0, 2.0])
        m = torch.ones(3, dtype=torch.bool)
        assert losses.poisson_loss(y_true=yt, y_pred=yp, mask=m).item() > 0


class TestMSE:
    def test_perfect_prediction_is_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        m = torch.ones(3, dtype=torch.bool)
        assert losses.mse(y_true=y, y_pred=y, mask=m).item() < 1e-7

    def test_simple_error(self):
        # ((2-1)^2 + (4-2)^2) / 2 = 2.5
        yt = torch.tensor([1.0, 2.0])
        yp = torch.tensor([2.0, 4.0])
        m = torch.ones(2, dtype=torch.bool)
        assert torch.isclose(losses.mse(y_true=yt, y_pred=yp, mask=m),
                             torch.tensor(2.5), atol=1e-6)


class TestMultinomialLoss:
    def test_masking_equals_truncation(self):
        # Masking a position must be indistinguishable from deleting it.
        yt = torch.tensor([[[10.0, 1.0, 3.0], [5.0, 2.0, 20.0]]])
        yp = torch.tensor([[[0.5, 2.5, 1.0], [2.5, 0.5, 1.0]]])

        full = losses.multinomial_loss(
            y_true=yt, y_pred=yp,
            mask=torch.tensor([[[True, True, True]]]),
            multinomial_resolution=1, positional_weight=1.0)["loss"]

        masked = losses.multinomial_loss(
            y_true=yt, y_pred=yp,
            mask=torch.tensor([[[True, True, False]]]),
            multinomial_resolution=1, positional_weight=1.0)["loss"]

        truncated = losses.multinomial_loss(
            y_true=torch.tensor([[[10.0, 1.0], [5.0, 2.0]]]),
            y_pred=torch.tensor([[[0.5, 2.5], [2.5, 0.5]]]),
            mask=torch.tensor([[[True, True]]]),
            multinomial_resolution=1, positional_weight=1.0)["loss"]

        np.testing.assert_almost_equal(masked.item(), truncated.item(), decimal=5)
        assert masked.item() < full.item()

    def test_returns_expected_keys(self):
        # alphagenome_pt returns zero_loss_positional; genomicsxai/alphagenome-pytorch returns max_sum_preds instead.
        out = losses.multinomial_loss(
            y_true=torch.ones((1, 4, 1)), y_pred=torch.ones((1, 4, 1)),
            mask=torch.ones((1, 1, 1), dtype=torch.bool),
            multinomial_resolution=1, positional_weight=1.0)
        assert set(out) == {"loss", "loss_total", "loss_positional",
                            "zero_loss_positional"}
        assert all(torch.isfinite(v).all() for v in out.values())

    @pytest.mark.parametrize("resolution", [1, 2, 4])
    def test_finite_at_every_resolution(self, resolution):
        out = losses.multinomial_loss(
            y_true=torch.ones((1, 4, 1)), y_pred=torch.ones((1, 4, 1)),
            mask=torch.ones((1, 1, 1), dtype=torch.bool),
            multinomial_resolution=resolution, positional_weight=1.0)
        assert torch.isfinite(out["loss"]).all()

    def test_positional_weight_scales_only_positional_term(self):
        kw = dict(y_true=torch.tensor([[[10.0, 5.0], [5.0, 10.0]]]),
                  y_pred=torch.tensor([[[8.0, 6.0], [4.0, 12.0]]]),
                  mask=torch.ones((1, 1, 2), dtype=torch.bool),
                  multinomial_resolution=2)
        w1 = losses.multinomial_loss(positional_weight=1.0, **kw)
        w2 = losses.multinomial_loss(positional_weight=2.0, **kw)

        # The count term must not move when the positional weight changes.
        torch.testing.assert_close(w1["loss_total"], w2["loss_total"])
        # The difference is exactly one extra positional term.
        delta = (w2["loss"] - w1["loss"]).item()
        assert np.isclose(delta, w1["zero_loss_positional"].item(), atol=1e-5)


class TestCrossEntropy:
    def test_from_logits_perfect_prediction_is_low(self):
        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=torch.tensor([[[10.0, -10.0, -10.0]]]),
            y_true=torch.tensor([[[1.0, 0.0, 0.0]]]),
            mask=torch.ones(1, 1, 1, dtype=torch.bool), axis=-1)
        assert loss.item() < 1e-3

    def test_bce_stable_at_extreme_logits(self):
        loss = losses.binary_crossentropy_from_logits(
            y_pred=torch.tensor([[[100.0, -100.0]]]),
            y_true=torch.tensor([[[1.0, 0.0]]]),
            mask=torch.ones(1, 1, 2, dtype=torch.bool))
        assert torch.isfinite(loss)

    def test_cross_entropy_non_negative(self):
        loss = losses.cross_entropy_loss(
            y_true=torch.tensor([[[1.0, 2.0, 3.0]]]),
            y_pred=torch.tensor([[[3.0, 2.0, 1.0]]]),
            mask=torch.ones(1, 1, 3, dtype=torch.bool), axis=-1)
        assert loss.item() >= 0


class TestExtremeValues:
    def test_poisson_tiny_predictions(self):
        loss = losses.poisson_loss(
            y_true=torch.tensor([1.0, 2.0]), y_pred=torch.tensor([1e-8, 1e-8]),
            mask=torch.ones(2, dtype=torch.bool))
        assert torch.isfinite(loss)

    def test_poisson_large_predictions(self):
        loss = losses.poisson_loss(
            y_true=torch.tensor([1.0, 2.0]), y_pred=torch.tensor([1e8, 1e8]),
            mask=torch.ones(2, dtype=torch.bool))
        assert torch.isfinite(loss)


class TestGradientThroughLoss:
    def test_poisson_gradient(self):
        yp = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        losses.poisson_loss(y_true=torch.tensor([1.5, 2.5, 2.0]), y_pred=yp,
                            mask=torch.ones(3, dtype=torch.bool)).backward()
        assert yp.grad is not None and torch.isfinite(yp.grad).all()

    def test_mse_gradient(self):
        yp = torch.tensor([1.0, 2.0], requires_grad=True)
        losses.mse(y_true=torch.tensor([2.0, 4.0]), y_pred=yp,
                   mask=torch.ones(2, dtype=torch.bool)).backward()
        assert yp.grad is not None and torch.isfinite(yp.grad).all()

    def test_multinomial_gradient(self):
        # Sequence axis must be divisible by multinomial_resolution, so S=2 here.
        yp = torch.tensor([[[1.0, 2.0, 0.5, 3.0],
                            [0.5, 1.5, 0.8, 2.0]]], requires_grad=True)
        losses.multinomial_loss(
            y_true=torch.tensor([[[10.0, 30.0, 5.0, 20.0],
                                  [8.0, 15.0, 3.0, 12.0]]]), y_pred=yp,
            mask=torch.ones(1, 1, 4, dtype=torch.bool),
            multinomial_resolution=2, positional_weight=1.0)["loss"].backward()
        assert yp.grad is not None and torch.isfinite(yp.grad).all()

    def test_cross_entropy_from_logits_gradient(self):
        logits = torch.tensor([[[2.0, 0.5, -1.0]]], requires_grad=True)
        losses.cross_entropy_loss_from_logits(
            y_pred_logits=logits, y_true=torch.tensor([[[0.9, 0.05, 0.05]]]),
            mask=torch.ones(1, 1, 1, dtype=torch.bool), axis=-1).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


class TestGoldenValues:
    # These pin numerical behaviour against the reference JAX implementation.
    # A failure here means the maths changed. That may be intentional; it is
    # never accidental.
    Y_PRED_3 = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
    Y_TRUE_3 = torch.tensor([[[0.8, 2.2, 2.9], [0.6, 1.4, 2.6]]])
    MASK_3 = torch.ones(1, 1, 3, dtype=torch.bool)

    def test_poisson_golden(self):
        loss = losses.poisson_loss(y_true=self.Y_TRUE_3, y_pred=self.Y_PRED_3,
                                   mask=self.MASK_3)
        assert torch.isclose(loss, torch.tensor(0.0079382462), atol=1e-7)

    def test_mse_golden(self):
        loss = losses.mse(y_true=self.Y_TRUE_3, y_pred=self.Y_PRED_3,
                          mask=self.MASK_3)
        assert torch.isclose(loss, torch.tensor(0.0199999977), atol=1e-7)

    def test_cross_entropy_from_logits_golden(self):
        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=torch.tensor([[[2.0, 0.5, -1.0, 0.1]]]),
            y_true=torch.tensor([[[0.9, 0.05, 0.03, 0.02]]]),
            mask=torch.ones(1, 1, 1, dtype=torch.bool), axis=-1)
        assert torch.isclose(loss, torch.tensor(0.5554059148), atol=1e-7)

    def test_binary_crossentropy_golden(self):
        loss = losses.binary_crossentropy_from_logits(
            y_pred=torch.tensor([[[2.0, -1.0, 0.5]]]),
            y_true=torch.tensor([[[1.0, 0.0, 1.0]]]),
            mask=torch.ones(1, 1, 3, dtype=torch.bool))
        assert torch.isclose(loss, torch.tensor(0.3047555685), atol=1e-7)

    def test_cross_entropy_golden(self):
        loss = losses.cross_entropy_loss(
            y_true=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]]),
            y_pred=torch.tensor([[[1.5, 2.5, 0.5, 3.0], [0.8, 1.0, 2.0, 4.0]]]),
            mask=torch.ones((1, 2, 4), dtype=torch.bool), axis=-1)
        # 1e-6 here, not 1e-7: accumulation order differs slightly from JAX.
        assert torch.isclose(loss, torch.tensor(1.4022779465), atol=1e-6)


class TestMultinomialGoldenBothBranches:
    # multinomial_loss has a min_zero switch that the reference JAX implementation (google-deepmind/alphagenome) lacks.
    # min_zero=False reproduces the JAX value exactly. min_zero=True is this implementation's
    # default and is pinned separately so neither branch can drift.
    KW = dict(
        y_true=torch.tensor([[[10.0, 30.0, 5.0, 20.0], [8.0, 15.0, 3.0, 12.0]]]),
        y_pred=torch.tensor([[[1.0, 2.0, 0.5, 3.0], [0.5, 1.5, 0.8, 2.0]]]),
        mask=torch.ones(1, 1, 4, dtype=torch.bool),
        multinomial_resolution=2, positional_weight=1.0)

    def test_components_match_jax(self):
        out = losses.multinomial_loss(**self.KW)
        assert torch.isclose(out["loss_total"], torch.tensor(17.7364959717), atol=1e-4)
        assert torch.isclose(out["loss_positional"], torch.tensor(8.7234439850), atol=1e-4)

    def test_min_zero_false_matches_jax(self):
        out = losses.multinomial_loss(min_zero=False, **self.KW)
        assert torch.isclose(out["loss"], torch.tensor(26.4599399567), atol=1e-4)

    def test_min_zero_true_is_the_default(self):
        out = losses.multinomial_loss(min_zero=True, **self.KW)
        assert torch.isclose(out["loss"], torch.tensor(18.0260715485), atol=1e-4)
        torch.testing.assert_close(out["loss"],
                                   losses.multinomial_loss(**self.KW)["loss"])
