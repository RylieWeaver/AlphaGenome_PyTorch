# Adapted from genomicsxai/alphagenome-pytorch, tests/unit/test_attention.py
# (Apache-2.0), the TestApplyRope and TestApplyRopeGradients classes.
# No golden numbers. Every assertion is a property.
#
# Deviations from the source file:
#   - dropped the in-place and CUDA memory tests; this implementation's apply_rope has no
#     in-place path, so there is nothing to compare against
#   - this implementation's signature is apply_rope(x, positions, max_position) with
#     max_position required, so it is passed explicitly everywhere

import pytest
import torch

from alphagenome_pt.attention import apply_rope

pytestmark = pytest.mark.unit

B, S, H, C = 1, 16, 2, 64


def _x(seed=42):
    torch.manual_seed(seed)
    return torch.randn(B, S, H, C)


class TestApplyRope:
    def test_output_shape_is_unchanged(self):
        x = _x()
        out = apply_rope(x, positions=None, max_position=S)
        assert out.shape == x.shape

    def test_does_not_mutate_input(self):
        x = _x()
        before = x.clone()
        apply_rope(x, positions=None, max_position=S)
        torch.testing.assert_close(x, before)

    def test_values_actually_change(self):
        # A no-op rope would pass every shape check.
        x = _x()
        out = apply_rope(x, positions=None, max_position=S)
        assert not torch.allclose(out, x)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        x = _x().to(dtype)
        assert apply_rope(x, positions=None, max_position=S).dtype == dtype

    def test_explicit_positions_match_default(self):
        x = _x()
        default = apply_rope(x, positions=None, max_position=S)
        explicit = apply_rope(x, positions=torch.arange(S).unsqueeze(0),
                              max_position=S)
        torch.testing.assert_close(default, explicit)

    def test_different_max_position_gives_different_result(self):
        x = _x()
        a = apply_rope(x, positions=None, max_position=S)
        b = apply_rope(x, positions=None, max_position=S * 8)
        assert not torch.allclose(a, b)


class TestRopeIsTranslationInvariant:
    # The whole point of rotary embeddings: attention should depend on how far
    # apart two positions are, not where they sit in the buffer. A rope with the
    # sin and cos swapped passes every shape check and fails this instantly.

    def test_relative_geometry_survives_a_shift(self):
        x = _x()
        out_a = apply_rope(x.clone(), positions=torch.arange(S).unsqueeze(0),
                           max_position=S * 8)
        out_b = apply_rope(x.clone(),
                           positions=torch.arange(10, S + 10).unsqueeze(0),
                           max_position=S * 8)

        # Absolute outputs must differ: rope does encode position.
        assert not torch.allclose(out_a, out_b, atol=1e-5)

        # But adjacent pairs have the same offset in both, so their inner
        # products must match.
        dots_a = (out_a[:, :-1] * out_a[:, 1:]).sum(dim=-1)
        dots_b = (out_b[:, :-1] * out_b[:, 1:]).sum(dim=-1)
        torch.testing.assert_close(dots_a, dots_b, rtol=1e-4, atol=1e-4)

    def test_norm_is_preserved(self):
        # Rope is a rotation, so it must not change vector length.
        x = _x()
        out = apply_rope(x, positions=None, max_position=S)
        torch.testing.assert_close(x.norm(dim=-1), out.norm(dim=-1),
                                   rtol=1e-5, atol=1e-5)


class TestRopeGradients:
    def test_gradient_flows(self):
        x = _x().requires_grad_(True)
        apply_rope(x, positions=None, max_position=S).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_gradient_is_nonzero(self):
        x = _x().requires_grad_(True)
        apply_rope(x, positions=None, max_position=S).pow(2).sum().backward()
        assert x.grad.norm() > 1e-12
