# Adapted from genomicsxai/alphagenome-pytorch,
# tests/integration/test_backward.py (Apache-2.0). No golden numbers.
#
# Deviations from the source file:
#   - genomicsxai/alphagenome-pytorch hand-rolls a compute_combined_loss helper;
#     alphagenome_pt has model.loss(batch) built in, returning LossOutput
#   - component names differ: genomicsxai/alphagenome-pytorch's MHABlock is this implementation's MHA, genomicsxai/alphagenome-pytorch's
#     RMSBatchNorm is this implementation's BatchNorm(rms_norm=True)
#   - sync_bn=False everywhere in the component tests, so they run single-process

import pytest
import torch

from alphagenome_pt import small_alphagenome, synthetic_batch, synthetic_metadata
from alphagenome_pt.convolutions import (
    ConvBlock,
    DownResBlock,
    StandardizedConv1d,
    UpResBlock,
)
from alphagenome_pt.layers import BatchNorm
from alphagenome_pt.precision import FLOAT32_DTYPE_POLICY, dtype_policy_context

from .helpers import ALL_HEADS, build_with_batch

pytestmark = pytest.mark.integration


def _healthy(param, name):
    assert param.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(param.grad).all(), f"{name}: non-finite gradient"
    norm = param.grad.norm().item()
    # Treat gradients at or below 1e-12 as vanished in practice.
    assert norm > 1e-12, f"{name}: gradient vanished ({norm})"
    assert norm < 1e8, f"{name}: gradient exploded ({norm})"


class TestFullModelBackward:
    def test_every_parameter_receives_a_gradient(self):
        # The most valuable test in the tier. A parameter with no gradient never
        # trains: it sits at its random init for the whole run and nothing says so.
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch).total.backward()

        dead = [n for n, p in model.named_parameters()
                if p.grad is None or p.grad.norm() == 0]
        assert not dead, (f"{len(dead)} parameters received no gradient:\n"
                          + "\n".join(dead[:20]))

    def test_no_nan_or_inf_in_any_gradient(self):
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch).total.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, bad

    def test_optimizer_step_changes_the_weights(self):
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.loss(batch).total.backward()
        opt.step()
        assert any(not torch.equal(a, b)
                   for a, b in zip(before, model.parameters()))

    def test_all_heads_can_be_trained_together(self):
        # the existing tests in tests/ only ever enable one head at a time, so the combined
        # loss path had no coverage until now.
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        output = model.loss(batch, return_predictions=True)
        assert torch.isfinite(output.total)
        assert output.predictions is not None
        assert len(output.predictions) == len(ALL_HEADS)
        output.total.backward()


class TestJunctionHead:
    def test_junction_head_runs_and_produces_a_finite_loss(self):
        model, batch = build_with_batch(ALL_HEADS)
        output = model.loss(batch, return_predictions=True)
        assert output.predictions is not None
        assert "splice_sites_junction" in output.predictions
        assert torch.isfinite(output.total)


class TestComponentGradients:
    @pytest.fixture(autouse=True)
    def _float32_policy(self):
        with dtype_policy_context(FLOAT32_DTYPE_POLICY, "cpu"):
            yield

    def test_standardized_conv1d(self):
        conv = StandardizedConv1d(16, 32, kernel_size=5)
        x = torch.randn(2, 16, 64, requires_grad=True)
        conv(x).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for attr in ("weight", "scale", "bias"):
            _healthy(getattr(conv, attr), f"StandardizedConv1d.{attr}")

    def test_conv_block(self):
        block = ConvBlock(16, 32, sync_bn=False)
        x = torch.randn(2, 16, 64, requires_grad=True)
        block(x).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        _healthy(block.conv.weight, "ConvBlock.conv.weight")

    def test_down_res_block(self):
        block = DownResBlock(16, 32, sync_bn=False)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        (out[0] if isinstance(out, tuple) else out).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_up_res_block(self):
        block = UpResBlock(32, 16, sync_bn=False)
        x = torch.randn(2, 32, 32, requires_grad=True)
        skip = torch.randn(2, 16, 64)
        block(x, skip).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_batch_norm(self):
        norm = BatchNorm(16, sync=False, channels_dim=1)
        norm.train()
        x = torch.randn(2, 16, 64, requires_grad=True)
        norm(x).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        _healthy(norm.scale, "BatchNorm.scale")
