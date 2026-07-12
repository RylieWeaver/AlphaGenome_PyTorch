# Adapted from genomicsxai/alphagenome-pytorch,
# tests/integration/test_backward.py (Apache-2.0). No golden numbers.
#
# Deviations from the source file:
#   - genomicsxai/alphagenome-pytorch hand-rolls a compute_combined_loss helper; alphagenome_pt has model.loss(batch)
#     built in, returning (total_loss, scalars, predictions)
#   - component names differ: genomicsxai/alphagenome-pytorch's MHABlock is this implementation's MHA, genomicsxai/alphagenome-pytorch's
#     RMSBatchNorm is this implementation's BatchNorm(rms_norm=True)
#   - added test_zero_init_starves_scale_params_on_the_first_step, which has no
#     genomicsxai/alphagenome-pytorch equivalent: alphagenome_pt zero-initialises StandardizedConv1d.weight, so
#     87 of 363 parameters have no gradient at step 0. That is an init artifact,
#     not a bug, but the naive port of genomicsxai/alphagenome-pytorch's sweep fails without accounting
#     for it, so the sweep here takes one optimiser step first.
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

from ._helpers import ALL_HEADS, HEADS_WITHOUT_JUNCTION, build_with_batch

pytestmark = pytest.mark.integration


def _healthy(param, name):
    assert param.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(param.grad).all(), f"{name}: non-finite gradient"
    norm = param.grad.norm().item()
    # A gradient of 1e-30 is dead in practice even though it technically exists.
    assert norm > 1e-12, f"{name}: gradient vanished ({norm})"
    assert norm < 1e8, f"{name}: gradient exploded ({norm})"


class TestFullModelBackward:
    def test_every_parameter_receives_a_gradient(self):
        # The most valuable test in the tier. A parameter with no gradient never
        # trains: it sits at its random init for the whole run and nothing says so.
        # One optimiser step is taken first because alphagenome_pt zero-initialises the
        # standardized convs, which starves their scale params at step 0 only.
        model, batch = build_with_batch(HEADS_WITHOUT_JUNCTION)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        opt.zero_grad()
        model.loss(batch)[0].backward()
        opt.step()

        opt.zero_grad()
        model.loss(batch)[0].backward()

        dead = [n for n, p in model.named_parameters()
                if p.grad is None or p.grad.norm() == 0]
        assert not dead, (f"{len(dead)} parameters received no gradient:\n"
                          + "\n".join(dead[:20]))

    def test_no_nan_or_inf_in_any_gradient(self):
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch)[0].backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, bad

    def test_optimizer_step_changes_the_weights(self):
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.loss(batch)[0].backward()
        opt.step()
        assert any(not torch.equal(a, b)
                   for a, b in zip(before, model.parameters()))

    def test_all_heads_can_be_trained_together(self):
        # the existing tests in tests/ only ever enable one head at a time, so the combined
        # loss path had no coverage until now.
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        total, scalars, predictions = model.loss(batch)
        assert torch.isfinite(total)
        assert len(predictions) == len(ALL_HEADS)
        total.backward()


class TestZeroInitialisation:
    def test_standardized_conv_weights_start_at_zero(self):
        # Documents why the sweep above needs a warm-up step.
        conv = StandardizedConv1d(8, 16, kernel_size=5)
        assert bool((conv.weight == 0).all())

    def test_zero_init_starves_scale_params_on_the_first_step(self):
        # With w == 0, weight standardization gives a zero kernel whatever the
        # scale is, so d(loss)/d(scale) == 0 at step 0. The weights themselves do
        # get a gradient, so this clears after one step. Pinned so that a change
        # in the init scheme is noticed.
        model, batch = build_with_batch(HEADS_WITHOUT_JUNCTION)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        model.loss(batch)[0].backward()
        dead_before = sum(1 for p in model.parameters()
                          if p.grad is None or p.grad.norm() == 0)
        opt.step()

        opt.zero_grad()
        model.loss(batch)[0].backward()
        dead_after = sum(1 for p in model.parameters()
                         if p.grad is None or p.grad.norm() == 0)

        assert dead_before > 0, "expected starved scale params at init"
        assert dead_after == 0, f"{dead_after} params still dead after one step"


class TestJunctionHeadGradients:
    def test_junction_head_is_starved_by_an_untrained_classifier(self):
        # Not a bug. The junction head only predicts at positions the
        # classification head flags as splice sites, and an untrained classifier
        # flags none, so every position is masked out. genomicsxai/alphagenome-pytorch excludes this head
        # from its sweep for the same reason.
        model, batch = build_with_batch(ALL_HEADS)
        model.train()
        model.loss(batch)[0].backward()

        junction = [(n, p) for n, p in model.named_parameters()
                    if n.startswith("_heads.splice_sites_junction.")]
        assert junction
        starved = [n for n, p in junction
                   if p.grad is None or p.grad.norm() == 0]
        assert starved, "junction head unexpectedly received gradients"

    def test_junction_head_runs_and_produces_a_finite_loss(self):
        model, batch = build_with_batch(ALL_HEADS)
        total, scalars, predictions = model.loss(batch)
        assert "splice_sites_junction" in predictions
        assert torch.isfinite(total)


class TestComponentGradients:
    def test_standardized_conv1d(self):
        conv = StandardizedConv1d(16, 32, kernel_size=5)
        torch.nn.init.normal_(conv.weight, std=0.1)   # zero init would starve scale
        x = torch.randn(2, 16, 64, requires_grad=True)
        conv(x).pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for attr in ("weight", "scale", "bias"):
            _healthy(getattr(conv, attr), f"StandardizedConv1d.{attr}")

    def test_conv_block(self):
        block = ConvBlock(16, 32, sync_bn=False)
        torch.nn.init.normal_(block.conv.weight, std=0.1)
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
