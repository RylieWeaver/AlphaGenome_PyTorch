# Component-gradient tests in this file are adapted from
# genomicsxai/alphagenome-pytorch, tests/integration/test_backward.py
# (Apache-2.0). No golden numbers are carried over.
#
# Deviations from the source file:
#   - sync_bn=False everywhere in the component tests, so they run single-process

import pytest
import torch

from alphagenome_pt.convolutions import (
    ConvBlock,
    DownResBlock,
    StandardizedConv1d,
    UpResBlock,
)
from alphagenome_pt.precision import FLOAT32_DTYPE_POLICY, dtype_policy_context


def _healthy(param, name):
    assert param.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(param.grad).all(), f"{name}: non-finite gradient"
    norm = param.grad.norm().item()
    # Treat gradients at or below 1e-12 as vanished in practice.
    assert norm > 1e-12, f"{name}: gradient vanished ({norm})"
    assert norm < 1e8, f"{name}: gradient exploded ({norm})"


class TestComponentGradients:
    @pytest.fixture(autouse=True)
    def _float32_policy(self):
        torch.manual_seed(0)
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
