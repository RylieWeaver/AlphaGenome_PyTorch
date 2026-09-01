# Local BatchNorm gradient coverage is adapted from
# genomicsxai/alphagenome-pytorch, tests/integration/test_backward.py
# (Apache-2.0). Sync wiring coverage is original work.

# External
import pytest
import torch

# Internal
from alphagenome_pt import small_alphagenome
from alphagenome_pt.layers import BatchNorm
from alphagenome_pt.precision import FLOAT32_DTYPE_POLICY, dtype_policy_context


class TestBatchNorm:
    @pytest.mark.parametrize("sync_bn", [True, False])
    def test_alphagenome_sync_bn_wires_through(self, sync_bn: bool):
        model = small_alphagenome(sync_bn=sync_bn)
        batch_norms = [
            module for module in model.modules()
            if isinstance(module, BatchNorm)
        ]

        assert batch_norms
        assert all(module.sync is sync_bn for module in batch_norms)

    def test_local_batch_norm_gradient_is_healthy(self):
        torch.manual_seed(0)
        with dtype_policy_context(FLOAT32_DTYPE_POLICY, "cpu"):
            norm = BatchNorm(16, sync=False, channels_dim=1)
            norm.train()
            x = torch.randn(2, 16, 64, requires_grad=True)
            norm(x).pow(2).mean().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert norm.scale.grad is not None
        assert torch.isfinite(norm.scale.grad).all()
        gradient_norm = norm.scale.grad.norm().item()
        assert 1e-12 < gradient_norm < 1e8
