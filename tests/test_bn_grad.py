import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from alphagenome_pt.distributed import dist_sum, is_dist


class DummyBatchNorm(nn.Module):
    """Minimal RMS batch norm to test gradients through distributed variance."""

    def __init__(self, num_channels: int, sync: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.sync = sync
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sync and is_dist():
            sum_square = torch.square(x).sum(dim=[0, 1])
            count = torch.ones_like(x).sum(dim=[0, 1])
            var = dist_sum(sum_square) / dist_sum(count)
        else:
            var = torch.square(x).mean(dim=[0, 1])

        rms = torch.sqrt(var + self.eps).view(1, 1, -1)
        return x / rms


def _setup_dist_from_torchrun() -> tuple[int, int]:
    if is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return dist.get_world_size(), local_rank

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("Run with `torchrun --nproc_per_node=2 pytest tests/test_bn_grad.py -q`.")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, local_rank


def test_dist_sum_batch_norm_grad_matches_local_when_inputs_match():
    """Optional distributed test for gradient propagation through dist_sum.

    Normal pytest runs skip this test. Launch it explicitly with torchrun when
    validating sync batch norm on distributed hardware.
    """
    world_size, local_rank = _setup_dist_from_torchrun()
    assert world_size >= 2

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        ddp_kwargs = {"device_ids": [local_rank]}
    else:
        device = torch.device("cpu")
        ddp_kwargs = {}

    torch.manual_seed(42)
    batch_size, seq_len, channels = 2, 8, 32
    x = torch.randn(batch_size, seq_len, channels, device=device)
    x_baseline = x.clone().detach().requires_grad_(True)
    x_ddp = x.clone().detach().requires_grad_(True)

    model = DummyBatchNorm(channels, sync=True).to(device)
    ddp_model = DDP(model, **ddp_kwargs)

    y_baseline = model(x_baseline)
    loss_baseline = (y_baseline**2).mean()
    loss_baseline.backward()

    y_ddp = ddp_model(x_ddp)
    loss_ddp = (y_ddp**2).mean()
    loss_ddp.backward()

    assert x_baseline.grad is not None
    assert x_ddp.grad is not None
    assert torch.isfinite(x_baseline.grad).all()
    assert torch.isfinite(x_ddp.grad).all()
    assert torch.allclose(x_baseline.grad, x_ddp.grad, atol=1e-5, rtol=1e-5)
