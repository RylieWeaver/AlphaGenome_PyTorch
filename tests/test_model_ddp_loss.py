# Run this optional DDP test from the repository root with:
#
#   torchrun --standalone --nproc_per_node=1 --module tests.test_model_ddp_loss
#
# A normal `pytest tests` invocation skips it because no distributed process
# group has been configured by torchrun.

# External
from collections.abc import Iterator
import os

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Internal
from alphagenome_pt import (
    AlphaGenome,
    DataBatch,
    HeadName,
    small_alphagenome,
    synthetic_metadata,
    synthetic_mlm,
)
from alphagenome_pt.distributed import is_dist

from .helpers import DNA_SEQUENCE


def _setup_dist_from_torchrun() -> tuple[int, int]:
    if is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return dist.get_world_size(), local_rank

    if "RANK" not in os.environ:
        pytest.skip(
            "Run with `torchrun --nproc_per_node=1 --module "
            "tests.test_model_ddp_loss`."
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return dist.get_world_size(), local_rank


@pytest.fixture(scope="module")
def distributed_context() -> Iterator[tuple[int, int]]:
    initialized_here = not is_dist()
    context = _setup_dist_from_torchrun()
    try:
        yield context
    finally:
        if initialized_here:
            dist.destroy_process_group()


@pytest.fixture
def ddp_loss_setup(
    distributed_context: tuple[int, int],
) -> Iterator[tuple[AlphaGenome, DDP, DataBatch]]:
    """Create a DDP model under a process group started by torchrun."""
    _, local_rank = distributed_context

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        ddp_kwargs = {"device_ids": [local_rank]}
    else:
        device = torch.device("cpu")
        ddp_kwargs = {}

    torch.manual_seed(0)
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(
        metadata,
        max_seq_len=len(DNA_SEQUENCE),
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
    ).to(device)
    model.eval()
    ddp_model = DDP(
        model,
        find_unused_parameters=True,
        **ddp_kwargs,
    )
    batch = model.as_data_batch(DNA_SEQUENCE)
    batch.mlm = synthetic_mlm(
        batch_size=1,
        seq_len=len(DNA_SEQUENCE),
    )
    batch.to(device)

    yield model, ddp_model, batch


def test_ddp_loss_gradients_match_when_total_is_computed_after_forward(
    ddp_loss_setup,
):
    model, ddp_model, batch = ddp_loss_setup

    returned = ddp_model(batch, mode="loss")
    returned.total.backward()
    expected_total = returned.total.detach().clone()
    expected_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    assert any(gradient is not None for gradient in expected_gradients.values())

    model.zero_grad(set_to_none=True)
    computed = ddp_model(batch, mode="loss")
    computed_total = computed.tree.total_loss()
    torch.testing.assert_close(computed_total, computed.total)
    torch.testing.assert_close(computed_total, expected_total)
    computed_total.backward()

    for name, parameter in model.named_parameters():
        expected = expected_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
