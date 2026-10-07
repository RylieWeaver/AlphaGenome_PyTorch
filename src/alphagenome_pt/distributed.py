# External
import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal, get_args

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:  # PyTorch 2.0 and 2.1
    from torch.distributed._tensor import DeviceMesh

# Internal
from .schemas import DataBatch

if TYPE_CHECKING:  # prevent circular import error
    from .model import AlphaGenome



### HELPERS ###
def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_rank0():
    return (not is_dist()) or dist.get_rank() == 0


def dist_print(*args, **kwargs):
    if is_rank0():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _prepare_fsdp_kwargs(**fsdp_kwargs):
    """Validate sharding options and return them with a hybrid-sharding flag."""
    sharding_strategy = fsdp_kwargs.get("sharding_strategy")
    hybrid = sharding_strategy is not None and sharding_strategy.name in (
        "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"
    )
    if not hybrid:
        return fsdp_kwargs, False

    if fsdp_kwargs.get("process_group") is not None:
        raise ValueError(
            "Hybrid FSDP uses device_mesh to configure its groups. "
            "Leave process_group unset and specify device_mesh."
        )

    fsdp_mesh = fsdp_kwargs.get("device_mesh")
    if fsdp_mesh is None:
        # No FSDP layout supplied: using default sharding over WORLD.
        fsdp_kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
        return fsdp_kwargs, False

    if fsdp_mesh.ndim != 2:
        raise ValueError("Hybrid FSDP device_mesh must be 2D: replication, sharding.")
    ranks = fsdp_mesh.mesh
    if sorted(ranks.flatten().tolist()) != list(range(dist.get_world_size())):
        raise ValueError("Hybrid FSDP device_mesh must contain every WORLD rank once.")
    # Process groups sort members internally without changing mesh membership.
    # FSDP assigns shards by that order within each row, while replication
    # combines gradients among each column's members. Different row permutations
    # can therefore mix gradients for different parameter shards. Requiring
    # ascending rows is a simple sufficient check that column members own
    # matching shards.
    if not torch.equal(ranks, ranks.sort(dim=1).values):
        raise ValueError(
            "This wrapper requires ascending rank order within each hybrid FSDP "
            "sharding row so replication groups combine matching parameter shards."
        )
    if fsdp_mesh.size(1) < 2:
        # FSDP1's NO_SHARD fallback would skip replication-group reduction.
        # Discard this FSDP layout and fully shard over WORLD instead.
        fsdp_kwargs.pop("device_mesh")
        fsdp_kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
        return fsdp_kwargs, False

    return fsdp_kwargs, True


### OBJECTS ###
@dataclass(frozen=True)
class ParallelContext:
    """DP/SP topology and the group used to combine loss statistics.

    Missing axes default to size 1. With SP, the mesh defines both logical
    axes. Without SP, loss_group is also the DP group when set, so only
    DDP/FSDP process groups do not require a mesh.

    Mesh indices identify positions in the supplied layout. Group ranks
    follow PyTorch's rank-sorted convention.
    """

    mesh: DeviceMesh | None = None
    loss_group: dist.ProcessGroup | None = None

    def _mesh_dim(self, name: str) -> int | None:
        names = None if self.mesh is None else self.mesh.mesh_dim_names
        if names is None or name not in names:
            return None
        return names.index(name)

    @property
    def dp_enabled(self) -> bool:
        # NOTE: "Enabled" means a DP group is configured, even if its size is 1.
        return (
            (not self.sp_enabled and self.loss_group is not None)   # Native DDP/FSDP group without SP.
            or self._mesh_dim("dp") is not None                     # DP mesh axis, with or without SP.
        )

    @property
    def sp_enabled(self) -> bool:
        return self._mesh_dim("sp") is not None

    @property
    def world_group(self):
        return dist.group.WORLD

    def _mesh_group(self, name: str):
        mesh_dim = self._mesh_dim(name)
        if self.mesh is None or mesh_dim is None:
            raise RuntimeError(
                f"ParallelContext has no {name!r} mesh dimension."
            )
        if hasattr(self.mesh, "get_group"):
            return self.mesh.get_group(mesh_dim)
        return self.mesh.get_dim_groups(mesh_dim)

    @property
    def dp_group(self):
        if not self.sp_enabled and self.loss_group is not None:
            return self.loss_group
        return self._mesh_group("dp")

    @property
    def sp_group(self):
        return self._mesh_group("sp")

    @property
    def world_rank(self) -> int:
        return dist.get_rank()

    @property
    def dp_rank(self) -> int:
        return dist.get_rank(self.dp_group) if self.dp_enabled else 0

    @property
    def sp_rank(self) -> int:
        return dist.get_rank(self.sp_group) if self.sp_enabled else 0

    @property
    def dp_mesh_index(self) -> int:
        mesh_dim = self._mesh_dim("dp")
        return self.mesh.get_coordinate()[mesh_dim] if mesh_dim is not None else self.dp_rank

    @property
    def sp_mesh_index(self) -> int:
        mesh_dim = self._mesh_dim("sp")
        return self.mesh.get_coordinate()[mesh_dim] if mesh_dim is not None else self.sp_rank

    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    @property
    def dp_size(self) -> int:
        return dist.get_world_size(self.dp_group) if self.dp_enabled else 1

    @property
    def sp_size(self) -> int:
        return dist.get_world_size(self.sp_group) if self.sp_enabled else 1

    def sp_bounds(self, global_length: int) -> tuple[int, int]:
        """Return this SP rank's equal-sized, contiguous share of a length."""
        if global_length % self.sp_size:
            raise ValueError(
                f"Length {global_length} must be divisible by SP size "
                f"{self.sp_size}."
            )
        chunk_size = global_length // self.sp_size
        start = self.sp_rank * chunk_size
        return start, start + chunk_size

    def sp_shard(self, x: torch.Tensor, *, dim: int) -> torch.Tensor:
        start, end = self.sp_bounds(x.shape[dim])
        return x.narrow(dim, start, end - start)  # equivalent to slicing along a dim

    def sp_buffered_global_bounds(
        self, global_length: int, *, buffer_size: int
    ) -> tuple[int, int]:
        # NOTE: Extend the SP bounds by the buffer size
        start, end = self.sp_bounds(global_length)
        read_start = max(0, start - buffer_size)
        read_end = min(global_length, end + buffer_size)
        return read_start, read_end

    def sp_buffered_local_bounds(
        self, global_length: int, *, buffer_size: int
    ) -> tuple[int, int]:
        start, end = self.sp_bounds(global_length)
        read_start, _ = self.sp_buffered_global_bounds(
            global_length, buffer_size=buffer_size
        )
        return start - read_start, end - read_start

    def sp_buffered_shard(
        self, x: torch.Tensor, *, dim: int, buffer_size: int
    ) -> torch.Tensor:
        read_start, read_end = self.sp_buffered_global_bounds(
            x.shape[dim], buffer_size=buffer_size
        )
        return x.narrow(
            dim, read_start, read_end - read_start
        )  # equivalent to slicing along a dim


class _ParallelAlphaGenomePublicAPI:
    """Route AlphaGenome methods to preserve wrapper forward hooks."""

    def save(self, *args, **kwargs) -> None:
        if is_rank0():
            self.module.save(*args, **kwargs)

    def as_data_batch(self, *args, **kwargs):
        return self.module.as_data_batch(*args, **kwargs)

    def embed(self, *args, **kwargs):
        return self(*args, mode="embed", **kwargs)

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        return self(*args, mode="predict", **kwargs)

    def loss(self, *args, **kwargs):
        return self(*args, mode="loss", **kwargs)


class DDP_AlphaGenome(_ParallelAlphaGenomePublicAPI, DDP):
    """DP wrap AlphaGenome (and optionally SP from mesh)."""

    def __init__(
        self,
        model: "AlphaGenome",
        *,
        mesh: DeviceMesh | None = None,
        sync_bn: bool = True,
        **ddp_kwargs,
    ) -> None:
        model.set_sync_bn(sync_bn)
        # Selected heads may leave some trainable parameters unused.
        ddp_kwargs.setdefault("find_unused_parameters", True)
        super().__init__(model, **ddp_kwargs)
        # NOTE: self.process_group is stored by DDP.__init__()
        self.parallel_context = ParallelContext(
            mesh=mesh,
            loss_group=self.process_group,
        )
        if (
            self.parallel_context.sp_enabled
            and dist.get_world_size(self.process_group) != dist.get_world_size()
        ):
            raise ValueError(
                "SP + DDP requires a process group containing every world rank."
            )

    def forward(self, *args, **kwargs):
        kwargs["parallel_context"] = self.parallel_context
        return super().forward(*args, **kwargs)


class FSDP_AlphaGenome(_ParallelAlphaGenomePublicAPI, FSDP):
    """FSDP wraps AlphaGenome (and optionally SP from mesh).

    Hybrid strategies use a separate 2D ``device_mesh`` with replication
    first and sharding second. If it is absent or has one rank per sharding
    group, hybrid requests use FULL_SHARD over WORLD, independently of the
    DP/SP mesh. Hybrid loss statistics always cover WORLD.
    """

    def __init__(
        self,
        model: "AlphaGenome",
        *,
        mesh: DeviceMesh | None = None,
        sync_bn: bool = True,
        **fsdp_kwargs,
    ) -> None:
        fsdp_kwargs, hybrid = _prepare_fsdp_kwargs(**fsdp_kwargs)

        model.set_sync_bn(sync_bn)
        fsdp_kwargs.setdefault("use_orig_params", True)
        super().__init__(model, **fsdp_kwargs)
        # Hybrid FSDP averages gradients across both mesh axes. Loss
        # statistics must therefore cover both axes, not just the sharding group.
        loss_group = dist.group.WORLD if hybrid else self.process_group
        self.parallel_context = ParallelContext(
            mesh=mesh,
            loss_group=loss_group,
        )
        if (
            self.parallel_context.sp_enabled
            and dist.get_world_size(loss_group) != dist.get_world_size()
        ):
            raise ValueError(
                "SP + FSDP requires a process group containing every world rank."
            )

    def forward(self, *args, **kwargs):
        kwargs["parallel_context"] = self.parallel_context
        return super().forward(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        """Collect and save full parameters on rank zero.

        Every rank in the FSDP process group must call this collective method.
        """
        with FSDP.summon_full_params(
            self,
            recurse=True,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        ):
            super().save(*args, **kwargs)


ParallelStrategy = Literal[
    "none",
    "sp",
    "ddp",
    "fsdp",
    "sp_ddp",
    "sp_fsdp",
]


def distribute_alphagenome(
    model: "AlphaGenome",
    *,
    strategy: ParallelStrategy = "none",
    mesh: DeviceMesh | None = None,
    sync_bn: bool = True,
    **dp_kwargs,
) -> nn.Module:
    """Apply the selected sequence and parameter parallelism strategy.

    ``sp``, ``sp_ddp``, and ``sp_fsdp`` require a topology ``mesh``
    with an ``"sp"`` dimension, typically named ``("dp", "sp")``.

    Every distributed strategy wraps AlphaGenome directly in DDP/FSDP.
    With SP, gradients and loss statistics cover WORLD. Hybrid FSDP accepts
    a separate ``device_mesh`` for replication/sharding. If it is absent or
    has one rank per sharding group, hybrid requests use FULL_SHARD over
    WORLD, independently of the DP/SP mesh.
    """
    valid_strategies = get_args(ParallelStrategy)
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy must be one of {valid_strategies}, got {strategy!r}."
        )

    if strategy == "none":
        if dp_kwargs:
            unexpected = ", ".join(sorted(dp_kwargs))
            raise TypeError(
                f"strategy='none' received DP options: {unexpected}"
            )
        if mesh is not None:
            raise TypeError("strategy='none' does not accept mesh")
        return model

    sp_enabled = mesh is not None and "sp" in (mesh.mesh_dim_names or ())
    if strategy == "sp":
        if not sp_enabled:
            raise TypeError(
                "strategy='sp' requires a mesh with an 'sp' dimension"
            )
        if is_rank0():
            warnings.warn(
                "strategy='sp' uses DDP internally to synchronize replicated "
                "parameter gradients and is equivalent to strategy='sp_ddp'.",
                stacklevel=2,
            )
        strategy = "sp_ddp"

    if strategy == "ddp":
        if sp_enabled:
            raise TypeError(
                "strategy='ddp' does not enable SP; use strategy='sp_ddp'."
            )
        return DDP_AlphaGenome(
            model,
            mesh=mesh,
            sync_bn=sync_bn,
            **dp_kwargs,
        )

    if strategy == "fsdp":
        if sp_enabled:
            raise TypeError(
                "strategy='fsdp' does not enable SP; use strategy='sp_fsdp'."
            )
        return FSDP_AlphaGenome(
            model,
            mesh=mesh,
            sync_bn=sync_bn,
            **dp_kwargs,
        )

    if strategy == "sp_ddp":
        if not sp_enabled:
            raise TypeError(
                "strategy='sp_ddp' requires a mesh with an 'sp' dimension"
            )
        return DDP_AlphaGenome(
            model,
            mesh=mesh,
            sync_bn=sync_bn,
            **dp_kwargs,
        )

    if strategy == "sp_fsdp":
        if not sp_enabled:
            raise TypeError(
                "strategy='sp_fsdp' requires a mesh with an 'sp' dimension"
            )
        return FSDP_AlphaGenome(
            model,
            mesh=mesh,
            sync_bn=sync_bn,
            **dp_kwargs,
        )


### DATA DISTRIBUTION ###
def sp_broadcast_data_batch(
    batch: DataBatch | None,
    *,
    parallel_context: ParallelContext,
    device: torch.device | str | None = None,
) -> DataBatch:
    """Broadcast a DataBatch from one rank to its sequence-parallel peers.

    Every rank in the SP group must call this function. The process with
    ``sp_mesh_index == 0`` must provide a DataBatch; batches passed by other
    ranks are ignored. Tensor fields are broadcast without autograd history
    and returned on ``device``.
    """
    if not parallel_context.sp_enabled:
        raise ValueError(
            "sp_broadcast_data_batch requires an SP-enabled context."
        )

    group = parallel_context.sp_group
    assert parallel_context.mesh is not None
    mesh_device_type = parallel_context.mesh.device_type
    device = torch.device(mesh_device_type if device is None else device)
    if device.type != mesh_device_type:
        raise ValueError(
            f"Broadcast device type {device.type!r} must match the "
            f"SP mesh device type {mesh_device_type!r}."
        )

    # Resolve SP mesh index zero to its global rank, keeping other axes fixed.
    source_coordinate = list(parallel_context.mesh.get_coordinate())
    source_coordinate[parallel_context._mesh_dim("sp")] = 0
    global_src = int(parallel_context.mesh.mesh[tuple(source_coordinate)].item())
    is_source = parallel_context.sp_mesh_index == 0

    # For batch and every tensor field in batch:
    # - validate the field
    # - collect specs to specify the broadcasting
    manifest = None
    if is_source:
        error = None
        field_specs = []
        if not isinstance(batch, DataBatch):
            error = "The source SP rank must provide a DataBatch."
        else:
            for field in fields(DataBatch):
                value = getattr(batch, field.name)
                if isinstance(value, torch.Tensor):
                    if value.layout != torch.strided:
                        error = (
                            f"DataBatch field {field.name!r} must be a dense tensor."
                        )
                        break
                    field_specs.append(
                        (field.name, "tensor", tuple(value.shape), value.dtype)
                    )
                else:
                    field_specs.append((field.name, "object", value))
        manifest = {"error": error, "fields": field_specs}

    # NOTE: Broadcast the manifest (must be in a list to broadcast)
    manifest_list = [manifest]
    dist.broadcast_object_list(manifest_list, src=global_src, group=group)
    manifest = manifest_list[0]

    # Check manifest["error"]
    if manifest["error"] is not None:
        raise TypeError(manifest["error"])

    # Broadcast as specified by manifest["fields"]
    values = {}
    for name, kind, *metadata in manifest["fields"]:
        if kind == "object":
            values[name] = metadata[0]
            continue

        shape, dtype = metadata
        if is_source:
            tensor = getattr(batch, name).detach().to(device).contiguous()
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(tensor, src=global_src, group=group)
        values[name] = tensor

    return DataBatch(**values)


### COLLECTIVES ###
def _all_gather_dim0(x: torch.Tensor, group) -> torch.Tensor:
    """Gather equal-sized tensors into one tensor along dimension zero."""
    group_size = dist.get_world_size(group)
    output_shape = (x.shape[0] * group_size, *x.shape[1:])

    if hasattr(dist, "all_gather_single"):
        gathered = x.new_empty(output_shape)
        dist.all_gather_single(gathered, x, group=group)
    elif str(dist.get_backend(group)).lower() == "nccl":
        gathered = x.new_empty(output_shape)
        dist.all_gather_into_tensor(gathered, x, group=group)
    else:
        gathered_parts = [torch.empty_like(x) for _ in range(group_size)]
        dist.all_gather(gathered_parts, x, group=group)
        gathered = torch.cat(gathered_parts, dim=0)
    return gathered


def _sum_reduce_scatter_dim0(x: torch.Tensor, group) -> torch.Tensor:
    """Sum and scatter equal-sized chunks along dimension zero."""
    group_size = dist.get_world_size(group)
    output_shape = (x.shape[0] // group_size, *x.shape[1:])

    if hasattr(dist, "reduce_scatter_single"):
        reduced = x.new_empty(output_shape)
        dist.reduce_scatter_single(reduced, x, group=group)
    elif str(dist.get_backend(group)).lower() == "nccl":
        reduced = x.new_empty(output_shape)
        dist.reduce_scatter_tensor(reduced, x, group=group)
    else:
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        start = dist.get_rank(group) * output_shape[0]
        reduced = x.narrow(0, start, output_shape[0]).contiguous()
    return reduced


class _AllGatherForwardReduceScatterBackward(Function):
    """All-gather forward and sum reduce-scatter backward.

    The explicit backward and backend fallback keep this on public APIs across
    supported PyTorch versions.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim

        x_dim0 = x.movedim(dim, 0).contiguous()
        gathered = _all_gather_dim0(x_dim0, group)
        return gathered.movedim(0, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dim0 = grad_output.movedim(ctx.dim, 0).contiguous()
        grad_input = _sum_reduce_scatter_dim0(grad_dim0, ctx.group)
        # NOTE: Lots of `None` returns because the backward must return
        # the same number of arguments as passed to forward()
        return grad_input.movedim(0, ctx.dim), None, None


def all_gather(
    x: torch.Tensor, *, dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Autograd-aware all-gather along ``dim`` over an explicit group."""
    return _AllGatherForwardReduceScatterBackward.apply(x, group, dim)


class _NeighborGatherForwardNeighborScatterAddBackward(Function):
    """Exchange between left and right neighbors but nowhere else.

    Backward returns neighbor gradients to their owning ranks and adds them to the
    corresponding edges of the local gradient.
    """

    @staticmethod
    def _exchange(
        x: torch.Tensor,
        *,
        buffer_size: int,
        group,
        rank: int,
        world_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        send_left = x[:buffer_size].contiguous() if rank > 0 else None
        send_right = (
            x[-buffer_size:].contiguous() if rank + 1 < world_size else None
        )
        recv_left = torch.empty_like(send_left) if send_left is not None else None
        recv_right = (
            torch.empty_like(send_right) if send_right is not None else None
        )
        ops = []
        if recv_left is not None:
            left_peer = dist.get_global_rank(group, rank - 1)
            ops.extend((
                dist.P2POp(dist.irecv, recv_left, left_peer, group),
                dist.P2POp(dist.isend, send_left, left_peer, group),
            ))
        if recv_right is not None:
            right_peer = dist.get_global_rank(group, rank + 1)
            ops.extend((
                dist.P2POp(dist.irecv, recv_right, right_peer, group),
                dist.P2POp(dist.isend, send_right, right_peer, group),
            ))
        # NOTE: dist.batch_isend_irecv() launches all the communications asynchronously
        comms = dist.batch_isend_irecv(ops)
        # Wait until every communication has completed
        for comm in comms:
            comm.wait()
        return recv_left, recv_right

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, group, dim: int, buffer_size: int
    ) -> torch.Tensor:
        """[B, S_local, C] -> [B, S_local + B_left + B_right, C]."""
        if buffer_size < 0 or buffer_size > x.shape[dim]:
            raise ValueError(
                f"buffer_size must be in [0, {x.shape[dim]}], got {buffer_size}."
            )

        ctx.group = group
        ctx.dim = dim
        ctx.buffer_size = buffer_size
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.local_size = x.shape[dim]
        if buffer_size == 0 or ctx.world_size == 1:
            return x

        x_dim0 = x.movedim(dim, 0).contiguous()
        recv_left, recv_right = (
            _NeighborGatherForwardNeighborScatterAddBackward._exchange(
                x_dim0,
                buffer_size=buffer_size,
                group=group,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
        )
        pieces = [piece for piece in (recv_left, x_dim0, recv_right) if piece is not None]
        return torch.cat(pieces, dim=0).movedim(0, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """[B, S_local + B_left + B_right, C] -> [B, S_local, C].

        Autograd requires ``grad_input.shape == x.shape``. Buffer gradients
        are returned to their owning ranks and added accordingly.
        """
        if ctx.buffer_size == 0 or ctx.world_size == 1:
            return grad_output, None, None, None

        grad_dim0 = grad_output.movedim(ctx.dim, 0).contiguous()
        start = ctx.buffer_size if ctx.rank > 0 else 0
        length = ctx.local_size
        grad_input = grad_dim0.narrow(0, start, length).clone()
        recv_left, recv_right = (
            _NeighborGatherForwardNeighborScatterAddBackward._exchange(
                grad_dim0,
                buffer_size=ctx.buffer_size,
                group=ctx.group,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
        )
        if recv_left is not None:
            grad_input[:ctx.buffer_size].add_(recv_left)
        if recv_right is not None:
            grad_input[-ctx.buffer_size:].add_(recv_right)
        return grad_input.movedim(0, ctx.dim), None, None, None


def neighbor_gather(
    x: torch.Tensor,
    *,
    dim: int,
    buffer_size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Autograd-aware exchange of adjacent-rank buffers along ``dim``."""
    return _NeighborGatherForwardNeighborScatterAddBackward.apply(
        x, group, dim, buffer_size
    )


class _AllReduceSumForwardAllReduceSumBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


class _ContiguousSubgroupSumForwardContiguousSubgroupSumBackward(Function):
    """Sum within contiguous blocks of process-group ranks in both passes."""

    @staticmethod
    def _sum(tensor, *, group, ranks_per_subgroup):
        group_rank = dist.get_rank(group)
        subgroup_start = (
            group_rank // ranks_per_subgroup
        ) * ranks_per_subgroup
        subgroup_rank = group_rank - subgroup_start

        send = tensor.contiguous()
        parts = [None] * ranks_per_subgroup
        parts[subgroup_rank] = send
        ops = []
        for peer_rank in range(
            subgroup_start, subgroup_start + ranks_per_subgroup
        ):
            if peer_rank == group_rank:
                continue
            global_peer = dist.get_global_rank(group, peer_rank)
            peer_tensor = torch.empty_like(send)
            parts[peer_rank - subgroup_start] = peer_tensor
            ops.extend((
                dist.P2POp(dist.irecv, peer_tensor, global_peer, group),
                dist.P2POp(dist.isend, send, global_peer, group),
            ))

        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()

        summed = parts[0].clone()
        for part in parts[1:]:
            summed.add_(part)
        return summed

    @staticmethod
    def forward(ctx, tensor, group, ranks_per_subgroup):
        ctx.group = group
        ctx.ranks_per_subgroup = ranks_per_subgroup
        return _ContiguousSubgroupSumForwardContiguousSubgroupSumBackward._sum(
            tensor,
            group=group,
            ranks_per_subgroup=ranks_per_subgroup,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad = _ContiguousSubgroupSumForwardContiguousSubgroupSumBackward._sum(
            grad_output,
            group=ctx.group,
            ranks_per_subgroup=ctx.ranks_per_subgroup,
        )
        return grad, None, None


def dist_sum(tensor, group=None, *, ranks_per_subgroup=None):
    """Autograd-aware sum over a group or contiguous rank subgroups."""
    if ranks_per_subgroup is None:
        return _AllReduceSumForwardAllReduceSumBackward.apply(tensor, group)

    group = dist.group.WORLD if group is None else group
    group_size = dist.get_world_size(group)
    if (
        not 1 <= ranks_per_subgroup <= group_size
        or group_size % ranks_per_subgroup
    ):
        raise ValueError(
            f"{ranks_per_subgroup=} must divide process-group size "
            f"{group_size}."
        )
    if ranks_per_subgroup == 1:
        return tensor
    if ranks_per_subgroup == group_size:
        return _AllReduceSumForwardAllReduceSumBackward.apply(tensor, group)
    return _ContiguousSubgroupSumForwardContiguousSubgroupSumBackward.apply(
        tensor, group, ranks_per_subgroup
    )


class _AllToAllTransposeForwardAllToAllTransposeBackward(Function):
    """Transpose a row-sharded square matrix with an all-to-all exchange.

    In a nutshell: have local rows of X -> get local rows of Xᵀ

    Each rank starts with one complete row block, splits its columns into one
    block per destination rank, and exchanges those blocks. Transposing each
    received block produces the local rows of the global transpose. The
    operation is self-adjoint, so its backward is the same block transpose.
    """

    @staticmethod
    def _transpose_exchange(
        x: torch.Tensor,
        *,
        group,
        row_dim: int,
        column_dim: int,
    ) -> torch.Tensor:
        row_dim %= x.ndim
        column_dim %= x.ndim
        if row_dim == column_dim:
            raise ValueError("row_dim and column_dim must be different")

        group_size = dist.get_world_size(group)
        local_rows = x.shape[row_dim]
        global_columns = x.shape[column_dim]
        if global_columns != local_rows * group_size:
            raise ValueError(
                f"Expected {global_columns=} to equal {local_rows=} * "
                f"{group_size=} for a row-sharded square matrix."
            )

        # [B, P_local, P, F] -> [P_local, P, ...]
        row_column_first = x.movedim(
            (row_dim, column_dim), (0, 1)
        ).contiguous()

        # [P_local, P, ...] -> [P_local, group_size, P_local, ...]
        #     -> [group_size, P_local, P_local, ...]
        send = (
            row_column_first.unflatten(1, (group_size, local_rows))
            .movedim(1, 0)
            .contiguous()
        )

        received = torch.empty_like(send)               # [group_size, P_local, P_local, ...]
        group_rank = dist.get_rank(group)
        received[group_rank].copy_(send[group_rank])    # Keep the local row/column intersection block
        ops = []
        for peer_rank in range(group_size):
            if peer_rank == group_rank:
                continue
            global_peer = dist.get_global_rank(group, peer_rank)
            ops.extend((
                dist.P2POp(
                    dist.irecv, received[peer_rank], global_peer, group
                ),
                dist.P2POp(
                    dist.isend, send[peer_rank], global_peer, group
                ),
            ))
        if ops:
            requests = dist.batch_isend_irecv(ops)
            for request in requests:
                request.wait()

        # [group_size, P_local, P_local, ...]
        #     -> [P, P_local, ...]
        #     -> [P_local, P, ...]
        transposed = received.flatten(0, 1).transpose(0, 1).contiguous()
        return transposed.movedim(
            (0, 1), (row_dim, column_dim)
        )  # [B, P_local, P, F] for the current caller

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group,
        row_dim: int,
        column_dim: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.row_dim = row_dim
        ctx.column_dim = column_dim
        return _AllToAllTransposeForwardAllToAllTransposeBackward._transpose_exchange(
            x,
            group=group,
            row_dim=row_dim,
            column_dim=column_dim,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = (
            _AllToAllTransposeForwardAllToAllTransposeBackward._transpose_exchange(
                grad_output,
                group=ctx.group,
                row_dim=ctx.row_dim,
                column_dim=ctx.column_dim,
            )
        )
        return grad_input, None, None, None


def all_to_all_transpose(
    x: torch.Tensor,
    *,
    row_dim: int,
    column_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Return local rows of a globally transposed, row-sharded matrix."""
    return _AllToAllTransposeForwardAllToAllTransposeBackward.apply(
        x, group, row_dim, column_dim
    )
