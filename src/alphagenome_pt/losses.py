# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

# External
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, reduce

# Internal
from .distributed import dist_sum
from .precision import _ACTIVE_DTYPE_POLICY
from .utils import _register_pytree_dataclass


# NOTE: Dataclass fields let FSDP attach backward hooks to leaf tensors; pytree
# registration lets DDP wrap them when backward uses the returned tree.


@_register_pytree_dataclass
@dataclass(init=False, eq=False)
class LossLeaf:
    """Additive numerator and denominator for one mean loss term."""

    numerator: torch.Tensor
    denominator: torch.Tensor

    def __init__(
        self,
        numerator: torch.Tensor | float | int,
        denominator: torch.Tensor | float | int = 1.0,
    ) -> None:
        if isinstance(numerator, (float, int)):
            numerator = torch.tensor(numerator, dtype=torch.get_default_dtype())
        if not isinstance(numerator, torch.Tensor):
            raise TypeError(
                "LossLeaf numerator must be a torch.Tensor, float, or int."
            )
        denominator = torch.as_tensor(
            denominator,
            dtype=numerator.dtype,
            device=numerator.device,
        ).detach()

        for name, value in (("numerator", numerator), ("denominator", denominator)):
            if not value.is_floating_point():
                raise TypeError(f"LossLeaf {name} must be floating point.")
            if value.ndim != 0:
                raise ValueError(f"LossLeaf {name} must be a scalar tensor.")

        self.numerator = numerator
        self.denominator = denominator

    @property
    def value(self) -> torch.Tensor:
        # NOTE: Denominator is a single value, but we still use torch.where
        # to avoid GPU-to-CPU synchronization from a Python if and return
        # zero for empty losses without dividing by zero.
        has_values = self.denominator > 0
        safe_denominator = torch.where(
            has_values,
            self.denominator,
            torch.ones_like(self.denominator),
        )
        value = torch.where(
            has_values,
            self.numerator / safe_denominator,
            torch.zeros_like(self.numerator),
        )
        return value

    def scaled(self, weight: torch.Tensor | float) -> "LossLeaf":
        # How to apply loss weighting in this numerator/denominator regime
        return LossLeaf(self.numerator * weight, self.denominator)

    def add(
        self,
        other: "LossLeaf",
        *,
        detach: bool = True,
    ) -> "LossLeaf":
        """Combine statistics for two loss leaves."""
        if not isinstance(other, LossLeaf):
            raise TypeError("LossLeaf can only be added to another LossLeaf.")
        left_numerator = (
            self.numerator.detach() if detach else self.numerator
        )
        right_numerator = (
            other.numerator.detach() if detach else other.numerator
        )
        left_denominator = (
            self.denominator.detach() if detach else self.denominator
        )
        right_denominator = (
            other.denominator.detach() if detach else other.denominator
        )
        return LossLeaf(
            left_numerator + right_numerator,
            left_denominator + right_denominator,
        )

    def detach(self) -> "LossLeaf":
        return LossLeaf(
            self.numerator.detach(),
            self.denominator.detach(),
        )


MetricPath: TypeAlias = tuple[str, ...]
MetricNode: TypeAlias = LossLeaf | Mapping[str, "MetricNode"]
MetricDict: TypeAlias = dict[str, "MetricDictNode"]
MetricDictNode: TypeAlias = torch.Tensor | MetricDict


@_register_pytree_dataclass
@dataclass(init=False, eq=False)
class MetricTree:
    """
    A nested tree of model metrics, currently limited to loss leaves.

    Traversals generate tuple paths in sorted order whenever they are needed.
    It's crucial to have a canonical order for distributed reductions, so that
    every rank all-reduces the same sequence of leaves. However, it adds the
    assumption that each tree has the same set of paths. As a result, when no
    targets contribute to a leaf, its value should be a scalar zero tensor
    rather than omitting the leaf.
    """

    children: dict[str, MetricNode]

    def __init__(self, children: Mapping[str, MetricNode]):
        if not isinstance(children, Mapping):
            raise TypeError("MetricTree children must be a mapping.")
        if not children:
            raise ValueError("MetricTree cannot be empty.")
        self.children = dict(children)

    @staticmethod
    def _sorted_names(children: Mapping[str, MetricNode]) -> tuple[str, ...]:
        names = tuple(children)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Metric path names must be non-empty strings.")
        return tuple(sorted(names))

    @staticmethod
    def _walk(
        children: Mapping[str, MetricNode],
        prefix: MetricPath = (),
    ) -> Iterator[tuple[MetricPath, LossLeaf]]:
        for name in MetricTree._sorted_names(children):
            node = children[name]

            path = (*prefix, name)
            if isinstance(node, LossLeaf):
                yield path, node
            elif isinstance(node, Mapping):
                if not node:
                    raise ValueError(f"Metric branch {name!r} cannot be empty.")
                yield from MetricTree._walk(node, path)
            else:
                raise TypeError(
                    "MetricTree nodes must be LossLeaf objects or mappings."
                )

    def iter_leaves(self) -> Iterator[tuple[MetricPath, LossLeaf]]:
        """
        Yield (path, leaf) pairs in a canonical sorted order,
        which will be necessary for consistent traversal order
        when doing distributed reductions.
        """
        yield from self._walk(self.children)

    def leaf_paths(self) -> tuple[MetricPath, ...]:
        """Return all leaf paths in canonical sorted order."""
        return tuple(path for path, _ in self.iter_leaves())

    @classmethod
    def _to_dict_children(
        cls,
        children: Mapping[str, MetricNode],
    ) -> MetricDict:
        values: MetricDict = {}
        for name in cls._sorted_names(children):
            node = children[name]
            if isinstance(node, LossLeaf):
                values[name] = node.value
            elif isinstance(node, Mapping):
                if not node:
                    raise ValueError(f"Metric branch {name!r} cannot be empty.")
                values[name] = cls._to_dict_children(node)
            else:
                raise TypeError(
                    "MetricTree nodes must be LossLeaf objects or mappings."
                )
        return values

    def to_dict(self) -> MetricDict:
        """Return a nested dictionary of leaf tensors in sorted order."""
        return self._to_dict_children(self.children)

    @classmethod
    def _detach_children(
        cls,
        children: Mapping[str, MetricNode],
    ) -> dict[str, MetricNode]:
        detached: dict[str, MetricNode] = {}
        for name in cls._sorted_names(children):
            node = children[name]
            if isinstance(node, LossLeaf):
                detached[name] = node.detach()
            elif isinstance(node, Mapping):
                detached[name] = cls._detach_children(node)
            else:
                raise TypeError(
                    "MetricTree nodes must be LossLeaf objects or mappings."
                )
        return detached

    @classmethod
    def _add_children(
        cls,
        left: Mapping[str, MetricNode],
        right: Mapping[str, MetricNode],
        *,
        detach: bool,
        prefix: MetricPath = (),
    ) -> dict[str, MetricNode]:
        if left.keys() != right.keys():
            raise ValueError(
                "Metric trees must have identical paths; branches differ "
                f"at {prefix!r}."
            )

        children: dict[str, MetricNode] = {}
        for name in cls._sorted_names(left):
            left_node = left[name]
            right_node = right[name]
            path = (*prefix, name)
            if isinstance(left_node, LossLeaf) and isinstance(
                right_node, LossLeaf
            ):
                children[name] = left_node.add(right_node, detach=detach)
            elif isinstance(left_node, Mapping) and isinstance(
                right_node, Mapping
            ):
                children[name] = cls._add_children(
                    left_node,
                    right_node,
                    detach=detach,
                    prefix=path,
                )
            else:
                raise ValueError(
                    f"Metric tree shape conflict at {path!r}: one side is "
                    "a leaf and the other is a branch."
                )
        return children

    def detach(self) -> "MetricTree":
        return MetricTree(self._detach_children(self.children))

    def add(
        self,
        other: "MetricTree",
        *,
        detach: bool = True,
    ) -> "MetricTree":
        if not isinstance(other, MetricTree):
            raise TypeError("MetricTree can only be added to another MetricTree.")

        return MetricTree(
            self._add_children(
                self.children,
                other.children,
                detach=detach,
            )
        )

    def distributed_reduce(self, group=None) -> "MetricTree":
        leaves = dict(self.iter_leaves())
        numerators = torch.stack([leaf.numerator for leaf in leaves.values()])
        denominators = torch.stack([leaf.denominator for leaf in leaves.values()])
        numerators = dist_sum(numerators, group=group)
        denominators = dist_sum(denominators, group=group)

        replacements = {
            path: LossLeaf(numerator, denominator)
            # NOTE: tensor.unbind() splits among dim 0
            for path, numerator, denominator in zip(
                leaves.keys(),
                numerators.unbind(),
                denominators.unbind(),
                strict=True,
            )
        }

        # NOTE: This code may look complicated but basically just 
        # takes flattened paths to a tree structure for the dict
        children = {}
        for path, leaf in replacements.items():
            branch = children
            for name in path[:-1]:
                branch = branch.setdefault(name, {})
            branch[path[-1]] = leaf
        
        return MetricTree(children)

    def total_loss(self, *prefix: str) -> torch.Tensor:
        """Return the summed loss, possibly within prefix."""
        node: MetricNode = self.children
        for name in prefix:
            if not isinstance(node, Mapping):
                raise KeyError(
                    f"Metric path continues beyond a leaf: {prefix!r}."
                )
            if name not in node:
                raise KeyError(f"No metrics at path: {prefix!r}.")
            node = node[name]

        if isinstance(node, LossLeaf):
            return node.value
        values = [leaf.value for _, leaf in self._walk(node, prefix)]
        if not values:
            raise ValueError(f"Metric branch at {prefix!r} is empty.")
        return sum(values)

    def head_loss_totals(self) -> dict[str, torch.Tensor]:
        """Return the loss total for every top-level branch."""
        return {
            head_name: self.total_loss(head_name)
            for head_name in self._sorted_names(self.children)
        }


def _safe_masked_mean(
    x: torch.Tensor,                        # [*]
    mask: torch.Tensor | None = None,       # [#*]
) -> LossLeaf:
    """Return additive statistics for a safe masked mean."""
    if mask is None:
        masked = x
        mask = torch.ones_like(x, dtype=x.dtype)
    else:
        mask = mask.expand_as(x)
        mask = mask.to(x.dtype)
        masked = x * mask

    dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
    total = torch.sum(masked, dtype=dtype)
    count = torch.sum(mask, dtype=dtype)
    return LossLeaf(total, count)


def poisson_loss(
    *,
    y_true: torch.Tensor,                   # [*]
    y_pred: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
) -> LossLeaf:
    """Poisson loss with fixed dtype and shift to have min_loss = 0."""
    compute_uptype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
    y_true = torch.abs(y_true).to(compute_uptype)
    y_pred = y_pred.to(compute_uptype)
    y_pred_logits = torch.log(y_pred + 1e-7)
    # Substract the minimum value such that loss is zero at optimal prediction.
    min_value = y_true - y_true * torch.log(y_true + 1e-7)
    loss = (y_pred - y_true * y_pred_logits) - min_value
    return _safe_masked_mean(loss, mask)


def multinomial_loss(
    *,
    y_true: torch.Tensor,                   # [..., S, C]
    y_pred: torch.Tensor,                   # [..., S, C]
    mask: torch.Tensor,                     # [..., #S, C]
    multinomial_resolution: int | None = None,
    positional_weight: float,
    min_zero: bool = True,
    eps: float = 1e-7,
    sequence_group=None,
) -> dict[str, torch.Tensor | LossLeaf]:
    """Return count and positional losses over multinomial windows.

    Under sequence parallelism, ``multinomial_resolution`` must divide the
    local sequence length, or be a multiple of it that divides the global
    sequence length.
    """
    assert y_true.shape == y_pred.shape, (
        "Shapes of y_true and y_pred must be equal."
    )

    local_sequence_length = y_pred.shape[-2]
    sp_size = (
        dist.get_world_size(sequence_group)
        if sequence_group is not None
        else 1
    )
    global_sequence_length = local_sequence_length * sp_size

    # NOTE: This default keeps SP and non-SP behavior equivalent.
    if multinomial_resolution is None:
        multinomial_resolution = global_sequence_length

    if multinomial_resolution <= 0:
        raise ValueError(
            "multinomial_resolution must be positive, got "
            f"{multinomial_resolution}."
        )
    elif (
        multinomial_resolution <= local_sequence_length
        and local_sequence_length % multinomial_resolution != 0
    ):
        raise ValueError(
            f"{local_sequence_length=} must be divisible by "
            f"{multinomial_resolution=}."
        )
    elif multinomial_resolution > local_sequence_length and (
        sequence_group is None
        or multinomial_resolution % local_sequence_length != 0
        or global_sequence_length % multinomial_resolution != 0
    ):
        raise ValueError(
            "A multinomial_resolution larger than local_sequence_length "
            "requires sequence parallelism and must be a multiple of "
            "local_sequence_length that divides global_sequence_length."
        )

    multi_rank_windows = multinomial_resolution > local_sequence_length

    # Setup
    dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
    y_true = y_true.to(dtype)
    y_pred = y_pred.to(dtype)
    mask = mask.to(dtype)

    # Remove masked bins before computing count totals and probabilities
    y_true = torch.clamp(y_true, min=0) * mask
    y_pred = y_pred * mask

    # A multi-rank window uses each whole local shard as one contribution
    local_window_size = min(
        multinomial_resolution, local_sequence_length
    )
    y_pred = rearrange(
        y_pred, "... (n s) c -> ... n s c",
        s=local_window_size,
    )
    y_true = rearrange(
        y_true, "... (n s) c -> ... n s c",
        s=local_window_size,
    )
    total_pred = reduce(y_pred, "... n s c -> ... n 1 c", "sum")        # [..., S_sub, 1, C]
    total_true = reduce(y_true, "... n s c -> ... n 1 c", "sum")        # [..., S_sub, 1, C]

    if multi_rank_windows:
        K = multinomial_resolution // local_sequence_length             # SP ranks per window
        total_pred = dist_sum(
            total_pred,
            group=sequence_group,
            ranks_per_subgroup=K,
        )                                                               # [B, 1, 1, C]
        total_true = dist_sum(
            total_true,
            group=sequence_group,
            ranks_per_subgroup=K,
        )                                                               # [B, 1, 1, C]
    mask = mask[..., None, :]                                           # [..., #S, C] -> [..., #S, 1, C]

    # NOTE: For multi-rank windows, the small count loss is repeated on K
    # ranks. Divide both statistics by K so each window is counted once,
    # including when different DP sequence lengths give different values of K.
    loss_total_count = poisson_loss(
        y_pred=total_pred,
        y_true=total_true,
        mask=mask,
    )
    if multi_rank_windows:
        loss_total_count = LossLeaf(
            loss_total_count.numerator / K,
            loss_total_count.denominator / K,
        )
    # Poisson loss is O(n) with respect to resolution, so normalize it to
    # remain invariant to resolution.
    loss_total_count = loss_total_count.scaled(
        1.0 / multinomial_resolution
    )

    # Positional loss
    prob_predictions = y_pred / (total_pred + eps)              # [..., N_local, R_local, C]
    loss_pos = -y_true * torch.log(prob_predictions + eps)      # [..., N_local, R_local, C]

    # NOTE: Positional loss has a minimum value that we can account for.
    prob_targets = y_true / (total_true + eps)                  # [..., N_local, R_local, C]
    min_value = -y_true * torch.log(prob_targets + eps)         # [..., N_local, R_local, C]
    zero_loss_pos = loss_pos - min_value                        # [..., N_local, R_local, C]

    loss_pos = _safe_masked_mean(loss_pos, mask)               # LossLeaf
    zero_loss_pos = _safe_masked_mean(
        zero_loss_pos, mask
    )                                                           # LossLeaf
    loss = zero_loss_pos if min_zero else loss_pos

    return {
        # Composite convenience value for serial callers. Distributed model
        # training reduces the independently normalized component leaves.
        "loss": (
            loss_total_count.value + positional_weight * loss.value
        ),
        "loss_total": loss_total_count,
        "loss_positional": loss_pos,
        "zero_loss_positional": zero_loss_pos,
    }


def mse(
    y_pred: torch.Tensor,                   # [*]
    y_true: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [*]
) -> LossLeaf:
    """Mean squared error."""
    return _safe_masked_mean(torch.square(y_pred - y_true), mask)


def cross_entropy_loss_from_logits(
    *,
    y_pred_logits: torch.Tensor,            # [*]
    y_true: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
    axis: int,
) -> LossLeaf:
    """Cross-entropy loss from one-hot targets."""
    log_softmax_preds = F.log_softmax(
        y_pred_logits.to(_ACTIVE_DTYPE_POLICY.get().compute_uptype), dim=axis
    )
    loss = -torch.sum(
        y_true.to(_ACTIVE_DTYPE_POLICY.get().compute_uptype)
        * log_softmax_preds,
        dim=axis,
    )
    if mask is not None:
        mask = torch.any(mask, dim=axis)
    return _safe_masked_mean(loss, mask)


def sparse_cross_entropy_loss_from_logits(
    *,
    y_pred_logits: torch.Tensor,            # [..., C]
    y_true: torch.Tensor,                   # [...]
    ignore_index: int = -100,
) -> LossLeaf:
    """Cross-entropy loss from integer labels."""
    labels = y_true.reshape(-1)
    per_item_loss = F.cross_entropy(
        y_pred_logits.to(
            _ACTIVE_DTYPE_POLICY.get().compute_uptype
        ).reshape(-1, y_pred_logits.shape[-1]),
        labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    return _safe_masked_mean(
        per_item_loss, labels != ignore_index
    )


def binary_crossentropy_from_logits(
    *,
    y_true: torch.Tensor,                   # [*]
    y_pred: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
) -> LossLeaf:
    """Binary cross-entropy loss from sigmoid logits."""
    loss = (
        torch.max(y_pred, torch.zeros_like(y_pred))
        - y_pred * y_true
        + torch.log1p(torch.exp(-torch.abs(y_pred)))
    )
    return _safe_masked_mean(loss, mask)


def cross_entropy_loss(
    *,
    y_true: torch.Tensor,                   # [*]
    y_pred: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
    axis: int,
    eps: float = 1e-7,
    axis_group=None,
) -> LossLeaf:
    """Cross entropy loss on counts, with an optionally sharded axis.

    ``axis_group`` combines sufficient statistics when ``axis`` is sharded.
    """
    if mask is None:
        mask = torch.ones_like(y_true, dtype=torch.bool)
    else:
        mask = mask.expand_as(y_true).to(torch.bool)

    # Setup
    dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
    y_true = y_true.to(dtype) + eps
    y_pred = y_pred.to(dtype) + eps
    axis_mask = mask.any(dim=axis, keepdim=True)
    if axis_group is not None:
        axis_mask = dist_sum(
            axis_mask.to(dtype), group=axis_group
        ).bool()
    mask = torch.where(axis_mask, mask, True)

    total_true = y_true.masked_fill(~mask, 0).sum(dim=axis, keepdim=True)
    total_pred = y_pred.masked_fill(~mask, 0).sum(dim=axis, keepdim=True)
    if axis_group is not None:
        total_true = dist_sum(total_true, group=axis_group)
        total_pred = dist_sum(total_pred, group=axis_group)

    p_true = y_true / total_true
    p_pred = y_pred / total_pred
    log_loss = (-p_true * torch.log(p_pred)).masked_fill(~mask, 0).sum(dim=axis)
    if axis_group is not None:
        log_loss = dist_sum(log_loss, group=axis_group)
    return _safe_masked_mean(
        log_loss,
        axis_mask.squeeze(dim=axis),
    )
