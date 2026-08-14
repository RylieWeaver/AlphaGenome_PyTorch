# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

# External
from collections.abc import Iterator, Mapping
from typing import TypeAlias
import torch
import torch.nn.functional as F
from einops import rearrange, reduce



class LossLeaf:
    """
    Later we'll add numerator and denominator to this leaf so that we can
    calculate a true global mean across distributed processes. For now, it's
    a simple wrapper around a scalar tensor.
    """
    def __init__(self, value: torch.Tensor | float):
        if isinstance(value, float):
            value = torch.tensor(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError("LossLeaf value must be a torch.Tensor or float.")
        if not value.is_floating_point():
            raise TypeError("LossLeaf value must be floating point.")
        if value.ndim != 0:
            raise ValueError("LossLeaf value must be a scalar tensor.")
        self._value = value

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def add(
        self,
        other: "LossLeaf",
        *,
        detach: bool = True,
    ) -> "LossLeaf":
        """Return a leaf containing the sum of two leaf values."""
        if not isinstance(other, LossLeaf):
            raise TypeError("LossLeaf can only be added to another LossLeaf.")
        left = self.value.detach() if detach else self.value
        right = other.value.detach() if detach else other.value
        return LossLeaf(left + right)

    def detach(self) -> "LossLeaf":
        return LossLeaf(self.value.detach())


MetricPath: TypeAlias = tuple[str, ...]
MetricNode: TypeAlias = LossLeaf | Mapping[str, "MetricNode"]
MetricDict: TypeAlias = dict[str, "MetricDictNode"]
MetricDictNode: TypeAlias = torch.Tensor | MetricDict


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
    ) -> torch.Tensor:
    """Safe torch.mean that handles completely masked arrays."""
    if mask is None:
        masked = x
        mask = torch.ones_like(x, dtype=x.dtype)
    else:
        mask = mask.expand_as(x)
        mask = mask.to(x.dtype)
        masked = x * mask

    return torch.sum(masked, dtype=torch.float32) / torch.clamp(torch.sum(mask, dtype=torch.float32), min=1.0)


def poisson_loss(
    *,
    y_true: torch.Tensor,                   # [*]
    y_pred: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
) -> torch.Tensor:
    """Poisson loss with fixed dtype and shift to have min_loss = 0."""
    y_true = torch.abs(y_true).to(torch.float32)
    y_pred = y_pred.to(torch.float32)
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
    multinomial_resolution: int,
    positional_weight: float,
    min_zero: bool = True,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Returns sum of multinomial losses and Poison loss on total count."""
    assert y_true.shape == y_pred.shape, "Shapes of y_true, y_pred and mask must be equal."
    if y_pred.shape[-2] % multinomial_resolution != 0:
        raise ValueError(
            f'{y_pred.shape[-2]=} must be divisible by {multinomial_resolution=}.'
        )

    # Setup
    *extra_dims, S, C = y_pred.shape
    S_sub = multinomial_resolution
    dtype = torch.float32
    y_true = y_true.to(dtype)
    y_pred = y_pred.to(dtype)
    mask = mask.to(dtype)

    # Remove the masked out bins from the totals sum
    y_true = torch.clamp(y_true, min=0) * mask                                  # [..., S, C]
    y_pred = y_pred * mask                                                      # [..., S, C]

    # Split sequence into n sub-sequences of size multinomial_resolution
    y_pred = rearrange(y_pred, '... (n s) c -> ... n s c', s=S_sub)     # [..., S_sub, R, C]
    y_true = rearrange(y_true, '... (n s) c -> ... n s c', s=S_sub)     # [..., S_sub, R, C]

    # Pooled pred/true counts
    total_pred = reduce(y_pred, '... n s c -> ... n 1 c', 'sum')        # [..., S_sub, 1, C]
    total_true = reduce(y_true, '... n s c -> ... n 1 c', 'sum')        # [..., S_sub, 1, C]
    mask = mask[..., None, :]  # broadcast over segments

    # Poisson loss
    loss_total_count = poisson_loss(
        y_pred=total_pred,
        y_true=total_true,
        mask=mask,
    )
    ## NOTE: Poisson loss is O(n) wrt resolution so
    ## we normalize to be invariant to resolution
    loss_total_count /= multinomial_resolution

    # Positional loss
    prob_predictions = y_pred / (total_pred + eps)                          # [..., N, R, C]
    loss_pos = -y_true * torch.log(prob_predictions + eps)                  # [..., N, R, C]
    # NOTE: positional loss has a min value that we can account for
    prob_targets = y_true / (total_true + eps)                              # [..., N, 1, C]
    min_value = -y_true * torch.log(prob_targets + eps)                     # [..., N, R, C]
    zero_loss_pos = loss_pos - min_value                                    # [..., N, R, C]

    loss_pos = _safe_masked_mean(loss_pos, mask)                  # [1]
    zero_loss_pos = _safe_masked_mean(zero_loss_pos, mask)        # [1]
    loss = zero_loss_pos if min_zero else loss_pos
    
    return {
        'loss': loss_total_count + positional_weight * loss,
        'loss_total': loss_total_count,
        'loss_positional': loss_pos,
        'zero_loss_positional': zero_loss_pos,
    }


def mse(
    y_pred: torch.Tensor,                   # [*]
    y_true: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [*]
) -> torch.Tensor:
    """Mean squared error."""
    return _safe_masked_mean(torch.square(y_pred - y_true), mask)


def cross_entropy_loss_from_logits(
    *,
    y_pred_logits: torch.Tensor,            # [*]
    y_true: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
    axis: int,
) -> torch.Tensor:
    """Cross-entropy loss from logits."""
    log_softmax_preds = F.log_softmax(
        y_pred_logits.to(torch.float32), dim=axis
    )
    loss = -torch.sum(y_true.to(torch.float32) * log_softmax_preds, dim=axis)
    if mask is not None:
        mask = torch.any(mask, dim=axis)
    return _safe_masked_mean(loss, mask)


def binary_crossentropy_from_logits(
    *,
    y_true: torch.Tensor,                   # [*]
    y_pred: torch.Tensor,                   # [*]
    mask: torch.Tensor | None = None,       # [#*]
) -> torch.Tensor:
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
) -> torch.Tensor:
    """Cross entropy loss on counts."""
    if mask is None:
        mask = torch.ones_like(y_true, dtype=torch.bool)
    else:
        mask = mask.expand_as(y_true).to(torch.bool)

    y_true = torch.where(mask, y_true.to(torch.float32), torch.zeros_like(y_true, dtype=torch.float32))
    p_true = y_true / torch.clamp(torch.sum(y_true, dim=axis, keepdim=True), min=eps)

    log_normalizer = torch.log((torch.where(mask, y_pred.to(torch.float32), torch.zeros_like(y_pred, dtype=torch.float32)) + eps).sum(dim=axis))
    log_likelihood = torch.sum(p_true * torch.log(y_pred + eps), dim=axis)
    log_loss = log_normalizer - log_likelihood
    return _safe_masked_mean(log_loss, mask.any(dim=axis))
