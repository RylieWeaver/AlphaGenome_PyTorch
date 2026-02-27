# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch
import torch.nn.functional as F
from einops import rearrange

# AlphaGenome



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
    mask: torch.Tensor,                     # [..., 1, C]
    multinomial_resolution: int,
    positional_weight: float,
) -> dict[str, torch.Tensor]:
    """Returns sum of multinomial losses and Poison loss on total count."""
    assert y_true.shape == y_pred.shape, "Shapes of y_true, y_pred and mask must be equal."
    if y_pred.shape[-2] % multinomial_resolution != 0:
        raise ValueError(
            f'{y_pred.shape[-2]=} must be divisible by {multinomial_resolution=}.'
        )

    *extra_dims, S, C = y_pred.shape
    num_segments = S // multinomial_resolution                                  # N = S // R

    # Remove the masked out bins from the totals sum
    y_true = torch.clamp(y_true, min=0) * mask                                  # [..., S, C]
    y_pred = torch.clamp(y_pred, min=0) * mask                                  # [..., S, C]
    mask = mask.to(y_pred.dtype)                                                # [..., S, C]
    y_pred = y_pred * mask                                                      # [..., S, C]

    # Split sequence into n sub-sequences of size multinomial_resolution
    y_true = rearrange(y_true, '... (n s) c -> ... n s c', n=num_segments)      # [..., N, R, C]
    y_pred = rearrange(y_pred, '... (n s) c -> ... n s c', n=num_segments)      # [..., N, R, C]


    total_pred = torch.sum(y_pred, dim=-2, keepdim=True)                        # [..., N, 1, C]
    total_true = torch.sum(y_true, dim=-2, keepdim=True)                        # [..., N, 1, C]
    mask = mask[..., None, :]  # broadcast over segments

    loss_total_count = poisson_loss(
        y_true=total_true,
        y_pred=total_pred,
        mask=mask,
    )                                                                           # [1]
    # Poisson loss is O(n) wrt multinomial resolution so we
    # normalize to be invariant to multinomial_resolution
    loss_total_count /= multinomial_resolution                                  # [1]

    eps = 1e-7
    prob_predictions = y_pred.to(torch.float32) / (total_pred + eps)            # [..., N, R, C]
    prob_targets = y_true.to(torch.float32) / (total_true + eps)                # [..., N, 1, C]
    loss_positional = -y_true * torch.log(prob_predictions + eps)               # [..., N, R, C]
    loss_positional = _safe_masked_mean(loss_positional, mask=mask)             # [1]
    
    # NOTE: the above implementation has a loss floor above zero.
    # Adding a shifted version of the loss to the predictions dict.
    min_value = -y_true * torch.log(prob_targets + eps)                         # [..., N, 1, C]
    zero_loss_positional = loss_positional - min_value
    zero_loss_positional = _safe_masked_mean(zero_loss_positional, mask=mask)   # [1]
    
    return {
        'loss': loss_total_count + positional_weight * loss_positional,
        'loss_total': loss_total_count,
        'loss_positional': loss_positional,
        'loss_positional_zero_floor': zero_loss_positional,
        'max_sum_preds': torch.max(total_pred),
        'max_preds': torch.max(y_pred),
        'max_targets': torch.max(y_true.to(torch.float32)),
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
