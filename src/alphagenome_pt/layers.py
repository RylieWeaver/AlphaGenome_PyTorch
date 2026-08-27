# External
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

# Internal
from .distributed import is_dist, dist_sum
from .precision import _ACTIVE_DTYPE_POLICY



class GELU_1702(Module):
    # NOTE: AlphaGenome applies a custom scaled GELU
    # in its conv blocks and output embeddings.
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class BatchNorm(nn.Module):
    """
    BatchNorm:
    - scale, offset are learnable per-channel parameters
    - Tracks variance with an Exponential Moving Average (EMA)
      during training, then freezes it for eval.

    Standard (with mean subtraction):
    - x' = x - mean(x)
    - y = (x' / sqrt(mean_square(x') + eps)) * scale + offset
    RMS (without mean subtraction): 
    - y = (x / sqrt(mean_square(x) + eps)) * scale + offset

    This function is shape-agnostic. For example:
    - x: [B, C, S], set channels_dim=1
    - x: [B, S, C], set channels_dim=2
    - x: [B, S, S, C], set channels_dim=3

    NOTE: With sync=True and initialized torch.distributed, statistics are
    reduced across devices with gradients maintained through dist_sum.

    NOTE: 
    - Training: update the EMA stats but don't use them. 
    - Inference: use the EMA stats but don't update them.
    """
    def __init__(self, num_channels, sync=True, channels_dim=-1, rms_norm=True, eps=1e-5, decay=0.9):
        super().__init__()
        self.num_channels = num_channels
        self.sync = sync
        self.channels_dim = channels_dim
        self.rms_norm = rms_norm
        self.eps = eps
        self.decay = decay
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.offset = nn.Parameter(torch.zeros(num_channels))
        if not self.rms_norm:
            self.register_buffer("mean_EMA", torch.zeros(num_channels))
        self.register_buffer("var_EMA", torch.ones(num_channels))

    def _maybe_synced_mean(self, x, reduce_dims):
        stats_dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
        if self.sync and is_dist():
            total = x.sum(dim=reduce_dims, dtype=stats_dtype)
            count = torch.ones_like(x).sum(dim=reduce_dims, dtype=stats_dtype)
            total = dist_sum(total)
            count = dist_sum(count)
            mean = (total / count)
        else:
            mean = x.mean(dim=reduce_dims, dtype=stats_dtype)
        return mean

    def _maybe_synced_mean_square(self, x, reduce_dims):
        stats_dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
        if self.sync and is_dist():
            sum_square = torch.square(x).sum(dim=reduce_dims, dtype=stats_dtype)
            count = torch.ones_like(x).sum(dim=reduce_dims, dtype=stats_dtype)
            sum_square = dist_sum(sum_square)
            count = dist_sum(count)
            var = (sum_square / count)
        else:
            var = torch.square(x).mean(dim=reduce_dims, dtype=stats_dtype)
        return var

    def forward(self, x):
        # Make sure channels_dim is valid
        channels_dim = self.channels_dim % x.ndim
        reduce_dims = [i for i in range(x.ndim) if i != channels_dim]               # e.g. [0, 2] if channels_dim=1
        stats_shape = tuple(                                                        # shape for broadcasting (1 on all dims except channels_dim)
            self.num_channels if i == channels_dim else 1
            for i in range(x.ndim)
        )

        # (Maybe) Mean offset
        if not self.rms_norm:
            if self.training:
                mean = self._maybe_synced_mean(x, reduce_dims)                      # [C]: compute over all but the channels dim
                with torch.no_grad():
                    self.mean_EMA.mul_(self.decay).add_((1 - self.decay) * mean)    # update EMA with inplace operations
            else:
                mean = self.mean_EMA
            x = x - mean.view(stats_shape)
        
        # Variance
        if self.training:
            var = self._maybe_synced_mean_square(x, reduce_dims)                    # [C]: compute over all but the channels dim
            with torch.no_grad():
                self.var_EMA.mul_(self.decay).add_((1 - self.decay) * var)          # update EMA with inplace operations
        # During inference, use the EMA of variance but don't update it.
        else:
            var = self.var_EMA

        # # Warn low variance (can lead to instability)
        # if (var < 1e-4).any():
        #     import warnings
        #     warnings.warn(f"Low variance detected in BatchNorm: var={var.min().item():.4e}")

        # Apply
        scale = self.scale.to(x.dtype).view(stats_shape)
        offset = self.offset.to(x.dtype).view(stats_shape)
        inv = scale * torch.rsqrt(var.to(_ACTIVE_DTYPE_POLICY.get().compute_uptype) + self.eps).to(x.dtype).view(stats_shape)
        y = x * inv + offset
        return y


class LayerNorm(nn.Module):
    """
    LayerNorm:
    - scale, offset are learnable per-channel parameters

    Standard (with mean subtraction):
    - x' = x - mean(x)
    - y = (x' / sqrt(mean_square(x') + eps)) * scale + offset
    RMS (without mean subtraction): 
    - y = (x / sqrt(mean_square(x) + eps)) * scale + offset
    
    This function is shape-agnostic. For example:
    - x: [B, C, S], set channels_dim=1
    - x: [B, S, C], set channels_dim=2
    - x: [B, S, S, C], set channels_dim=3
    """
    def __init__(self, num_channels, channels_dim=-1, rms_norm=False, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.channels_dim = channels_dim
        self.rms_norm = rms_norm
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.offset = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # Make sure channels_dim is valid
        channels_dim = self.channels_dim % x.ndim
        stats_shape = tuple(                                            # shape for broadcasting (1 on all dims except channels_dim)
            self.num_channels if i == channels_dim else 1
            for i in range(x.ndim)
        )
        
        # Compute stats
        stats_dtype = _ACTIVE_DTYPE_POLICY.get().compute_uptype
        if not self.rms_norm:
            mean = x.mean(
                dim=channels_dim, dtype=stats_dtype, keepdim=True
            ).to(x.dtype)
            x = x - mean
        var = torch.square(x).mean(
            dim=channels_dim, dtype=stats_dtype, keepdim=True
        )

        # Apply normalization
        scale = self.scale.to(x.dtype).view(stats_shape)
        offset = self.offset.to(x.dtype).view(stats_shape)
        inv = scale * torch.rsqrt(var + self.eps).to(x.dtype)
        y = inv * x + offset
        return y
