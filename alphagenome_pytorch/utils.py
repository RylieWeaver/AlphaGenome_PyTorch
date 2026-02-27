# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

# General

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# AlphaGenome



def _pad_dim(x: torch.Tensor, n: int, dim=-1, value=0.0):
    """
    Pad tensor 'x' along 'dim' with 'value' to have the right size 'n'.
    No-op if current size >= n.

    Usage explanation:
    - Convolutions usually expect a 'channels first' shape for input tensors,
      (e.g. [B, C, S]), while other modules usually expect 'channels last' shape
      (e.g. [B, S, C]). Therefore, in order to pad the number of channels (e.g. for
      skip connections), it's helpful to have a flexible padding function that can 
      pad along any dimension.

    Example usage:
    - x: [B, C, S], n: 1024, dim: 1 --> pads C to 1024
    - x: [B, S, C], n: 1024, dim: -1 --> pads C to 1024
    """
    # Calculate the amount of padding needed
    ndim = x.dim()
    dim = dim % ndim
    size = x.size(dim)
    diff = n - size

    # Return immediately if no padding is needed
    if diff <= 0:
        return x

    # Move target dim to last
    perm = list(range(ndim))                        # evaluates to [0, 1, 2, ..., dim, ..., ndim-1]
    perm[dim], perm[-1] = perm[-1], perm[dim]       # evaluates to [0, 1, 2, ..., ndim-1, ..., dim]
    x = x.permute(perm)                             # swap the target dim and last dim

    # Do the padding
    x = F.pad(x, (0, diff), mode="constant", value=value)

    # Swap target dim and last dim back
    x = x.permute(perm)

    return x


class EMA_RMSBatchNorm(nn.Module):
    """
    RMS 'BatchNorm' (no mean subtraction):
      y = (x / sqrt(mean_square(x) + eps)) * gamma + beta
    - gamma, beta are learnable per-channel scale and shift.
    - Tracks variance with an Exponential Moving Average (EMA)
      during training, then freezes it for eval.
    
    This function is shape-agnostic. For example:
    - x: [B, C, S], set channels_dim=1
    - x: [B, S, C], set channels_dim=2
    - x: [B, S, S, C], set channels_dim=3

    NOTE: Future versions may add an argument for distributed 
    reduction across devices while maintaining grads.
    """
    def __init__(self, num_channels, channels_dim=2, eps=1e-6, decay=0.9, sync=True):
        super().__init__()
        self.num_channels = num_channels
        self.channels_dim = channels_dim
        self.eps = eps
        self.decay = decay
        self.gamma = nn.Parameter(torch.zeros(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.sync = sync
        self.register_buffer("var_EMA", torch.ones(num_channels))

    def forward(self, x):
        # Make sure channels_dim is valid
        channels_dim = self.channels_dim % x.ndim
        
        # During training, update the EMA of variance but don't use it.
        if self.training:
            # reduce all but the channels dim
            reduce_dims = [i for i in range(x.ndim) if i != channels_dim]       # e.g. [0, 2] if channels_dim=1
            var = torch.square(x).mean(dim=reduce_dims)                         # [C]: compute variance over all but the channels dim
            with torch.no_grad():
                self.var_EMA.mul_(self.decay).add_((1 - self.decay) * var)      # update EMA with inplace operations
        # During inference, use the EMA of variance but don't update it.
        else:
            var = self.var_EMA

        # Apply normalization
        rms = torch.sqrt(var + self.eps)
        stats_shape = (1,) * (channels_dim) + (-1,) + (1,) * (x.ndim - channels_dim - 1)  # shape for broadcasting (1 on all dims except channels_dim)
        y = (x / rms.view(stats_shape)) * (1 + self.gamma.view(stats_shape)) + self.beta.view(stats_shape)
        return y


class StandardizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialization
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):                               # x: [B, C_in, S]
        # Setup
        fan_in = self.kernel_size * self.in_channels
        w = self.weight

        # Normalize weights
        w = w - w.mean(dim=(1, 2), keepdim=True)        # [C_out, C_in, K]
        var_w = w.var(dim=(1, 2), keepdim=True)         # [C_out, 1, 1]
        w_scale = self.scale * torch.rsqrt(torch.clamp(fan_in * var_w, min=1e-4))
        w_standardized = w * w_scale                    # [C_out, C_in, K]

        # Apply conv1d with normalized weights
        out = F.conv1d(
            x,
            w_standardized,
            bias=self.bias,
            stride=1,
            padding=self.kernel_size // 2,
        )                                               # [B, C_in, S] --> [B, C_out, S]
        return out                                      # [B, C_out, S]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, width=5):
        super().__init__()
        self.norm = EMA_RMSBatchNorm(in_channels, channels_dim=1)
        self.act = nn.GELU()
        if width == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=width, padding=width // 2)  # equivalent to nn.Linear() but expects channels first shape
        else:
            self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size=width)

    def forward(self, x):       # [B, C_in, S]
        x = self.norm(x)        # [B, C_in, S]
        x = self.act(x)         # [B, C_in, S]
        return self.conv(x)     # [B, C_out, S]
