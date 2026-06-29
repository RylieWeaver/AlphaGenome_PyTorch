# External
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# Internal
from .layers import BatchNorm, GELU_1702



def _pad_dim(x: torch.Tensor, n: int, dim=-1, value=0.0):
    if dim < 0:
        dim += x.ndim
    if n < x.size(dim):
        raise ValueError(
            f"Cannot pad dimension {dim} from size {x.size(dim)} down to {n}."
        )
    if n == x.size(dim):
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = n - x.size(dim)
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


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
    def __init__(self, in_channels, out_channels, width=5, sync_bn=True):
        super().__init__()
        self.sync_bn = sync_bn
        self.norm = BatchNorm(in_channels, sync=self.sync_bn, channels_dim=1, rms_norm=True)
        self.act = GELU_1702()
        self.width = width
        if self.width == 1:
            self.conv = nn.Linear(in_channels, out_channels)
        else:
            self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size=width)

    def forward(self, x):       # [B, C_in, S]
        x = self.norm(x)        # [B, C_in, S]
        x = self.act(x)         # [B, C_in, S]
        if self.width == 1:
            x = x.transpose(1, 2)        # [B, S, C_in]
            x = self.conv(x)             # [B, S, C_out]
            return x.transpose(1, 2)     # [B, C_out, S]
        return self.conv(x)              # [B, C_out, S]


class DNAEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels, first_conv_width=15, block_width=5, sync_bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_conv_width = first_conv_width
        self.block_width = block_width
        self.sync_bn = sync_bn
        self.conv1_d = nn.Conv1d(in_channels, out_channels, kernel_size=first_conv_width, padding=first_conv_width // 2)
        self.conv_block = ConvBlock(out_channels, out_channels, width=block_width, sync_bn=self.sync_bn)

    def forward(self, x):                   # [B, C_in, S]
        x = self.conv1_d(x)                 # [B, C_out, S]
        x = x + self.conv_block(x)          # [B, C_out, S]
        return x                            # [B, C_out, S]


class DownResBlock(nn.Module):
    """
    It's expected that in_channels < out_channels.
    """
    def __init__(self, in_channels, out_channels, width=5, sync_bn=True):
        super().__init__()
        self.out_channels = out_channels
        self.sync_bn = sync_bn
        self.conv_block1 = ConvBlock(in_channels, out_channels,  width=width, sync_bn=self.sync_bn)
        self.conv_block2 = ConvBlock(out_channels, out_channels, width=width, sync_bn=self.sync_bn)

    def forward(self, x):                               # [B, C_in, S]
        out = self.conv_block1(x)                       # [B, C_out, S]
        out = out + _pad_dim(x, out.size(1), dim=1)     # [B, C_out, S]: residual connection with padding
        out = out + self.conv_block2(out)               # [B, C_out, S]
        return out


class UpResBlock(nn.Module):
    def __init__(self, num_channels, skip_channels, width=5, init_scale=0.9, sync_bn=True):
        super().__init__()
        self.num_channels = num_channels
        self.skip_channels = skip_channels
        self.width = width
        self.init_scale = init_scale
        self.sync_bn = sync_bn
        self.conv_in = ConvBlock(num_channels, skip_channels, width=width, sync_bn=self.sync_bn)
        self.residual_scale = nn.Parameter(torch.tensor([init_scale]))
        self.pointwise_conv_unet_skip = ConvBlock(skip_channels, skip_channels, width=1, sync_bn=self.sync_bn)
        self.conv_out = ConvBlock(skip_channels, skip_channels, width=width, sync_bn=self.sync_bn)

    def forward(self, x, unet_skip_x):                                  # [B, C, S], [B, C_skip, S]
        _, C_skip, _ = unet_skip_x.shape

        x = self.conv_in(x) + x[:, :C_skip, :]                          # [B, C_skip, S]
        x = repeat(
            x, 'b c s -> b c (s r)', r=2
        ) * self.residual_scale.to(x.dtype)                             # [B, C_skip, 2*S]
        x = x + self.pointwise_conv_unet_skip(unet_skip_x)              # [B, C_skip, 2*S]
        x = x + self.conv_out(x)                                        # [B, C_skip, 2*S]
        return x                                                        # [B, C_skip, 2*S]
