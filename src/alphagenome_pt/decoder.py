# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch
import torch.nn as nn

# AlphaGenome
from .utils import ConvBlock



class UpResBlock(nn.Module):
    def __init__(self, num_channels, skip_channels, width=5, init_scale=0.9):
        super().__init__()
        self.num_channels = num_channels
        self.skip_channels = skip_channels
        self.width = width
        self.init_scale = init_scale
        self.block1 = ConvBlock(num_channels, skip_channels, width=width)
        self.residual_scale = nn.Parameter(torch.tensor([init_scale]))
        self.block2 = ConvBlock(skip_channels, skip_channels, width=1)
        self.block3 = ConvBlock(skip_channels, skip_channels, width=width)

    def forward(self, x, unet_skip_x):                          # [B, C, S], [B, C_skip, S]
        _, C_skip, _ = unet_skip_x.shape

        x = self.block1(x) + x[:, :C_skip, :]                   # [B, C_skip, S]
        x = x.repeat(1, 1, 2) * self.residual_scale             # [B, C_skip, 2*S]
        x = x + self.block2(unet_skip_x)                        # [B, C_skip, 2*S]
        x = x + self.block3(x)                                  # [B, C_skip, 2*S]
        return x                                                # [B, C_skip, 2*S]

class SequenceDecoder(nn.Module):
    def __init__(self,
        channel_sizes,
        block_width=5,
        init_scale=0.9
    ):
        super().__init__()
        self.stages = len(channel_sizes) - 1
        self.channel_sizes = channel_sizes[::-1]  # reverse for decoding
        self.bin_sizes = [2**i for i in range(self.stages)][::-1]  # reverse for decoding
        self.block_width = block_width
        self.init_scale = init_scale

        self.upres_blocks = nn.ModuleDict()
        for i in range(self.stages):
            self.upres_blocks[f'bin_size_{self.bin_sizes[i]}'] = UpResBlock(
                num_channels=self.channel_sizes[max(i-1, 0)],
                skip_channels=self.channel_sizes[i],
                width=block_width,
                init_scale=init_scale
            )

    def forward(self, x, intermediates):        # x: [B, C', S'] | intermediates: dict that contains x_intermediate: [B, C_i, S_i] for U-Net skip connections
        for bin_size in self.bin_sizes:
            x = self.upres_blocks[f'bin_size_{bin_size}'](x, intermediates[f'bin_size_{bin_size}']['embeddings'])       # [B, C, S]
        return x                                                                                                        # [B, C, S]
