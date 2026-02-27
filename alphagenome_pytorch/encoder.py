# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch.nn as nn

# AlphaGenome
from .utils import ConvBlock, _pad_dim



class DNAEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels, first_conv_width=15, block_width=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_conv_width = first_conv_width
        self.block_width = block_width
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=first_conv_width, padding=first_conv_width // 2)
        self.conv_block = ConvBlock(out_channels, out_channels, width=block_width)

    def forward(self, x):                       # [B, C_in, S]
        out = self.conv(x)                      # [B, C_out, S]
        out = out + self.conv_block(out)        # [B, C_out, S]
        return out                              # [B, C_out, S]


class DownResBlock(nn.Module):
    """
    It's expected that out_channels < in_channels.
    """
    def __init__(self, in_channels, out_channels, width=5):
        super().__init__()
        self.out_channels = out_channels
        self.conv_block1 = ConvBlock(in_channels, out_channels,  width=width)
        self.conv_block2 = ConvBlock(out_channels, out_channels, width=width)

    def forward(self, x):                               # [B, C_in, S]
        out = self.conv_block1(x)                       # [B, C_out, S]
        out = out + _pad_dim(x, out.size(1), dim=1)     # [B, C_out, S]: residual connection with padding
        out = out + self.conv_block2(out)               # [B, C_out, S]
        return out


class SequenceEncoder(nn.Module):
    """
    Progressively downsample sequence by 2 in stages. The first stage is a DNA 
    embedder, and subsequent stages are residual blocks that increase channels.
    """
    def __init__(
            self,
            channel_sizes,
            first_conv_width,
            encoder_downsample_width,
            block_width
        ):
        super().__init__()
        self.stages = len(channel_sizes) - 1
        self.channel_sizes = channel_sizes
        self.bin_sizes = [2**i for i in range(self.stages)]
        self.first_conv_width = first_conv_width
        self.encoder_downsample_width = encoder_downsample_width    # W_e
        self.block_width = block_width

        self.pool = nn.MaxPool1d(kernel_size=self.encoder_downsample_width, stride=self.encoder_downsample_width)
        self.downres_blocks = nn.ModuleDict()
        for i in range(self.stages):
            if i == 0:
                self.downres_blocks[f'bin_size_{self.bin_sizes[i]}'] = DNAEmbedder(
                    in_channels=self.channel_sizes[i],
                    out_channels=self.channel_sizes[i+1],
                    first_conv_width=self.first_conv_width,
                    block_width=self.block_width
                )
            else:
                self.downres_blocks[f'bin_size_{self.bin_sizes[i]}'] = DownResBlock(
                    in_channels=self.channel_sizes[i],
                    out_channels=self.channel_sizes[i+1],
                    width=self.block_width
                )

    def forward(self, x):                                           # [B, C_0, S]
        # Setup
        intermediates = {}                                          # store intermediate outputs for skip connections

        # Iterate through blocks
        for i in range(self.stages):
            bin_size = self.bin_sizes[i]
            x = self.downres_blocks[f'bin_size_{bin_size}'](x)      # [B, C_{i+1}, S // (W_e)^i]
            intermediates[f"bin_size_{bin_size}"] = {
                "stage": i+1,
                "channels": self.channel_sizes[i+1],
                "embeddings": x
            }
            x = self.pool(x)                                        # [B, C_{i+1}, S // (W_e)^(i+1)]

        return x, intermediates                                     # [B, C_stages, S // (W_e)^stages] | dict of [B, C_i, S // (W_e)^i] for U-Net skip connections
