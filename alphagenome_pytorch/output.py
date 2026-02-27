# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch.nn as nn

# AlphaGenome
from .utils import EMA_RMSBatchNorm



class OutputEmbedder(nn.Module):
    def __init__(self, num_channels, num_organisms, mlp_ratio, skip_channels=None):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels
        self.mlp_ratio = mlp_ratio
        self.up_channels = num_channels * mlp_ratio
        self.skip_channels = skip_channels
        self.num_organisms = num_organisms

        # Modules
        self.fc1 = nn.Linear(self.num_channels, self.up_channels)
        self.fc_skip = nn.Linear(self.skip_channels, self.up_channels, bias=False) if self.skip_channels is not None else None
        self.org = nn.Embedding(self.num_organisms, self.up_channels)
        self.norm = EMA_RMSBatchNorm(self.up_channels, channels_dim=2)
        self.activation = nn.GELU()


    def forward(self, x, organism_index, skip_x=None):          # [B, S, C], [B, S', C'], int
        # Project input
        x = self.fc1(x)                                         # [B, S, 2C]

        # Add skip connection
        if skip_x is not None:
            skip_x = self.fc_skip(skip_x)                       # [B, S', C]
            repeats = x.shape[1] // skip_x.shape[1]     
            x = x + skip_x.repeat(1, repeats, 1)                # [B, S, C]

        # Norm, add organism embedding, and output
        x = self.norm(x)                                        # [B, S, C]
        if self.num_organisms >= 1:
            org_emb = self.org(organism_index).unsqueeze(1)     # [B, 1, C]
            x = x + org_emb                                     # [B, S, 2C]
        x = self.activation(x)                                  # [B, S, 2C]
        return x                                                # [B, S, 2C]
    

class OutputPairEmbedder(nn.Module):
    def __init__(
        self,
        pair_channels: int,
        num_organisms: int,
    ):
        super().__init__()
        self.num_organisms = num_organisms
        self.norm = EMA_RMSBatchNorm(pair_channels, channels_dim=3)
        self.embed = nn.Embedding(num_organisms, pair_channels)
        self.act = nn.GELU()

    def forward(self, x, organism_index):               # x: [B, P, P, F] | organism_index: [B]
        # Symmetrize
        x = (x + x.permute(0, 2, 1, 3)) / 2.0

        # Normalize
        x = self.norm(x)                                            # [B, P, P, F]

        # Get organism embedding
        if self.num_organisms >= 1:
            org_emb = self.embed(organism_index)
            org_emb = org_emb.unsqueeze(1).unsqueeze(1)             # [B, 1, 1, F]
            x = x + org_emb                                         # [B, P, P, F]

        # Apply activation
        x = self.act(x)                                             # [B, P, P, F]
        return x


class TokenPredBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Read inputs
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Modules
        self.block = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.LayerNorm(in_channels) if in_channels > 1 else nn.Identity(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x):           # [B, S, C_in]
        return self.block(x)        # [B, S, C_out_task]
