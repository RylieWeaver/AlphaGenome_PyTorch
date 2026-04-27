# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
import warnings
import numpy as np

# Torch
import torch
import torch.nn as nn
from einops.layers.torch import Reduce

# AlphaGenome
from .utils import RMSBatchNorm, RMSLayerNorm



def central_mask_features(rel, feature_size, max_sequence_length, device, dtype=torch.float32):
    # Setup
    half = feature_size // 2

    # Make geometrically spaced thresholds for each feature
    widths_lin = torch.arange(half, device=device, dtype=dtype)                     # [F/2]
    if half > max_sequence_length:
        widths_geo = torch.ones(half, device=device, dtype=dtype)                  # [F/2]
    else:
        widths_geo = torch.tensor(
            np.geomspace(
                1, max_sequence_length - half + 1, half, endpoint=False,
            ),
            device=device, dtype=dtype,
        )                                                                           # [F/2]
    center_widths = widths_lin + widths_geo                                         # [F/2]   

    # Pairwise compare all absolute relative positions and thresholds
    onehot = (center_widths.unsqueeze(0) > rel.abs().unsqueeze(1)).to(dtype)        # [2L, F/2]: 0/1 if within threshold

    # Concatenate unsigned and signed features
    sign = rel.sign().unsqueeze(1).to(dtype)                                        # [2L, 1]: -1/0/1 sign of relative position
    embeddings = torch.cat([onehot, onehot * sign], dim=-1)                         # [2L, F]
    return embeddings                                                               # [2L, F]


def relative_shift(x):
    """
    Relative shift takes a tensor indexed by relative offsets (length 2S)
    and re-indexes it so each query row i exposes a contiguous S-wide slice
    corresponding to keys j=0..S-1. In other words, we start with embeddings
    for all possible relative distances, then slide a window along those
    embeddings, where each window is a row in the [S, S] pairwise location matrix.

    Example (S=3):
    Input:
        [[-3, -2, -1,  0,  1,  2],
         [-3, -2, -1,  0,  1,  2],
         [-3, -2, -1,  0,  1,  2]]
    Output:
        [[ 0,  1,  2],
         [-1,  0,  1],
         [-2, -1,  0]]
    """
    B, *extra_dims, S, num_diagonals = x.shape                          # [B, ..., S, 2S]
    x = x.reshape(B, *extra_dims, num_diagonals, S)                     # [B..., 2S, S]
    y = x[..., 1:, :].reshape(B, *extra_dims, S, num_diagonals-1)       # [B..., S, 2S-1]
    return y[..., :S]                                                   # [B..., S, S]


class SequenceToPairBlock(nn.Module):
    def __init__(self, num_channels, pair_channels, max_pair_seq_len, pair_downsample_width, pair_heads, pos_channels, dropout, sync_bn=True):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels                        # C'
        self.pair_channels = pair_channels                      # F
        self.max_pair_seq_len = max_pair_seq_len                # P
        self.pair_downsample_width = pair_downsample_width      # W_p
        self.pair_heads = pair_heads                            # H_p
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Make pos_channels even
        if pos_channels % 2 != 0:
            new_pc = pos_channels + 1
            warnings.warn(f"pos_channels={pos_channels} is odd; using {new_pc} instead.")
            pos_channels = new_pc
        self.pos_channels = pos_channels                        # C_p

        # Reduction and normalization
        self.pool = Reduce('b (n pool) d -> b n d', 'mean', pool=pair_downsample_width)
        self.norm = RMSLayerNorm(self.num_channels, channels_dim=2)

        # Non-positional attention
        self.q_linear = nn.Linear(self.num_channels, self.pair_heads * self.pair_channels, bias=False)
        self.k_linear = nn.Linear(self.num_channels, self.pair_heads * self.pair_channels, bias=False)

        # Positional attention
        self.pos_linear = nn.Linear(self.pos_channels, self.pair_heads * self.pair_channels)
        self.q_r_bias = nn.Parameter(torch.zeros(1, 1, self.pair_heads, self.pair_channels))
        self.k_r_bias = nn.Parameter(torch.zeros(1, 1, self.pair_heads, self.pair_channels))

        # Output
        self.act = nn.GELU()
        self.y_q_linear = nn.Linear(self.num_channels, self.pair_channels, bias=False)
        self.y_k_linear = nn.Linear(self.num_channels, self.pair_channels, bias=False)
        self.out = nn.Linear(self.pair_heads, self.pair_channels)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):                                                                                   # [B, S', C']
        # Setup
        B, S = x.shape[0], x.shape[1]
        P = S // self.pair_downsample_width
        H = self.pair_heads
        F = self.pair_channels
        device = x.device
        dtype=x.dtype

        # Reduction and normalization
        x = self.pool(x)                                                                                    # [B, C', P]
        x = self.norm(x)                                                                                    # [B, P, C']

        # Non-positional attention
        q = self.q_linear(x).reshape(B, P, H, F)                                                            # [B, P, H_p, F]
        k = self.k_linear(x).reshape(B, P, H, F)                                                            # [B, P, H_p, F]

        # Positional attention
        relative_positions = torch.arange(-P, P, device=device)                                             # [2P] ([-L, ..., -1, 0, 1, ..., L-1])
        pos_features = central_mask_features(
            rel=relative_positions, feature_size=self.pos_channels,                                         # [2P, C_p]
            max_sequence_length=P, device=device, dtype=dtype
        )
        pos_encoding = self.pos_linear(pos_features).reshape(2*P, H, F)                                     # [2P, H_p, F]
        rel_q_a = torch.einsum('bqhc,phc->bqph', q + self.q_r_bias, pos_encoding) / 2                       # [B, P, 2P, H_p]
        rel_q_a = relative_shift(rel_q_a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                           # [B, P, P, H_p]
        rel_k_a = torch.einsum('bkhc,phc->bkph', k + self.k_r_bias, pos_encoding) / 2                       # [B, P, 2P, H_p]
        rel_k_a = relative_shift(rel_k_a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                           # [B, P, P, H_p]
        a = torch.einsum('bqhc,bkhc->bqkh', q, k) + 0.5*(rel_q_a + rel_k_a.transpose(1, 2))                 # [B, P, P, H_p]

        # Output
        y_q = self.y_q_linear(self.act(x))                                                                  # [B, P, F]
        y_k = self.y_k_linear(self.act(x))                                                                  # [B, P, F]
        pair_activations = self.out(a) + y_q[:, :, None, :] + y_k[:, None, :, :]                            # [B, P, P, F]
        return self.dropout(pair_activations)                                                               # [B, P, P, F]


class RowAttentionBlock(nn.Module):
    """
    Row-wise attention on pair features.
    Input/Output: [B, P, P, F]
      - For each row i, attend across columns j∈[0..P-1].
    """
    def __init__(self, pair_channels, dropout, sync_bn=True):
        super().__init__()
        # Read inputs
        self.pair_channels = pair_channels      # F
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Normalization
        self.norm = RMSLayerNorm(self.pair_channels, channels_dim=3)

        # Projections
        self.q_proj = nn.Linear(self.pair_channels, self.pair_channels, bias=False)
        self.k_proj = nn.Linear(self.pair_channels, self.pair_channels, bias=False)
        self.v_proj = nn.Linear(self.pair_channels, self.pair_channels, bias=True)

        # Output
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # Setup
        dtype = x.dtype

        # Normalize inputs
        x = self.norm(x)                                                    # [B, P, P, F]

        # Reshape to treat each row independently and project
        q = self.q_proj(x)                                                  # [B, P, P, F]
        k = self.k_proj(x)                                                  # [B, P, P, F]
        v = self.v_proj(x)                                                  # [B, P, P, F]

        # Demand upcast for sensitive ops
        q, k, v = q.float(), k.float(), v.float()

        # Compute attention update (back to original dtype)
        d = q.shape[-1]
        logits = torch.einsum('bpPf,bpkf->bpPk', q, k) / (d**0.5)           # [B, P, P, P]
        a = torch.softmax(logits, dim=3)                                    # [B, P, P, P]
        x = torch.einsum('bpPk,bpkf->bpPf', a, v).to(dtype)                 # [B, P, P, F]

        # Output
        x = self.dropout(x)                                                 # [B, P, P, F]
        return x


class PairMLPBlock(nn.Module):
    def __init__(self, pair_channels, mlp_ratio, dropout=0.0, sync_bn=True):
        super().__init__()
        # Read inputs
        self.pair_channels = pair_channels      # F
        self.mlp_ratio = mlp_ratio              # M
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Modules
        self.norm = RMSLayerNorm(self.pair_channels, channels_dim=3)
        self.fc1 = nn.Linear(self.pair_channels, self.pair_channels * self.mlp_ratio)
        self.fc2 = nn.Linear(self.pair_channels * self.mlp_ratio, self.pair_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.norm(x)                    # [B, P, P, F]
        x = self.fc1(x)                     # [B, P, P, F*M]
        x = self.act(x)                     # [B, P, P, F*M]
        x = self.fc2(x)                     # [B, P, P, F]
        x = self.dropout(x)                 # [B, P, P, F]
        return x


class PairUpdateBlock(nn.Module):
    def __init__(self, num_channels, pair_downsample_width, max_pair_seq_len, pair_channels, pos_channels, pair_heads, mlp_ratio, dropout, sync_bn=True):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels                        # C'
        self.pair_downsample_width = pair_downsample_width      # W_p
        self.max_pair_seq_len = max_pair_seq_len                # P
        self.pair_channels = pair_channels                      # F
        self.pair_heads = pair_heads                            # H_p
        self.pos_channels = pos_channels                        # C_p
        self.mlp_ratio = mlp_ratio                              # M
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Modules
        self.sequence_to_pair_block = SequenceToPairBlock(
            num_channels=self.num_channels,
            pair_channels=self.pair_channels,
            max_pair_seq_len=self.max_pair_seq_len,
            pair_downsample_width=self.pair_downsample_width,
            pair_heads=self.pair_heads,
            pos_channels=self.pos_channels,
            dropout=self.dropout,
            sync_bn=self.sync_bn
        )
        self.row_attn_block = RowAttentionBlock(
            pair_channels=self.pair_channels,
            dropout=self.dropout,
            sync_bn=self.sync_bn
        )
        self.pair_mlp_block = PairMLPBlock(
            pair_channels=self.pair_channels,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            sync_bn=self.sync_bn
        )

    def forward(self, sequence_input, pair_input):                               # [B, S', C'] | [B, P, P, F]
        y = self.sequence_to_pair_block(sequence_input)                          # [B, P, P, F]
        x = y if pair_input is None else pair_input + y                          # [B, P, P, F]
        x = x + self.row_attn_block(x)                                           # [B, P, P, F]
        x = x + self.pair_mlp_block(x)                                           # [B, P, P, F]
        return x                                                                 # [B, P, P, F]
