# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch
import torch.nn as nn

# AlphaGenome
from .pair_update import PairUpdateBlock
from .utils import EMA_RMSBatchNorm



class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, num_channels, max_position):
        """
        AlphaGenome uses a custom frequency schedule for Rotary Positional Encodings (RoPE).

        Apply Rotary Positional Embeddings (RoPE) to tensor 'x'.
        - x: [B, S, H, C] where C is even.
        - positions: [S] or None. If None, uses arange(S).
        - max_position: maximum position for frequency calculation.

        Notes:
        - x_rotated:
        - Rotates the x vector by 90 degrees in each 2D subspace by swapping evens/odds and negating the odds
        - Interleaves with a stack/reshape
        """
        super().__init__()
        self.num_channels = num_channels  # C
        self.max_position = max_position  # S
        assert num_channels % 2 == 0, "RoPE dimension must be even."

    def geomspace(self, start, end, steps, device=None, dtype=None):
        start = torch.as_tensor(start, device=device, dtype=dtype)
        end = torch.as_tensor(end, device=device, dtype=dtype)
        # log-space then exp
        return torch.exp(torch.linspace(torch.log(start), torch.log(end), steps, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        # Read and check inputs
        B, S, H, C = x.shape
        device, dtype = x.device, x.dtype
        assert C == self.num_channels

        # Default positions
        if positions is None:
            positions = torch.arange(S, device=device, dtype=dtype).view(1, S)                  # [1, S]
        else:
            if positions.ndim == 1:
                positions = positions.view(1, -1)
            positions = positions.to(device=device, dtype=dtype)
        
        num_freq = C // 2

        lin_space = torch.arange(num_freq, device=device, dtype=dtype)                          # [F]
        geom_space = self.geomspace(
            1.0,
            max(self.max_position - num_freq + 1.0, 1.0),
            num_freq,
            device=device,
            dtype=dtype
        )                                                                                       # [F]
        inv_freq = 1.0 / (lin_space + geom_space)                                               # [F]

        # Compute angles
        theta = torch.einsum('...s,f->...sf', positions, inv_freq)                              # [1, S, F]
        theta = theta.repeat_interleave(2, dim=-1).unsqueeze(-2)                                # [1, S, 1, C]

        # Apply RoPE
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)             # [B, S, H, C]
        x = x * torch.cos(theta) + x_rotated * torch.sin(theta)                                 # [B, S, H, C]
        return x


class MHA(nn.Module):
    def __init__(
        self,
        num_channels: int,
        transformer_seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        dropout: float,
    ):
        """
        Grouped-Query Attention (GQA):
        - Maps multiple query heads to a single shared key/value head.
        
        Since this implementation is replicating the architectural description provided in the AlphaGenome paper, it:
        - Does not use bias in Q/K/V projections.
        - Uses LayerNorm after linear projections for Q, K, and V.
        - Uses a Tanh function to soft-clip attention logits in the range [-5, 5].
        - Does not accept an attn_mask, nor a "causal" flag.
        """
        super().__init__()
        # Read and check inputs
        self.num_channels = num_channels                    # C'
        self.num_q_heads = num_q_heads                      # H_q
        self.num_kv_heads = num_kv_heads                    # H_kv
        self.head_repeats = num_q_heads // num_kv_heads     # R_h = H_q / H_kv
        self.qk_head_dim = qk_head_dim                      # C_qk
        self.v_head_dim = v_head_dim                        # C_v
        self.transformer_seq_len = transformer_seq_len      # S'
        self.logit_clip_value = 5.0
        self.dropout = dropout
        assert self.num_q_heads % self.num_kv_heads == 0

        # Initial normalization
        self.bn1 = EMA_RMSBatchNorm(self.num_channels)

        # Q/K/V modules
        self.q_proj = nn.Linear(self.num_channels, self.num_q_heads * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(self.num_channels, self.num_kv_heads * self.qk_head_dim, bias=False)
        self.v_proj = nn.Linear(self.num_channels, self.num_kv_heads * self.v_head_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.qk_head_dim)
        self.k_norm = nn.LayerNorm(self.qk_head_dim)
        self.v_norm = nn.LayerNorm(self.v_head_dim)

        # Positional embeddings
        self.rope = RotaryPositionalEmbeddings(num_channels=self.qk_head_dim, max_position=self.transformer_seq_len)

        # Output modules
        self.bn2 = EMA_RMSBatchNorm(self.num_channels, channels_dim=2)
        self.out = nn.Sequential(nn.Linear(self.num_q_heads * self.v_head_dim, self.num_channels, bias=True), self.bn2, nn.Dropout(self.dropout))

    def forward(self, x, attn_bias=None):                                               # x: [B, S', C'] | attn_bias: [B, H_q, S', S']
        # Setup
        B, S, _ = x.shape

        # Initial normalization
        x = self.bn1(x)                                                                  # [B, S', C']

        # Multihead Q, K, V followed by LN
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.qk_head_dim)               # [B, S', C'] --> [B, S', H_q, C_qk]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.qk_head_dim)              # [B, S', C'] --> [B, S', H_kv, C_qk]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.v_head_dim)               # [B, S', C'] --> [B, S', H_kv, C_v]
        q = self.q_norm(q)                                                              # [B, S', H_q, C_qk]
        k = self.k_norm(k)                                                              # [B, S', H_kv, C_qk]
        v = self.v_norm(v)                                                              # [B, S', H_kv, C_v]

        # Apply RoPE
        q = self.rope(q)  # [B, S', H_q, C_qk]
        k = self.rope(k)  # [B, S', H_kv, C_qk]

        # Reshape K and V for grouped-query attention
        k = k.repeat(1, 1, self.head_repeats, 1)                                        # [B, S', H_kv, C_qk] --> [B, S', H_q, C_qk]
        v = v.repeat(1, 1, self.head_repeats, 1)                                        # [B, S', H_kv, C_v] --> [B, S', H_q, C_v]

        # Compute attention weights
        logits = torch.einsum('bshc,bShc->bhsS', q, k) / (self.qk_head_dim ** 0.5)
        if attn_bias is not None:
            logits = logits + attn_bias                                                 # [B, H_q, S', S']
        logits = torch.tanh(logits / self.logit_clip_value) * self.logit_clip_value
        attn = torch.softmax(logits, dim=-1)                                            # [B, H_q, S', S']

        # Output calculation
        y = torch.einsum('bhsS,bShc->bshc', attn, v)                                    # [B, H_q, S', C_v]
        y = y.permute(0, 2, 1, 3)                                                       # [B, S', H_q, C_v]
        y = y.reshape(B, S, self.num_q_heads * self.v_head_dim)                         # [B, S', H_q * C_v]
        y = self.out(y)                                                                 # [B, S', C]
        
        return y


class MLPBlock(nn.Module):
    def __init__(self, num_channels, mlp_ratio, dropout):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels        # C'
        self.mlp_ratio = mlp_ratio              # M
        self.dropout = dropout

        # Modules
        self.bn1 = EMA_RMSBatchNorm(self.num_channels)
        self.fc1 = nn.Linear(self.num_channels, self.num_channels * self.mlp_ratio)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_channels * self.mlp_ratio, self.num_channels)
        self.bn2 = EMA_RMSBatchNorm(self.num_channels)
        self.block = nn.Sequential(self.bn1, self.fc1, self.act, self.dropout, self.fc2, self.bn2, self.dropout)

    def forward(self, x):       # [B, S, C]
        return self.block(x)    # [B, S, C]


class AttentionBias(nn.Module):
    def __init__(self, pair_channels, pair_downsample_width, num_q_heads, num_kv_heads):
        super().__init__()
        self.pair_channels = pair_channels                                              # F
        self.pair_downsample_width = pair_downsample_width                              # W_p
        self.num_q_heads = num_q_heads                                                  # H_q
        self.num_kv_heads = num_kv_heads                                                # H_kv
        self.head_group_size = num_q_heads // num_kv_heads                              # G = H_q // H_kv
        self.bn = EMA_RMSBatchNorm(self.pair_channels, channels_dim=3)
        self.act = nn.GELU()
        self.fc = nn.Linear(self.pair_channels, self.head_group_size, bias=False)

    def forward(self, x):                                                               # [B, P, P, F]
        x = self.bn(x)                                                                  # [B, P, P, F]
        x = self.act(x)                                                                 # [B, P, P, F]
        x = self.fc(x)                                                                  # [B, P, P, G]
        x = x.repeat_interleave(self.pair_downsample_width, dim=1)                      # [B, S', P, G]
        x = x.repeat_interleave(self.pair_downsample_width, dim=2)                      # [B, S', S', G]
        x = x.permute(0, 3, 1, 2)                                                       # [B, G, S', S']
        x = x.repeat_interleave(self.num_kv_heads, dim=1)                               # [B, H_q, S', S']
        return x                                                                        # [B, H_q, S', S']


class TransformerTowerBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        transformer_seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        do_pair_update: bool,
        pair_downsample_width: int,
        pair_seq_len: int,
        pair_channels: int,
        pair_heads: int,
        pos_channels: int,
        mlp_ratio=2,
        dropout=0.0,
    ):
        super().__init__()
        # Read and check inputs
        self.num_channels = num_channels                        # C'
        self.transformer_seq_len = transformer_seq_len          # S'
        self.num_q_heads = num_q_heads                          # H_q
        self.num_kv_heads = num_kv_heads                        # H_kv
        self.qk_head_dim = qk_head_dim                          # D_qk
        self.v_head_dim = v_head_dim                            # D_v
        self.do_pair_update = do_pair_update
        self.pair_downsample_width = pair_downsample_width      # W_p
        self.pair_seq_len = pair_seq_len                        # P
        self.pair_channels = pair_channels                      # F
        self.pair_heads = pair_heads                            # H_p
        self.pos_channels = pos_channels                        # C_p
        self.mlp_ratio = mlp_ratio                              # M
        self.dropout = dropout
        assert self.num_q_heads % self.num_kv_heads == 0

        # Modules
        if do_pair_update:
            self.pair_update = PairUpdateBlock(
                num_channels=self.num_channels,
                pair_downsample_width=self.pair_downsample_width,
                pair_seq_len=self.pair_seq_len,
                pair_channels=self.pair_channels,
                pair_heads=self.pair_heads,
                pos_channels=self.pos_channels,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
        self.attn_bias = AttentionBias(
            pair_channels=self.pair_channels,
            pair_downsample_width=self.pair_downsample_width,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads
        )
        self.mha = MHA(
            self.num_channels,
            self.transformer_seq_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.qk_head_dim,
            self.v_head_dim,
            self.dropout
        )
        self.mlp = MLPBlock(
            num_channels=self.num_channels,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout
        )

    def forward(self, x, pair_x):                       # x: [B, S', C'] | pair_x: [B, P, P, F] or None
        if self.do_pair_update:
            pair_x = self.pair_update(x, pair_x)        # [B, P, P, F]
        attn_bias = self.attn_bias(pair_x)              # [B, H_q // H_kv, S', S']
        x = x + self.mha(x, attn_bias=attn_bias)        # [B, S', C']
        x = x + self.mlp(x)                             # [B, S', C']
        return x, pair_x                                # [B, S', C'], [B, P, P, F]


class TransformerTower(nn.Module):
    def __init__(
        self,
        num_channels: int,
        transformer_seq_len: int,
        num_blocks: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        pair_downsample_width: int,
        pair_channels: int,
        pair_heads: int,
        pos_channels: int,
        mlp_ratio: int,
        dropout: float
    ):
        super().__init__()
        # Read and check inputs
        self.num_channels = num_channels                                                # C'
        self.transformer_seq_len = transformer_seq_len                                  # S'
        self.num_blocks = num_blocks                                                    # N
        self.num_q_heads = num_q_heads                                                  # H_q
        self.num_kv_heads = num_kv_heads                                                # H_kv
        self.qk_head_dim = qk_head_dim                                                  # D_qk
        self.v_head_dim = v_head_dim                                                    # D_v
        self.pair_downsample_width = pair_downsample_width                              # W_p
        self.pair_seq_len = self.transformer_seq_len // self.pair_downsample_width      # P
        self.pair_channels = pair_channels                                              # F
        self.pair_heads = pair_heads                                                    # H_p
        self.pos_channels = pos_channels                                                # C_p
        self.mlp_ratio = mlp_ratio                                                      # M
        self.dropout = dropout
        assert self.num_q_heads % self.num_kv_heads == 0

        # Define blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                TransformerTowerBlock(
                    num_channels=self.num_channels,
                    transformer_seq_len=self.transformer_seq_len,
                    num_q_heads=self.num_q_heads,
                    num_kv_heads=self.num_kv_heads,
                    qk_head_dim=self.qk_head_dim,
                    v_head_dim=self.v_head_dim,
                    do_pair_update=(i % 2 == 0),
                    pair_downsample_width=self.pair_downsample_width,
                    pair_seq_len=self.pair_seq_len,
                    pair_channels=self.pair_channels,
                    pair_heads=self.pair_heads,
                    pos_channels=self.pos_channels,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout
                )
            )

    def forward(self, x):                   # [B, S', C]
        pair_x = None                       # [None]
        for block in self.blocks:
            x, pair_x = block(x, pair_x)    # [B, S', C], [B, P, P, F]
        return x, pair_x                    # [B, S', C], [B, P, P, F]
