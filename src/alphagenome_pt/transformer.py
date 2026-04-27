# Provenance: Reimplementation based on the AlphaGenome bioRxiv paper. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General

# Torch
import torch
import torch.nn as nn
from einops import rearrange, repeat

# AlphaGenome
from .pair_update import PairUpdateBlock
from .utils import RMSBatchNorm



def geomspace(start, end, steps, device=None, dtype=None):
    start = torch.as_tensor(start, device=device, dtype=dtype)
    end = torch.as_tensor(end, device=device, dtype=dtype)
    # log-space then exp
    return torch.exp(torch.linspace(torch.log(start), torch.log(end), steps, device=device, dtype=dtype))

def apply_rope(x: torch.Tensor, positions: torch.Tensor | None, max_position: int):
        """
        AlphaGenome uses a custom frequency schedule for Rotary Positional Encodings (RoPE).

        NOTE: This RoPE function is shape-agnostic AS LONG AS it obeys [B, S, *, C] where C is even.
        This is important to allow it to be used at various points in the model, including attention
        and Splicing Heads, which have different shapes. Simply creating a theta tensor with the last
        dimensions and broadcasting would NOT always give what we want, since the sequence dimension 
        might not be in the second to last place (e.g. in Splicing Head, the shape is [B, P, T, C], 
        where P is the sequnce length, T is the number of tissues, and C is the number of channels).

        Apply Rotary Positional Embeddings (RoPE) to tensor 'x'.
        - x: [B, S, *, C] where C is even.
        - positions: [B, S] or [S] or None. If None, uses arange(S).
        - max_position: maximum position for frequency calculation.

        Notes:
        - x_rotated:
        - Rotates the x vector by 90 degrees in each 2D subspace by swapping evens/odds and negating the odds
        - Interleaves with a stack/reshape
        """
        # Read and check inputs
        B, S, *extra_dims, C = x.shape
        num_freq = C // 2
        device, dtype = x.device, x.dtype

        # Default positions
        if positions is None:
            positions = torch.arange(S, device=device, dtype=dtype)[None, :]    # [1, S]
        else:
            positions = positions.to(device=device, dtype=dtype)
            if positions.ndim == 1:
                positions = positions[None, :]                                  # [1, S]
            if positions.shape[-1] != S:
                raise ValueError(f"positions has last dim {positions.shape[-1]}, expected {S}")

        lin_space = torch.arange(num_freq, device=device, dtype=dtype)                          # [F]
        if num_freq > max_position:
            geom_space = torch.ones(num_freq, device=device, dtype=dtype)                      # [F]
        else:
            geom_space = geomspace(
                1.0,
                max_position - num_freq + 1.0,
                num_freq,
                device=device,
                dtype=dtype
            )                                                                                       # [F]
        inv_freq = 1.0 / (lin_space + geom_space)                                               # [F]

        # Compute angles
        theta = torch.einsum('b s, f -> b s f', positions, inv_freq)                            # [#B, S, F]
        theta = theta.reshape(*theta.shape[:2], *([1] * len(extra_dims)), num_freq)             # [#B, S, #*, F]
        theta = theta.repeat_interleave(2, dim=-1)                                              # [#B, S, ..., C]

        # Apply RoPE
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)             # [B, S, *, C]
        x = x * torch.cos(theta) + x_rotated * torch.sin(theta)                                 # [B, S, *, C]
        return x


class MHA(nn.Module):
    def __init__(
        self,
        num_channels: int,
        max_transformer_seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        dropout: float,
        sync_bn: bool = True,
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
        self.num_channels = num_channels                            # C'
        self.num_q_heads = num_q_heads                              # H_q
        self.num_kv_heads = num_kv_heads                            # H_kv
        self.head_group_size = num_q_heads // num_kv_heads          # G = H_q / H_kv
        self.qk_head_dim = qk_head_dim                              # C_qk
        self.v_head_dim = v_head_dim                                # C_v
        self.max_transformer_seq_len = max_transformer_seq_len      # S'
        self.logit_clip_value = 5.0
        self.dropout = dropout
        self.sync_bn = sync_bn
        assert self.num_q_heads % self.num_kv_heads == 0

        # Initial normalization
        self.bn1 = RMSBatchNorm(self.num_channels, sync=self.sync_bn)

        # Q/K/V modules
        self.q_proj = nn.Linear(self.num_channels, self.num_q_heads * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(self.num_channels, self.num_kv_heads * self.qk_head_dim, bias=False)
        self.v_proj = nn.Linear(self.num_channels, self.num_kv_heads * self.v_head_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.qk_head_dim)
        self.k_norm = nn.LayerNorm(self.qk_head_dim)
        self.v_norm = nn.LayerNorm(self.v_head_dim)

        # Output modules
        self.bn2 = RMSBatchNorm(self.num_channels, sync=self.sync_bn, channels_dim=2)
        self.out_lin = nn.Linear(self.num_q_heads * self.v_head_dim, self.num_channels, bias=True)
        self.out_drop = nn.Dropout(self.dropout)

    def forward(self, x, attn_bias=None):                                       # x: [B, S', C'] | attn_bias: [B, G, S', S']
        # Setup
        B, S, _ = x.shape
        dtype = x.dtype

        # Initial normalization
        x = self.bn1(x)                                                         # [B, S', C']

        # Multihead Q, K, V followed by LN
        q = rearrange(
            self.q_proj(x), 'b s (g h c) -> b s g h c', 
            g=self.head_group_size, h=self.num_kv_heads, c=self.qk_head_dim      
        )                                                                       # [B, S', C'] --> [B, S', G, H_q, C_qk]
        k = rearrange(
            self.k_proj(x), 'b s (h c) -> b s h c', 
            h=self.num_kv_heads, c=self.qk_head_dim
        )                                                                       # [B, S', C'] --> [B, S', H_kv, C_qk]
        v = rearrange(
            self.v_proj(x), 'b s (h c) -> b s h c', 
            h=self.num_kv_heads, c=self.v_head_dim
        )                                                                       # [B, S', C'] --> [B, S', H_kv, C_v]
        q = self.q_norm(q)                                                      # [B, S', G, H_kv, C_qk]
        k = self.k_norm(k)                                                      # [B, S', H_kv, C_qk]
        v = self.v_norm(v)                                                      # [B, S', H_kv, C_v]
        
        # Apply RoPE
        q = apply_rope(q, None, max_position=self.max_transformer_seq_len)      # [B, S', G, H_kv, C_qk]
        k = apply_rope(k, None, max_position=self.max_transformer_seq_len)      # [B, S', H_kv, C_qk]

        # Demand upcast for sensitive ops
        q, k, v = q.float(), k.float(), v.float()

        # Compute attention weights
        logits = torch.einsum('bsghc,bShc->bghsS', q, k) / (self.qk_head_dim ** 0.5)    # [B, G, H_kv, S', S']
        if attn_bias is not None:
            logits = logits + attn_bias.float()                                         # [B, G, H_kv, S', S']
        logits = torch.tanh(logits / self.logit_clip_value) * self.logit_clip_value
        attn = torch.softmax(logits, dim=-1)                                            # [B, G, H_kv, S', S']

        # Output calculation (back to original dtype)
        y = torch.einsum('bghsS,bShc->bsghc', attn, v).to(dtype)                        # [B, G, H_kv, S', C_v]
        y = rearrange(y, 'b s g h c -> b s (g h c)')                                    # [B, G, H_kv, S', C_v] --> [B, S', G, H_kv, C_v] --> [B, S', H_q * C_v]
        y = self.out_drop(self.bn2(self.out_lin(y)))                                    # [B, S', C]
        return y                                                                        # [B, S', C]


class MLPBlock(nn.Module):
    def __init__(self, num_channels, mlp_ratio, dropout, sync_bn=True):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels        # C'
        self.mlp_ratio = mlp_ratio              # M
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Modules
        self.bn1 = RMSBatchNorm(self.num_channels, sync=self.sync_bn)
        self.fc1 = nn.Linear(self.num_channels, self.num_channels * self.mlp_ratio)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_channels * self.mlp_ratio, self.num_channels)
        self.bn2 = RMSBatchNorm(self.num_channels, sync=self.sync_bn)
        self.drop2 = nn.Dropout(self.dropout)

    def forward(self, x):       # [B, S, C]
        x = self.bn1(x)         # [B, S, C]
        x = self.fc1(x)         # [B, S, C * M]
        x = self.act(x)         # [B, S, C * M]
        x = self.drop1(x)       # [B, S, C * M]
        x = self.fc2(x)         # [B, S, C]
        x = self.bn2(x)         # [B, S, C]
        x = self.drop2(x)       # [B, S, C]
        return x                # [B, S, C]


class AttentionBias(nn.Module):
    def __init__(self, pair_channels, pair_downsample_width, num_q_heads, num_kv_heads, sync_bn=True):
        super().__init__()
        self.pair_channels = pair_channels                                              # F
        self.pair_downsample_width = pair_downsample_width                              # W_p
        self.num_q_heads = num_q_heads                                                  # H_q
        self.num_kv_heads = num_kv_heads                                                # H_kv
        self.head_group_size = num_q_heads // num_kv_heads                              # G = H_q // H_kv
        self.sync_bn = sync_bn
        self.bn = RMSBatchNorm(self.pair_channels, sync=self.sync_bn, channels_dim=3)
        self.act = nn.GELU()
        self.fc = nn.Linear(self.pair_channels, self.num_q_heads, bias=False)

    def forward(self, x):                                                               # [B, P, P, F]
        x = self.bn(x)                                                                  # [B, P, P, F]
        x = self.act(x)                                                                 # [B, P, P, F]
        x = self.fc(x)                                                                  # [B, P, P, H_q]
        x = rearrange(x, 'b p1 p2 (g h) -> b g h p1 p2', g=self.head_group_size)        # [B, P, P, H_q] --> [B, G, H_kv, P, P]
        x = repeat(
            x, 'b g h p1 p2 -> b g h (p1 w1) (p2 w2)',                                  # [B, G, H_kv, P, P] --> [B, G, H_kv, S', S']
            w1=self.pair_downsample_width, w2=self.pair_downsample_width,
        )
        return x                                                                        # [B, G, H_kv, S', S']


class TransformerTowerBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        max_transformer_seq_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        do_pair_update: bool,
        pair_downsample_width: int,
        max_pair_seq_len: int,
        pair_channels: int,
        pair_heads: int,
        pos_channels: int,
        mlp_ratio=2,
        dropout=0.0,
        sync_bn=True,
    ):
        super().__init__()
        # Read and check inputs
        self.num_channels = num_channels                        # C'
        self.max_transformer_seq_len = max_transformer_seq_len  # S'
        self.num_q_heads = num_q_heads                          # H_q
        self.num_kv_heads = num_kv_heads                        # H_kv
        self.qk_head_dim = qk_head_dim                          # D_qk
        self.v_head_dim = v_head_dim                            # D_v
        self.do_pair_update = do_pair_update
        self.pair_downsample_width = pair_downsample_width      # W_p
        self.max_pair_seq_len = max_pair_seq_len                 # P
        self.pair_channels = pair_channels                      # F
        self.pair_heads = pair_heads                            # H_p
        self.pos_channels = pos_channels                        # C_p
        self.mlp_ratio = mlp_ratio                              # M
        self.dropout = dropout
        self.sync_bn = sync_bn
        assert self.num_q_heads % self.num_kv_heads == 0

        # Modules
        if do_pair_update:
            self.pair_update = PairUpdateBlock(
                num_channels=self.num_channels,
                pair_downsample_width=self.pair_downsample_width,
                max_pair_seq_len=self.max_pair_seq_len,
                pair_channels=self.pair_channels,
                pair_heads=self.pair_heads,
                pos_channels=self.pos_channels,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                sync_bn=self.sync_bn
            )
        self.attn_bias = AttentionBias(
            pair_channels=self.pair_channels,
            pair_downsample_width=self.pair_downsample_width,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            sync_bn=self.sync_bn
        )
        self.mha = MHA(
            self.num_channels,
            self.max_transformer_seq_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.qk_head_dim,
            self.v_head_dim,
            self.dropout,
            sync_bn=self.sync_bn
        )
        self.mlp = MLPBlock(
            num_channels=self.num_channels,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            sync_bn=self.sync_bn
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
        max_transformer_seq_len: int,
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
        dropout: float,
        sync_bn: bool = True
    ):
        super().__init__()
        # Read and check inputs
        self.num_channels = num_channels                                                        # C'
        self.max_transformer_seq_len = max_transformer_seq_len                                  # S'
        self.num_blocks = num_blocks                                                            # N
        self.num_q_heads = num_q_heads                                                          # H_q
        self.num_kv_heads = num_kv_heads                                                        # H_kv
        self.qk_head_dim = qk_head_dim                                                          # D_qk
        self.v_head_dim = v_head_dim                                                            # D_v
        self.pair_downsample_width = pair_downsample_width                                      # W_p
        self.max_pair_seq_len = self.max_transformer_seq_len // self.pair_downsample_width      # P
        self.pair_channels = pair_channels                                                      # F
        self.pair_heads = pair_heads                                                            # H_p
        self.pos_channels = pos_channels                                                        # C_p
        self.mlp_ratio = mlp_ratio                                                              # M
        self.dropout = dropout
        self.sync_bn = sync_bn
        assert self.num_q_heads % self.num_kv_heads == 0

        # Define blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                TransformerTowerBlock(
                    num_channels=self.num_channels,
                    max_transformer_seq_len=self.max_transformer_seq_len,
                    num_q_heads=self.num_q_heads,
                    num_kv_heads=self.num_kv_heads,
                    qk_head_dim=self.qk_head_dim,
                    v_head_dim=self.v_head_dim,
                    do_pair_update=(i % 2 == 0),
                    pair_downsample_width=self.pair_downsample_width,
                    max_pair_seq_len=self.max_pair_seq_len,
                    pair_channels=self.pair_channels,
                    pair_heads=self.pair_heads,
                    pos_channels=self.pos_channels,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    sync_bn=self.sync_bn
                )
            )

    def forward(self, x):                   # [B, S', C]
        pair_x = None                       # [None]
        for block in self.blocks:
            x, pair_x = block(x, pair_x)    # [B, S', C], [B, P, P, F]
        return x, pair_x                    # [B, S', C], [B, P, P, F]
