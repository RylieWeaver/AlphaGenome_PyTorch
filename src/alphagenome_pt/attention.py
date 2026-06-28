# External
import warnings
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from einops import rearrange, repeat

# Internal
from .layers import BatchNorm, LayerNorm



def _shift(x):
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
        self.bn1 = BatchNorm(self.num_channels, sync=self.sync_bn)

        # Q/K/V modules
        self.q_layer = nn.Linear(self.num_channels, self.num_q_heads * self.qk_head_dim, bias=False)
        self.k_layer = nn.Linear(self.num_channels, self.num_kv_heads * self.qk_head_dim, bias=False)
        self.v_layer = nn.Linear(self.num_channels, self.num_kv_heads * self.v_head_dim, bias=False)
        self.norm_q = LayerNorm(self.qk_head_dim)
        self.norm_k = LayerNorm(self.qk_head_dim)
        self.norm_v = LayerNorm(self.v_head_dim)

        # Output modules
        self.linear_embedding = nn.Linear(self.num_q_heads * self.v_head_dim, self.num_channels, bias=True)
        self.bn2 = BatchNorm(self.num_channels, sync=self.sync_bn, channels_dim=2)
        self.out_drop = nn.Dropout(self.dropout)

    def forward(self, x, attn_bias=None):                                       # x: [B, S', C'] | attn_bias: [B, G, S', S']
        # Setup
        B, S, _ = x.shape
        dtype = x.dtype

        # Initial normalization
        x = self.bn1(x)                                                         # [B, S', C']

        # Multihead Q, K, V followed by LN
        q = rearrange(
            self.q_layer(x), 'b s (g h c) -> b s g h c', 
            g=self.head_group_size, h=self.num_kv_heads, c=self.qk_head_dim      
        )                                                                       # [B, S', C'] --> [B, S', G, H_q, C_qk]
        k = rearrange(
            self.k_layer(x), 'b s (h c) -> b s h c', 
            h=self.num_kv_heads, c=self.qk_head_dim
        )                                                                       # [B, S', C'] --> [B, S', H_kv, C_qk]
        v = rearrange(
            self.v_layer(x), 'b s (h c) -> b s h c', 
            h=self.num_kv_heads, c=self.v_head_dim
        )                                                                       # [B, S', C'] --> [B, S', H_kv, C_v]
        q = self.norm_q(q)                                                      # [B, S', G, H_kv, C_qk]
        k = self.norm_k(k)                                                      # [B, S', H_kv, C_qk]
        v = self.norm_v(v)                                                      # [B, S', H_kv, C_v]
        
        # Apply RoPE
        q = apply_rope(q, None, max_position=self.max_transformer_seq_len)      # [B, S', G, H_kv, C_qk]
        k = apply_rope(k, None, max_position=self.max_transformer_seq_len)      # [B, S', H_kv, C_qk]

        # Demand upcast to float32 for sensitive ops
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
        y = self.out_drop(self.bn2(self.linear_embedding(y)))                           # [B, S', C]
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
        self.bn1 = BatchNorm(self.num_channels, sync=self.sync_bn)
        self.fc1 = nn.Linear(self.num_channels, self.num_channels * self.mlp_ratio)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_channels * self.mlp_ratio, self.num_channels)
        self.bn2 = BatchNorm(self.num_channels, sync=self.sync_bn)
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


class PairMLPBlock(nn.Module):
    def __init__(self, pair_channels, mlp_ratio, dropout=0.0, sync_bn=True):
        super().__init__()
        # Read inputs
        self.pair_channels = pair_channels      # F
        self.mlp_ratio = mlp_ratio              # M
        self.dropout = dropout
        self.sync_bn = sync_bn

        # Modules
        self.norm = LayerNorm(self.pair_channels, channels_dim=3)
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


class AttentionBiasBlock(nn.Module):
    def __init__(self, pair_channels, pair_downsample_width, num_q_heads, num_kv_heads, sync_bn=True):
        super().__init__()
        self.pair_channels = pair_channels                                              # F
        self.pair_downsample_width = pair_downsample_width                              # W_p
        self.num_q_heads = num_q_heads                                                  # H_q
        self.num_kv_heads = num_kv_heads                                                # H_kv
        self.head_group_size = num_q_heads // num_kv_heads                              # G = H_q // H_kv
        self.sync_bn = sync_bn
        self.bn = BatchNorm(self.pair_channels, sync=self.sync_bn, channels_dim=3)
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
        self.norm = LayerNorm(self.pair_channels, channels_dim=3, rms_norm=True)

        # Projections
        self.linear_q = nn.Linear(self.pair_channels, self.pair_channels, bias=False)
        self.linear_k = nn.Linear(self.pair_channels, self.pair_channels, bias=False)
        self.linear_v = nn.Linear(self.pair_channels, self.pair_channels, bias=True)

        # Output
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # Setup
        dtype = x.dtype

        # Normalize inputs
        x = self.norm(x)                                                    # [B, P, P, F]

        # Reshape to treat each row independently and project
        q = self.linear_q(x)                                                # [B, P, P, F]
        k = self.linear_k(x)                                                # [B, P, P, F]
        v = self.linear_v(x)                                                # [B, P, P, F]

        # Demand upcast to float32 for sensitive ops
        q, k, v = q.float(), k.float(), v.float()

        # Compute attention update (back to original dtype)
        d = q.shape[-1]
        logits = torch.einsum('bpPf,bpkf->bpPk', q, k) / (d**0.5)           # [B, P, P, P]
        a = torch.softmax(logits, dim=3)                                    # [B, P, P, P]
        x = torch.einsum('bpPk,bpkf->bpPf', a, v).to(dtype)                 # [B, P, P, F]

        # Output
        x = self.dropout(x)                                                 # [B, P, P, F]
        return x


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
        self.norm = LayerNorm(self.num_channels, channels_dim=2, rms_norm=True)

        # Non-positional attention
        self.linear_q = nn.Linear(self.num_channels, self.pair_heads * self.pair_channels, bias=False)
        self.linear_k = nn.Linear(self.num_channels, self.pair_heads * self.pair_channels, bias=False)

        # Positional attention
        self.linear_pos_features = nn.Linear(self.pos_channels, self.pair_heads * self.pair_channels)
        self.q_r_bias = nn.Parameter(torch.zeros(1, 1, self.pair_heads, self.pair_channels))
        self.k_r_bias = nn.Parameter(torch.zeros(1, 1, self.pair_heads, self.pair_channels))

        # Output
        self.act = nn.GELU()
        self.linear_y_q = nn.Linear(self.num_channels, self.pair_channels, bias=False)
        self.linear_y_k = nn.Linear(self.num_channels, self.pair_channels, bias=False)
        self.linear_pair = nn.Linear(self.pair_heads, self.pair_channels)
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
        q = self.linear_q(x).reshape(B, P, H, F)                                                            # [B, P, H_p, F]
        k = self.linear_k(x).reshape(B, P, H, F)                                                            # [B, P, H_p, F]

        # Positional attention
        relative_positions = torch.arange(-P, P, device=device)                                             # [2P] ([-L, ..., -1, 0, 1, ..., L-1])
        pos_features = central_mask_features(
            rel=relative_positions, feature_size=self.pos_channels,                                         # [2P, C_p]
            max_sequence_length=P, device=device, dtype=dtype
        )
        pos_encoding = self.linear_pos_features(pos_features).reshape(2*P, H, F)                            # [2P, H_p, F]
        rel_q_a = torch.einsum('bqhc,phc->bqph', q + self.q_r_bias, pos_encoding)                           # [B, P, 2P, H_p]
        rel_q_a = _shift(rel_q_a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                                   # [B, P, P, H_p]
        rel_k_a = torch.einsum('bkhc,phc->bkph', k + self.k_r_bias, pos_encoding)                           # [B, P, 2P, H_p]
        rel_k_a = _shift(rel_k_a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                                   # [B, P, P, H_p]
        a = torch.einsum('bqhc,bkhc->bqkh', q, k) + 0.5*(rel_q_a + rel_k_a.transpose(1, 2))                 # [B, P, P, H_p]

        # Output
        y_q = self.linear_y_q(self.act(x))                                                                  # [B, P, F]
        y_k = self.linear_y_k(self.act(x))                                                                  # [B, P, F]
        pair_activations = self.linear_pair(a) + y_q[:, :, None, :] + y_k[:, None, :, :]                    # [B, P, P, F]
        return self.dropout(pair_activations)                                                               # [B, P, P, F]


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
        self.row_attention_block = RowAttentionBlock(
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

    def forward(self, sequence_input, pair_input):              # [B, S', C'] | [B, P, P, F]
        y = self.sequence_to_pair_block(sequence_input)         # [B, P, P, F]
        x = y if pair_input is None else pair_input + y         # [B, P, P, F]
        x = x + self.row_attention_block(x)                     # [B, P, P, F]
        x = x + self.pair_mlp_block(x)                          # [B, P, P, F]
        return x                                                # [B, P, P, F]


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
        self.attn_bias = AttentionBiasBlock(
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
