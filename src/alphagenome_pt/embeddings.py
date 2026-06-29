# External
from dataclasses import dataclass
import torch
import torch.nn as nn
from einx import add
from einops import repeat

# Internal
from .layers import BatchNorm, LayerNorm, GELU_1702



@dataclass
class Embeddings:
    """
    AlphaGenome embeddings.
    - S: sequence length
    - S': S // 128
    - P: S // 2048
    """
    
    embeddings_1bp: torch.Tensor | None = None        # [B, S, C_0]
    embeddings_128bp: torch.Tensor | None = None      # [B, S', C_stages]
    embeddings_pair: torch.Tensor | None = None       # [B, P, P, F]

    def get_sequence_embeddings(self, resolution: int) -> torch.Tensor:
        if resolution == 128:
            return self.embeddings_128bp
        elif resolution == 1:
            return self.embeddings_1bp
        else:
            raise ValueError(f'Unsupported resolution: {resolution}')


class OutputEmbedder(nn.Module):
    def __init__(self, num_channels, num_organisms, mlp_ratio, skip_channels=None, sync_bn=True):
        super().__init__()
        # Read inputs
        self.num_channels = num_channels
        self.mlp_ratio = mlp_ratio
        self.up_channels = num_channels * mlp_ratio
        self.skip_channels = skip_channels
        self.num_organisms = num_organisms
        self.sync_bn = sync_bn

        # Modules
        self.fc1 = nn.Linear(self.num_channels, self.up_channels)
        self.fc_skip = nn.Linear(self.skip_channels, self.up_channels, bias=False) if self.skip_channels is not None else None
        self.org = nn.Embedding(self.num_organisms, self.up_channels)
        self.norm = BatchNorm(self.up_channels, sync=self.sync_bn, channels_dim=2, rms_norm=True)
        self.activation = GELU_1702()


    def forward(self, x, organism_index, skip_x=None):                  # [B, S, C], [B, S', C'], int
        # Project input
        x = self.fc1(x)                                                 # [B, S, 2C]

        # Add skip connection
        if skip_x is not None:
            skip_x = self.fc_skip(skip_x)                               # [B, S', C]
            repeats = x.shape[1] // skip_x.shape[1]
            x = x + repeat(skip_x, 'b s c -> b (s r) c', r=repeats)     # [B, S, C]

        # Norm, add organism embedding, and output
        x = self.norm(x)                                                # [B, S, C]
        if self.num_organisms >= 1:
            org_emb = self.org(organism_index)                          # [B, 1, C]
            x = add('b s c, b c -> b s c', x, org_emb)                  # [B, S, 2C] + [B, 2C] --> [B, S, 2C]
        x = self.activation(x)                                          # [B, S, 2C]
        return x                                                        # [B, S, 2C]


class OutputPairEmbedder(nn.Module):
    def __init__(
        self,
        pair_channels: int,
        num_organisms: int,
    ):
        super().__init__()
        self.num_organisms = num_organisms
        self.norm = LayerNorm(pair_channels, channels_dim=3, rms_norm=True)
        self.organism_embed = nn.Embedding(num_organisms, pair_channels)
        self.act = GELU_1702()

    def forward(self, x, organism_index):                           # x: [B, P, P, F] | organism_index: [B]
        # Symmetrize
        x = add('b p1 p2 f, b p2 p1 f -> b p1 p2 f', x, x) / 2.0    # [B, P, P, F] + [B, P, P, F] --> [B, P, P, F]

        # Normalize
        x = self.norm(x)                                            # [B, P, P, F]

        # Get organism embedding
        if self.num_organisms >= 1:
            org_emb = self.organism_embed(organism_index)
            x = add('b p1 p2 f, b f -> b p1 p2 f', x, org_emb)      # [B, P, P, F] + [B, F] --> [B, P, P, F]

        # Apply activation
        x = self.act(x)                                             # [B, P, P, F]
        return x
