# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

# General
from dataclasses import dataclass

# Torch
import torch

# AlphaGenome



@dataclass
class Embeddings:
    """
    AlphaGenome embeddings.
    - S: sequence length
    - S': S // 128
    - P: S // 2048
    """
    
    embeddings_1bp: torch.Tensor | None = None        # [B, S, C_0]
    embeddings_128bp: torch.Tensor | None = None      # [B, S', C_7]
    embeddings_pair: torch.Tensor | None = None       # [B, P, P, F]

    def get_sequence_embeddings(self, resolution: int) -> torch.Tensor:
        if resolution == 128:
            return self.embeddings_128bp
        elif resolution == 1:
            return self.embeddings_1bp
        else:
            raise ValueError(f'Unsupported resolution: {resolution}')
