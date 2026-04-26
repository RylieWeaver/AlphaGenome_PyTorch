"""
JAX checkpoint to current alphagenome_pt state_dict mapping.
- It's made fairly explicitly to be easy to read and modify, at the cost of some verbosity and repetition.
- Comments are included to show the corresponding JAX and PyTorch keys and shapes for each mapping, 
  to make it easy to verify correctness and add new mappings as needed.
"""

from __future__ import annotations

# General
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Literal, Mapping, Sequence
from abc import ABC, abstractmethod

# Torch
from torch import Tensor
from einops import rearrange

# JAX2PT
from utils import jax_to_torch_tensor



"""
DEFINED OBJECTS:
"""
@dataclass(frozen=True)
class MapSpec:
    """
    Mapping from one or more flat JAX keys to one or more torch state_dict keys.
    Prefixes and suffixes are concatenated directly; include any "/" or "." separators
    in the prefix, key, or suffix strings.
    """
    jax_keys: tuple[str, ...] | str
    torch_keys: tuple[str, ...] | str
    transform: TransformFn
    jax_prefix: str | None = None
    jax_suffix: str | None = None
    torch_prefix: str | None = None
    torch_suffix: str | None = None
    description: str | None = None

    def _full_keys(self, prefix: str | None, keys: tuple[str, ...] | str, suffix: str | None) -> tuple[str, ...]:
        keys = (keys,) if isinstance(keys, str) else keys
        full_keys = []
        for key in keys:
            full_keys.append(f"{prefix or ''}{key}{suffix or ''}")
        return tuple(full_keys)

    @property
    def jax_full_keys(self) -> tuple[str, ...]:
        return self._full_keys(self.jax_prefix, self.jax_keys, self.jax_suffix)

    @property
    def torch_full_keys(self) -> tuple[str, ...]:
        return self._full_keys(self.torch_prefix, self.torch_keys, self.torch_suffix)

    def convert(self, flat_jax: Mapping[str, Any]) -> dict[str, Tensor]:
        source_tensors = [jax_to_torch_tensor(flat_jax[key]) for key in self.jax_full_keys]
        converted = self.transform(source_tensors)
        converted_tensors = [converted.clone() for _ in self.torch_full_keys]
        return dict(zip(self.torch_full_keys, converted_tensors))


class TransformFn(ABC):
    @abstractmethod
    def __call__(self, tensors: Sequence[Tensor] | Tensor) -> Tensor:
        pass

class Identity(TransformFn):
    def __call__(self, tensors: Sequence[Tensor] | Tensor) -> Tensor:
        if isinstance(tensors, Tensor):
            tensors = [tensors]
        assert len(tensors) == 1, f"identity expects 1 tensor, got {len(tensors)}"
        return tensors[0]

class Rearrange(TransformFn):
    def __init__(self, pattern: str, **axes_lengths: int):
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def __call__(self, tensors: Sequence[Tensor] | Tensor) -> Tensor:
        if isinstance(tensors, Tensor):
            tensors = [tensors]
        assert len(tensors) == 1, f"rearrange expects 1 tensor, got {len(tensors)}"
        return rearrange(tensors[0], self.pattern, **self.axes_lengths)

    

"""
PREPARATION FUNCTIONS: 
"""



"""
DEFINED MAPPINGS:
- The separation of mappings is somewhat arbitrary.
"""

def build_etc_mappings() -> list[MapSpec]:
    mappings = []

    """Org Embedder"""
    # alphagenome/embed/embeddings: [2x1536] (float32)
    # org_embedder.weight: [2x1536] (float32)
    mappings.append(
        MapSpec(
            jax_keys=("alphagenome/embed/embeddings"),
            torch_keys=("org_embedder.weight"),
            transform=Identity(),
        ),
    )
    return mappings


def build_encoder_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []

    """DNA EMBEDDER / BIN SIZE 1"""
    jax_prefix = f"alphagenome/sequence_encoder/dna_embedder/"
    torch_prefix = f"encoder.downres_blocks.bin_size_1."
    # alphagenome/sequence_encoder/dna_embedder/conv1_d/b: [768] (float32)
    # alphagenome/sequence_encoder/dna_embedder/conv1_d/w: [15x4x768] (float32)
    # encoder.downres_blocks.bin_size_1.conv.bias: [768] (float32)
    # encoder.downres_blocks.bin_size_1.conv.weight: [768x4x15] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv1_d/b"),
            torch_prefix=torch_prefix,
            torch_keys=("conv.bias"),
            transform=Identity()
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv1_d/w"),
            torch_prefix=torch_prefix,
            torch_keys=("conv.weight"),
            transform=Rearrange("w c1 c2 -> c2 c1 w")
        )
    ])
    # alphagenome/sequence_encoder/dna_embedder/conv_block/rms_batch_norm/offset: [1x1x768] (float32)
    # alphagenome/sequence_encoder/dna_embedder/conv_block/rms_batch_norm/scale: [1x1x768] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.norm.beta: [768] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.norm.gamma: [768] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv_block/rms_batch_norm/offset"),
            torch_prefix=torch_prefix,
            torch_keys=("conv_block.norm.beta"),
            transform=Rearrange("() () c -> c"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv_block/rms_batch_norm/scale"),
            torch_prefix=torch_prefix,
            torch_keys=("conv_block.norm.gamma"),
            transform=Rearrange("() () c -> c"),
        )
    ])
    # alphagenome/sequence_encoder/dna_embedder/conv_block/standardized_conv1_d/bias: [768] (float32)
    # alphagenome/sequence_encoder/dna_embedder/conv_block/standardized_conv1_d/scale: [1x1x768] (float32)
    # alphagenome/sequence_encoder/dna_embedder/conv_block/standardized_conv1_d/w: [5x768x768] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.conv.bias: [768] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.conv.scale: [768x1x1] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.conv.weight: [768x768x5] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv_block/standardized_conv1_d/bias"),
            torch_prefix=torch_prefix,
            torch_keys=("conv_block.conv.bias"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv_block/standardized_conv1_d/scale"),
            torch_prefix=torch_prefix,
            torch_keys=("conv_block.conv.scale"),
            transform=Rearrange("() () c -> c () ()"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("conv_block/standardized_conv1_d/w"),
            torch_prefix=torch_prefix,
            torch_keys=("conv_block.conv.weight"),
            transform=Rearrange("w c1 c2 -> c2 c1 w"),
        )
    ])

    """ENCODERS / BIN SIZES > 2"""
    jax_indices = ["_0", "_1", "_2", "_3", "_4", "_5"]
    torch_indices = ["2", "4", "8", "16", "32", "64"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_encoder/downres_block{jax_idx}/"
        torch_prefix = f"encoder.downres_blocks.bin_size_{torch_idx}."
        # alphagenome/sequence_encoder/downres_block_0/conv_block/rms_batch_norm/offset: [1x1x768] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block/rms_batch_norm/scale: [1x1x768] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.norm.beta: [768] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.norm.gamma: [768] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.norm.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.norm.gamma"),
                transform=Rearrange("() () c -> c"),
            )
        ])
        # alphagenome/sequence_encoder/downres_block_0/conv_block/standardized_conv1_d/bias: [896] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block/standardized_conv1_d/scale: [1x1x896] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block/standardized_conv1_d/w: [5x768x896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.conv.bias: [896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.conv.scale: [896x1x1] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.conv.weight: [896x768x5] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/standardized_conv1_d/bias"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.conv.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/standardized_conv1_d/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.conv.scale"),
                transform=Rearrange("b s c -> c s b"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/standardized_conv1_d/w"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.conv.weight"),
                transform=Rearrange("w c1 c2 -> c2 c1 w"),
            )
        ])
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/rms_batch_norm/offset: [1x1x896] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/rms_batch_norm/scale: [1x1x896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.norm.beta: [896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.norm.gamma: [896] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.norm.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.norm.gamma"),
                transform=Rearrange("() () c -> c"),
            )
        ])
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/standardized_conv1_d/bias: [896] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/standardized_conv1_d/scale: [1x1x896] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/standardized_conv1_d/w: [5x896x896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.conv.bias: [896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.conv.scale: [896x1x1] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.conv.weight: [896x896x5] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/standardized_conv1_d/bias"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.conv.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/standardized_conv1_d/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.conv.scale"),
                transform=Rearrange("b s c -> c s b"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/standardized_conv1_d/w"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.conv.weight"),
                transform=Rearrange("w c1 c2 -> c2 c1 w"),
            )
        ])
    return mappings


def build_transformer_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []
    jax_prefix = f"alphagenome/transformer_tower/"
    torch_prefix = f"transformer.blocks."

    """Transformer Blocks"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]
    torch_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        ### MHA ###
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm/offset: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm/scale: [1x1x1536] (float32)
        # transformer.blocks.2.mha.bn1.beta: [1536] (float32)
        # transformer.blocks.2.mha.bn1.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.bn1.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.bn1.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/transformer_tower/mha_block_1/q_layer/w: [1536x1024] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_q/offset: [128] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_q/scale: [128] (float32)
        # transformer.blocks.2.mha.q_proj.weight: [1024x1536] (float32)
        # transformer.blocks.2.mha.q_norm.bias: [128] (float32)
        # transformer.blocks.2.mha.q_norm.weight: [128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/q_layer/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.q_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_q/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.q_norm.bias"),
                transform=Identity()
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_q/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.q_norm.weight"),
                transform=Identity()
            )
        ])
        # alphagenome/transformer_tower/mha_block_1/k_layer/w: [1536x128] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_k/offset: [128] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_k/scale: [128] (float32)
        # transformer.blocks.2.mha.k_proj.weight: [128x1536] (float32)
        # transformer.blocks.2.mha.k_norm.bias: [128] (float32)
        # transformer.blocks.2.mha.k_norm.weight: [128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/k_layer/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.k_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_k/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.k_norm.bias"),
                transform=Identity()
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_k/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.k_norm.weight"),
                transform=Identity()
            )
        ])
        # alphagenome/transformer_tower/mha_block_1/v_layer/w: [1536x192] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_v/offset: [192] (float32)
        # alphagenome/transformer_tower/mha_block_1/norm_v/scale: [192] (float32)
        # transformer.blocks.2.mha.v_proj.weight: [192x1536] (float32)
        # transformer.blocks.2.mha.v_norm.bias: [192] (float32)
        # transformer.blocks.2.mha.v_norm.weight: [192] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/v_layer/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.v_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_v/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.v_norm.bias"),
                transform=Identity()
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/norm_v/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.v_norm.weight"),
                transform=Identity()
            )
        ])
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm_1/offset: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm_1/scale: [1x1x1536] (float32)
        # transformer.blocks.2.mha.bn2.beta: [1536] (float32)
        # transformer.blocks.2.mha.bn2.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm_1/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.bn2.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm_1/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.bn2.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/transformer_tower/mha_block_1/linear_embedding/b: [1536] (float32)
        # alphagenome/transformer_tower/mha_block_1/linear_embedding/w: [1536x1536] (float32)
        # transformer.blocks.2.mha.out_lin.bias: [1536] (float32)
        # transformer.blocks.2.mha.out_lin.weight: [1536x1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/linear_embedding/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.out_lin.bias"),
                transform=Identity()
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/linear_embedding/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mha.out_lin.weight"),
                transform=Rearrange("c1 c2 -> c2 c1")
            )
        ])
        ### Fully Connected ###
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm/offset: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm/scale: [1x1x1536] (float32)
        # transformer.blocks.2.mlp.bn1.beta: [1536] (float32)
        # transformer.blocks.2.mlp.bn1.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.bn1.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm_1/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.bn1.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/transformer_tower/mlp_block_1/linear/b: [3072] (float32)
        # alphagenome/transformer_tower/mlp_block_1/linear/w: [1536x3072] (float32)
        # transformer.blocks.2.mlp.fc1.bias: [3072] (float32)
        # transformer.blocks.2.mlp.fc1.weight: [3072x1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/linear/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.fc1.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/linear/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.fc1.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/mlp_block_1/linear_1/b: [1536] (float32)
        # alphagenome/transformer_tower/mlp_block_1/linear_1/w: [3072x1536] (float32)
        # transformer.blocks.2.mlp.fc2.bias: [1536] (float32)
        # transformer.blocks.2.mlp.fc2.weight: [1536x3072] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/linear_1/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.fc2.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/linear_1/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.fc2.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm_1/offset: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm_1/scale: [1x1x1536] (float32)
        # transformer.blocks.2.mlp.bn2.beta: [1536] (float32)
        # transformer.blocks.2.mlp.bn2.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm_1/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.bn2.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.mlp.bn2.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        ### Attn Bias ###
        # alphagenome/transformer_tower/attention_bias_block_1/rms_batch_norm/offset: [1x1x1x128] (float32)
        # alphagenome/transformer_tower/attention_bias_block_1/rms_batch_norm/scale: [1x1x1x128] (float32)
        # transformer.blocks.2.attn_bias.bn.beta: [128] (float32)
        # transformer.blocks.2.attn_bias.bn.gamma: [128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"attention_bias_block{jax_idx}/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.attn_bias.bn.beta"),
                transform=Rearrange("() () () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"attention_bias_block{jax_idx}/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.attn_bias.bn.gamma"),
                transform=Rearrange("() () () c -> c"),
            )
        ])
        # alphagenome/transformer_tower/attention_bias_block_1/linear/w: [128x8] (float32)
        # transformer.blocks.2.attn_bias.fc.weight: [8x128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"attention_bias_block{jax_idx}/linear/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.attn_bias.fc.weight"),
                transform=Rearrange("h c -> c h"),
            )
        ])
    
    """Pair Update Blocks"""
    jax_indices = ["", "_1", "_2", "_3", "_4"]
    torch_indices = ["0", "2", "4", "6", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        ### Seq2Pair ###
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/norm_seq2pair/offset: [1536] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/norm_seq2pair/scale: [1536] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.norm.beta: [1536] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.norm.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/norm_seq2pair/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.norm.beta"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/norm_seq2pair/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.norm.gamma"),
                transform=Identity(),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_q/w: [1536x4096] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_k/w: [1536x4096] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.q_linear.weight: [4096x1536] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.k_linear.weight: [4096x1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_q/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.q_linear.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_k/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.k_linear.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_pos_features/b: [4096] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_pos_features/w: [64x4096] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.pos_linear.bias: [4096] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.pos_linear.weight: [4096x64] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pos_features/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.pos_linear.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pos_features/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.pos_linear.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/q_r_bias: [1x1x32x128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/k_r_bias: [1x1x32x128] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.q_r_bias: [1x1x32x128] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.k_r_bias: [1x1x32x128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/q_r_bias"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.q_r_bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/k_r_bias"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.k_r_bias"),
                transform=Identity(),
            ),
        ])        
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_pair/b: [128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_pair/w: [32x128] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.out.bias: [128] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.out.weight: [128x32] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pair/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.out.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pair/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.out.weight"),
                transform=Rearrange("c1 c2 -> c2 c1")
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_y_q/w: [1536x128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/sequence_to_pair_block/linear_y_k/w: [1536x128] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.y_q_linear.weight: [128x1536] (float32)
        # transformer.blocks.2.pair_update.sequence_to_pair_block.y_k_linear.weight: [128x1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_y_q/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.y_q_linear.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_y_k/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.y_k_linear.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        ### Row Attention ###
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/layer_norm/offset: [128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/layer_norm/scale: [128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.norm.beta: [128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.norm.gamma: [128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/layer_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.norm.beta"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/layer_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.norm.gamma"),
                transform=Identity(),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/linear_k/w: [128x128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/linear_q/w: [128x128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.k_proj.weight: [128x128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.q_proj.weight: [128x128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/linear_k/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.k_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/linear_q/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.q_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/linear_v/b: [128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/row_attention_block/linear_v/w: [128x128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.v_proj.bias: [128] (float32)
        # transformer.blocks.2.pair_update.row_attn_block.v_proj.weight: [128x128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/linear_v/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.v_proj.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/row_attention_block/linear_v/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.row_attn_block.v_proj.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        ### Pair MLP ###
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/layer_norm/offset: [128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/layer_norm/scale: [128] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.norm.beta: [128] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.norm.gamma: [128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/layer_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.norm.beta"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/layer_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.norm.gamma"),
                transform=Identity(),
            )
        ])
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/linear/b: [256] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/linear/w: [128x256] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.fc1.bias: [256] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.fc1.weight: [256x128] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/linear/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.fc1.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/linear/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.fc1.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/linear_1/b: [128] (float32)
        # alphagenome/transformer_tower/pair_update_block_1/pair_mlp_block/linear_1/w: [256x128] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.fc2.bias: [128] (float32)
        # transformer.blocks.2.pair_update.pair_mlp_block.fc2.weight: [128x256] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/linear_1/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.fc2.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"pair_update_block{jax_idx}/pair_mlp_block/linear_1/w"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_idx}.pair_update.pair_mlp_block.fc2.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
    return mappings


def build_decoder_param_mappings() -> list[MapSpec]:
    """DECODER"""
    mappings: list[MapSpec] = []
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6"]
    torch_indices = ["64", "32", "16", "8", "4", "2", "1"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_idx}/"
        torch_prefix = f"decoder.upres_blocks.bin_size_{torch_idx}."
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/rms_batch_norm/offset: [1x1x1536] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/rms_batch_norm/scale: [1x1x1536] (float32)
        # decoder.upres_blocks.bin_size_2.block1.norm.beta: [1536] (float32)
        # decoder.upres_blocks.bin_size_2.block1.norm.gamma: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.norm.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.norm.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/standardized_conv1_d/bias: [1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/standardized_conv1_d/scale: [1x1x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/standardized_conv1_d/w: [5x1536x1408] (float32)
        # decoder.upres_blocks.bin_size_2.block1.conv.bias: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block1.conv.scale: [1408x1x1] (float32)
        # decoder.upres_blocks.bin_size_2.block1.conv.weight: [1408x1536x5] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/standardized_conv1_d/bias"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.conv.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/standardized_conv1_d/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.conv.scale"),
                transform=Rearrange("() () c -> c () ()"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/standardized_conv1_d/w"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.conv.weight"),
                transform=Rearrange("w c1 c2 -> c2 c1 w"),
            ),
        ])
        # alphagenome/sequence_decoder/up_res_block_1/pointwise_conv_unet_skip/rms_batch_norm/offset: [1x1x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/pointwise_conv_unet_skip/rms_batch_norm/scale: [1x1x1408] (float32)
        # decoder.upres_blocks.bin_size_2.block2.norm.beta: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block2.norm.gamma: [1408] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("pointwise_conv_unet_skip/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=("block2.norm.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("pointwise_conv_unet_skip/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("block2.norm.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/sequence_decoder/up_res_block_1/pointwise_conv_unet_skip/linear/b: [1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/pointwise_conv_unet_skip/linear/w: [1408x1408] (float32)
        # decoder.upres_blocks.bin_size_2.block2.conv.bias: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block2.conv.weight: [1408x1408] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("pointwise_conv_unet_skip/linear/b"),
                torch_prefix=torch_prefix,
                torch_keys=("block2.conv.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("pointwise_conv_unet_skip/linear/w"),
                torch_prefix=torch_prefix,
                torch_keys=("block2.conv.weight"),
                transform=Rearrange("c1 c2 -> c2 c1"),
            ),
        ])
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/rms_batch_norm/offset: [1x1x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/rms_batch_norm/scale: [1x1x1408] (float32)
        # decoder.upres_blocks.bin_size_2.block3.norm.beta: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block3.norm.gamma: [1408] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/rms_batch_norm/offset"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.norm.beta"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/rms_batch_norm/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.norm.gamma"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/standardized_conv1_d/bias: [1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/standardized_conv1_d/scale: [1x1x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/standardized_conv1_d/w: [5x1408x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/residual_scale: [] (float32)
        # decoder.upres_blocks.bin_size_2.block3.conv.bias: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block3.conv.scale: [1408x1x1] (float32)
        # decoder.upres_blocks.bin_size_2.block3.conv.weight: [1408x1408x5] (float32)
        # decoder.upres_blocks.bin_size_2.residual_scale: [1] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/standardized_conv1_d/bias"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.conv.bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/standardized_conv1_d/scale"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.conv.scale"),
                transform=Rearrange("() () c -> c () ()"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/standardized_conv1_d/w"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.conv.weight"),
                transform=Rearrange("w c1 c2 -> c2 c1 w"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("residual_scale"),
                torch_prefix=torch_prefix,
                torch_keys=("residual_scale"),
                transform=Rearrange("-> ()"),
            ),
        ])
    return mappings


def build_output_embedder_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []

    """OUTPUT EMBEDDER T"""
    jax_prefix = "alphagenome/output_embedder/"
    torch_prefix = "embedder_t."
    # alphagenome/output_embedder/linear/b: [3072] (float32)
    # alphagenome/output_embedder/linear/w: [1536x3072] (float32)
    # alphagenome/output_embedder/embed/embeddings: [2x3072] (float32)
    # embedder_t.fc1.bias: [3072] (float32)
    # embedder_t.fc1.weight: [3072x1536] (float32)
    # embedder_t.org.weight: [2x3072] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("linear/b"),
            torch_prefix=torch_prefix,
            torch_keys=("fc1.bias"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("linear/w"),
            torch_prefix=torch_prefix,
            torch_keys=("fc1.weight"),
            transform=Rearrange("c1 c2 -> c2 c1"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("org.weight"),
            transform=Identity(),
        ),
    ])
    # alphagenome/output_embedder/rms_batch_norm/offset: [1x1x3072] (float32)
    # alphagenome/output_embedder/rms_batch_norm/scale: [1x1x3072] (float32)
    # embedder_t.norm.beta: [3072] (float32)
    # embedder_t.norm.gamma: [3072] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("rms_batch_norm/offset"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.beta"),
            transform=Rearrange("() () c -> c"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("rms_batch_norm/scale"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.gamma"),
            transform=Rearrange("() () c -> c"),
        ),
    ])

    """OUTPUT EMBEDDER X"""
    jax_prefix = "alphagenome/output_embedder_1/"
    torch_prefix = "embedder_x."
    # alphagenome/output_embedder_1/linear/b: [1536] (float32)
    # alphagenome/output_embedder_1/linear/w: [768x1536] (float32)
    # alphagenome/output_embedder_1/linear_1/w: [3072x1536] (float32)
    # alphagenome/output_embedder_1/embed/embeddings: [2x1536] (float32)
    # embedder_x.fc1.bias: [1536] (float32)
    # embedder_x.fc1.weight: [1536x768] (float32)
    # embedder_x.fc_skip.weight: [1536x3072] (float32)
    # embedder_x.org.weight: [2x1536] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("linear/b"),
            torch_prefix=torch_prefix,
            torch_keys=("fc1.bias"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("linear/w"),
            torch_prefix=torch_prefix,
            torch_keys=("fc1.weight"),
            transform=Rearrange("c1 c2 -> c2 c1"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("linear_1/w"),
            torch_prefix=torch_prefix,
            torch_keys=("fc_skip.weight"),
            transform=Rearrange("c1 c2 -> c2 c1"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("org.weight"),
            transform=Identity(),
        ),
    ])
    # alphagenome/output_embedder_1/rms_batch_norm/offset: [1x1x1536] (float32)
    # alphagenome/output_embedder_1/rms_batch_norm/scale: [1x1x1536] (float32)
    # embedder_x.norm.beta: [1536] (float32)
    # embedder_x.norm.gamma: [1536] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("rms_batch_norm/offset"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.beta"),
            transform=Rearrange("() () c -> c"),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("rms_batch_norm/scale"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.gamma"),
            transform=Rearrange("() () c -> c"),
        ),
    ])

    """OUTPUT PAIR"""
    jax_prefix = "alphagenome/output_pair/"
    torch_prefix = "embedder_pair."
    # alphagenome/output_pair/embed/embeddings: [2x128] (float32)
    # alphagenome/output_pair/layer_norm/offset: [128] (float32)
    # alphagenome/output_pair/layer_norm/scale: [128] (float32)
    # embedder_pair.embed.weight: [2x128] (float32)
    # embedder_pair.norm.beta: [128] (float32)
    # embedder_pair.norm.gamma: [128] (float32)
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("embed.weight"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("layer_norm/offset"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.beta"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("layer_norm/scale"),
            torch_prefix=torch_prefix,
            torch_keys=("norm.gamma"),
            transform=Identity(),
        ),
    ])
    return mappings


def build_head_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []
    genome_track_heads = {
        "atac": (1, 128),
        "cage": (1, 128),
        "dnase": (1, 128),
        "procap": (1, 128),
        "rna_seq": (1, 128),
        "chip_tf": (128,),
        "chip_histone": (128,),
    }
    for head_name, resolutions in genome_track_heads.items():
        for resolution in resolutions:
            jax_prefix = f"alphagenome/head/{head_name}/resolution_{resolution}/"
            torch_prefix = f"_heads.{head_name}."
            # alphagenome/head/atac/resolution_1/multi_organism_linear/b: [2x256] (float32)
            # alphagenome/head/atac/resolution_1/multi_organism_linear/w: [2x1536x256] (float32)
            # alphagenome/head/atac/resolution_1/learnt_scale: [2x256] (float32)
            # _heads.atac.multiorg_linear.1.bias: [2x256] (float32)
            # _heads.atac.multiorg_linear.1.weight: [2x1536x256] (float32)
            # _heads.atac.residual_scales.1: [2x256] (float32)
            mappings.extend([
                MapSpec(
                    jax_prefix=jax_prefix,
                    jax_keys=("multi_organism_linear/b"),
                    torch_prefix=torch_prefix,
                    torch_keys=(f"multiorg_linear.{resolution}.bias"),
                    transform=Identity(),
                ),
                MapSpec(
                    jax_prefix=jax_prefix,
                    jax_keys=("multi_organism_linear/w"),
                    torch_prefix=torch_prefix,
                    torch_keys=(f"multiorg_linear.{resolution}.weight"),
                    transform=Identity(),
                ),
                MapSpec(
                    jax_prefix=jax_prefix,
                    jax_keys=("learnt_scale"),
                    torch_prefix=torch_prefix,
                    torch_keys=(f"residual_scales.{resolution}"),
                    transform=Identity(),
                ),
            ])

    for head_name in (
        "contact_maps",
        "splice_sites_classification",
        "splice_sites_usage",
        "splice_sites_junction",
    ):
        jax_prefix = f"alphagenome/head/{head_name}/multi_organism_linear/"
        torch_prefix = f"_heads.{head_name}.multiorg_linear."
        # alphagenome/head/contact_maps/multi_organism_linear/b: [2x28] (float32)
        # alphagenome/head/contact_maps/multi_organism_linear/w: [2x128x28] (float32)
        # _heads.contact_maps.multiorg_linear.bias: [2x28] (float32)
        # _heads.contact_maps.multiorg_linear.weight: [2x128x28] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("b"),
                torch_prefix=torch_prefix,
                torch_keys=("bias"),
                transform=Identity(),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("w"),
                torch_prefix=torch_prefix,
                torch_keys=("weight"),
                transform=Identity(),
            ),
        ])

    # alphagenome/head/splice_sites_junction/neg_acceptor_logits/embeddings: [2x563712] (float32)
    # alphagenome/head/splice_sites_junction/neg_donor_logits/embeddings: [2x563712] (float32)
    # alphagenome/head/splice_sites_junction/pos_acceptor_logits/embeddings: [2x563712] (float32)
    # alphagenome/head/splice_sites_junction/pos_donor_logits/embeddings: [2x563712] (float32)
    # _heads.splice_sites_junction.neg_acceptor_logits_embeddings: [2x2x367x768] (float32)
    # _heads.splice_sites_junction.neg_donor_logits_embeddings: [2x2x367x768] (float32)
    # _heads.splice_sites_junction.pos_acceptor_logits_embeddings: [2x2x367x768] (float32)
    # _heads.splice_sites_junction.pos_donor_logits_embeddings: [2x2x367x768] (float32)
    jax_prefix = "alphagenome/head/splice_sites_junction/"
    torch_prefix = "_heads.splice_sites_junction."
    for embedding_name in (
        "neg_acceptor_logits",
        "neg_donor_logits",
        "pos_acceptor_logits",
        "pos_donor_logits",
    ):
        mappings.append(
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"{embedding_name}/embeddings"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{embedding_name}_embeddings"),
                transform=Rearrange(
                    "org (scale_offset tissue channel) -> org scale_offset tissue channel",
                    scale_offset=2,
                    tissue=367,
                    channel=768,
                ),
            )
        )

    return mappings


def build_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []
    mappings.extend(build_etc_mappings())
    mappings.extend(build_head_param_mappings())
    mappings.extend(build_output_embedder_param_mappings())
    mappings.extend(build_encoder_mappings())
    mappings.extend(build_decoder_param_mappings())
    mappings.extend(build_transformer_param_mappings())
    return mappings



def build_state_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []

    """ENCODER"""
    # alphagenome/sequence_encoder/dna_embedder/conv_block/rms_batch_norm/var_ema: [1x1x768] (float32)
    # encoder.downres_blocks.bin_size_1.conv_block.norm.var_EMA: [768] (float32)
    mappings.append(
        MapSpec(
            jax_keys=("alphagenome/sequence_encoder/dna_embedder/conv_block/rms_batch_norm/var_ema"),
            torch_keys=("encoder.downres_blocks.bin_size_1.conv_block.norm.var_EMA"),
            transform=Rearrange("() () c -> c"),
        )
    )

    jax_indices = ["_0", "_1", "_2", "_3", "_4", "_5"]
    torch_indices = ["2", "4", "8", "16", "32", "64"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_encoder/downres_block{jax_idx}/"
        torch_prefix = f"encoder.downres_blocks.bin_size_{torch_idx}."
        # alphagenome/sequence_encoder/downres_block_0/conv_block/rms_batch_norm/var_ema: [1x1x768] (float32)
        # alphagenome/sequence_encoder/downres_block_0/conv_block_1/rms_batch_norm/var_ema: [1x1x896] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block1.norm.var_EMA: [768] (float32)
        # encoder.downres_blocks.bin_size_2.conv_block2.norm.var_EMA: [896] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block1.norm.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_block_1/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("conv_block2.norm.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
        ])

    """TRANSFORMER"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]
    torch_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = "alphagenome/transformer_tower/"
        torch_prefix = f"transformer.blocks.{torch_idx}."
        # alphagenome/transformer_tower/attention_bias_block_1/rms_batch_norm/var_ema: [1x1x1x128] (float32)
        # transformer.blocks.1.attn_bias.bn.var_EMA: [128] (float32)
        mappings.append(
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"attention_bias_block{jax_idx}/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("attn_bias.bn.var_EMA"),
                transform=Rearrange("() () () c -> c"),
            )
        )
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm/var_ema: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mha_block_1/rms_batch_norm_1/var_ema: [1x1x1536] (float32)
        # transformer.blocks.1.mha.bn1.var_EMA: [1536] (float32)
        # transformer.blocks.1.mha.bn2.var_EMA: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("mha.bn1.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mha_block{jax_idx}/rms_batch_norm_1/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("mha.bn2.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
        ])
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm/var_ema: [1x1x1536] (float32)
        # alphagenome/transformer_tower/mlp_block_1/rms_batch_norm_1/var_ema: [1x1x1536] (float32)
        # transformer.blocks.1.mlp.bn1.var_EMA: [1536] (float32)
        # transformer.blocks.1.mlp.bn2.var_EMA: [1536] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("mlp.bn1.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"mlp_block{jax_idx}/rms_batch_norm_1/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("mlp.bn2.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
        ])

    """DECODER"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6"]
    torch_indices = ["64", "32", "16", "8", "4", "2", "1"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_idx}/"
        torch_prefix = f"decoder.upres_blocks.bin_size_{torch_idx}."
        # alphagenome/sequence_decoder/up_res_block_1/conv_in/rms_batch_norm/var_ema: [1x1x1536] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/pointwise_conv_unet_skip/rms_batch_norm/var_ema: [1x1x1408] (float32)
        # alphagenome/sequence_decoder/up_res_block_1/conv_out/rms_batch_norm/var_ema: [1x1x1408] (float32)
        # decoder.upres_blocks.bin_size_2.block1.norm.var_EMA: [1536] (float32)
        # decoder.upres_blocks.bin_size_2.block2.norm.var_EMA: [1408] (float32)
        # decoder.upres_blocks.bin_size_2.block3.norm.var_EMA: [1408] (float32)
        mappings.extend([
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_in/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("block1.norm.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("pointwise_conv_unet_skip/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("block2.norm.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("conv_out/rms_batch_norm/var_ema"),
                torch_prefix=torch_prefix,
                torch_keys=("block3.norm.var_EMA"),
                transform=Rearrange("() () c -> c"),
            ),
        ])

    """OUTPUT EMBEDDERS"""
    # alphagenome/output_embedder/rms_batch_norm/var_ema: [1x1x3072] (float32)
    # alphagenome/output_embedder_1/rms_batch_norm/var_ema: [1x1x1536] (float32)
    # embedder_t.norm.var_EMA: [3072] (float32)
    # embedder_x.norm.var_EMA: [1536] (float32)
    mappings.extend([
        MapSpec(
            jax_keys=("alphagenome/output_embedder/rms_batch_norm/var_ema"),
            torch_keys=("embedder_t.norm.var_EMA"),
            transform=Rearrange("() () c -> c"),
        ),
        MapSpec(
            jax_keys=("alphagenome/output_embedder_1/rms_batch_norm/var_ema"),
            torch_keys=("embedder_x.norm.var_EMA"),
            transform=Rearrange("() () c -> c"),
        ),
    ])

    return mappings



"""
CONVERSION
"""
# Create sets of mapped keys
def _mapped_keys(mappings: Sequence[MapSpec]) -> tuple[set[str], set[str]]:
    jax_keys: set[str] = set()
    torch_keys: set[str] = set()
    for mapping in mappings:
        jax_keys.update(mapping.jax_full_keys)
        torch_keys.update(mapping.torch_full_keys)
    return jax_keys, torch_keys

def mapped_param_keys() -> tuple[set[str], set[str]]:
    return _mapped_keys(build_param_mappings())

def mapped_state_keys() -> tuple[set[str], set[str]]:
    return _mapped_keys(build_state_mappings())

def mapped_keys() -> tuple[set[str], set[str]]:
    param_jax_keys, param_torch_keys = _mapped_keys(build_param_mappings())
    state_jax_keys, state_torch_keys = _mapped_keys(build_state_mappings())
    jax_keys = param_jax_keys | state_jax_keys
    torch_keys = param_torch_keys | state_torch_keys
    return jax_keys, torch_keys

# Convert
def convert_flat_jax_params_to_torch(
    flat_params: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Convert flattened JAX params into a PyTorch state_dict fragment."""
    converted: dict[str, Tensor] = {}
    for mapping in build_param_mappings():
        converted.update(mapping.convert(flat_params))
    return converted

def convert_flat_jax_state_to_torch(
    flat_state: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Convert flattened JAX state into a PyTorch state_dict fragment."""
    converted: dict[str, Tensor] = {}
    for mapping in build_state_mappings():
        converted.update(mapping.convert(flat_state))
    return converted

def convert_flat_jax_to_torch(
    flat_params: Mapping[str, Any],
    flat_state: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Convert flattened JAX params/state into a PyTorch state_dict fragment."""
    converted = convert_flat_jax_params_to_torch(flat_params)
    converted.update(convert_flat_jax_state_to_torch(flat_state))
    return converted



"""
CHECK CONVERSION
"""
# Pattern match expected checkpoint/model differences
EXPECTED_JAX_CHECKPOINT_ONLY_PARAM_PATTERNS: tuple[str, ...] = ()
EXPECTED_TORCH_MODEL_ONLY_PARAM_PATTERNS: tuple[str, ...] = (
    "_heads.masked_language_modeling.*",
)
EXPECTED_JAX_CHECKPOINT_ONLY_STATE_PATTERNS: tuple[str, ...] = ()
EXPECTED_TORCH_MODEL_ONLY_STATE_PATTERNS: tuple[str, ...] = (
    "_heads.*._track_means",
)

def _find_keys(keys: Sequence[str], patterns: Sequence[str]) -> set[str]:
    return {
        key
        for key in keys
        if any(fnmatch(key, pattern) for pattern in patterns)
    }

def expected_jax_only_param_keys(jax_params: Mapping[str, Any]) -> set[str]:
    return _find_keys(tuple(jax_params), EXPECTED_JAX_CHECKPOINT_ONLY_PARAM_PATTERNS)

def expected_torch_only_param_keys(torch_params: Mapping[str, Tensor]) -> set[str]:
    return _find_keys(tuple(torch_params), EXPECTED_TORCH_MODEL_ONLY_PARAM_PATTERNS)

def expected_jax_only_state_keys(jax_state: Mapping[str, Any]) -> set[str]:
    return _find_keys(tuple(jax_state), EXPECTED_JAX_CHECKPOINT_ONLY_STATE_PATTERNS)

def expected_torch_only_state_keys(torch_state: Mapping[str, Tensor]) -> set[str]:
    return _find_keys(tuple(torch_state), EXPECTED_TORCH_MODEL_ONLY_STATE_PATTERNS)

def expected_jax_only_keys(jax_params: Mapping[str, Any], jax_state: Mapping[str, Any]) -> set[str]:
    return expected_jax_only_param_keys(jax_params) | expected_jax_only_state_keys(jax_state)

def expected_torch_only_keys(torch_params: Mapping[str, Tensor], torch_state: Mapping[str, Tensor]) -> set[str]:
    return expected_torch_only_param_keys(torch_params) | expected_torch_only_state_keys(torch_state)

# Find unmapped keys
def unmapped_jax_param_keys(jax_params: Mapping[str, Any]) -> set[str]:
    """Return JAX param keys that are neither mapped nor intentionally JAX-only."""
    mapped_jax_param_keys, _ = mapped_param_keys()
    return set(jax_params) - mapped_jax_param_keys - expected_jax_only_param_keys(jax_params)

def unmapped_torch_param_keys(torch_params: Mapping[str, Tensor]) -> set[str]:
    """Return torch param keys that are neither mapped nor intentionally torch-only."""
    _, mapped_torch_param_keys = mapped_param_keys()
    return set(torch_params) - mapped_torch_param_keys - expected_torch_only_param_keys(torch_params)

def unmapped_jax_state_keys(jax_state: Mapping[str, Any]) -> set[str]:
    """Return JAX state keys that are neither mapped nor intentionally JAX-only."""
    mapped_jax_state_keys, _ = mapped_state_keys()
    return set(jax_state) - mapped_jax_state_keys - expected_jax_only_state_keys(jax_state)

def unmapped_torch_state_keys(torch_state: Mapping[str, Tensor]) -> set[str]:
    """Return torch state keys that are neither mapped nor intentionally torch-only."""
    _, mapped_torch_state_keys = mapped_state_keys()
    return set(torch_state) - mapped_torch_state_keys - expected_torch_only_state_keys(torch_state)

# Show differences
def key_differences(
    left_params: Mapping[str, Any],
    left_state: Mapping[str, Any],
    right_params: Mapping[str, Any],
    right_state: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return key differences between two param/state dictionaries."""
    left_param_keys = set(left_params)
    left_state_keys = set(left_state)
    right_param_keys = set(right_params)
    right_state_keys = set(right_state)
    return {
        "left_only_params": left_param_keys - right_param_keys,
        "right_only_params": right_param_keys - left_param_keys,
        "left_only_state": left_state_keys - right_state_keys,
        "right_only_state": right_state_keys - left_state_keys,
    }

def key_differences_report(
    left_params: Mapping[str, Any],
    left_state: Mapping[str, Any],
    right_params: Mapping[str, Any],
    right_state: Mapping[str, Any],
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> str:
    """Return a readable report of key differences between two param/state dictionaries."""

    def format_keys(keys: set[str]) -> str:
        if not keys:
            return "    <none>"
        return "\n".join(f"    {key}" for key in sorted(keys))

    diffs = key_differences(left_params, left_state, right_params, right_state)
    lines = [
        f"Key Differences: {left_name} vs {right_name}",
        "Key counts:",
        f"  {left_name}: params {len(left_params)}, state {len(left_state)}",
        f"  {right_name}: params {len(right_params)}, state {len(right_state)}",
        "",
        "Param differences:",
        f"  {left_name}-only ({len(diffs['left_only_params'])}):",
        format_keys(diffs["left_only_params"]),
        f"  {right_name}-only ({len(diffs['right_only_params'])}):",
        format_keys(diffs["right_only_params"]),
        "",
        "State differences:",
        f"  {left_name}-only ({len(diffs['left_only_state'])}):",
        format_keys(diffs["left_only_state"]),
        f"  {right_name}-only ({len(diffs['right_only_state'])}):",
        format_keys(diffs["right_only_state"]),
    ]
    return "\n".join(lines)

def mapping_differences_report(
    jax_params: Mapping[str, Any],
    jax_state: Mapping[str, Any],
    torch_params: Mapping[str, Tensor],
    torch_state: Mapping[str, Tensor],
) -> str:
    """Return reports comparing checkpoint keys to their same-namespace mapping keys."""
    mapped_jax_param_keys, mapped_torch_param_keys = mapped_param_keys()
    mapped_jax_state_keys, mapped_torch_state_keys = mapped_state_keys()
    mapped_jax_params = dict.fromkeys(mapped_jax_param_keys)
    mapped_jax_state = dict.fromkeys(mapped_jax_state_keys)
    mapped_torch_params = dict.fromkeys(mapped_torch_param_keys)
    mapped_torch_state = dict.fromkeys(mapped_torch_state_keys)
    return "\n\n".join(
        [
            "\n".join(
                [
                    "Mapped key counts:",
                    f"  JAX: params {len(mapped_jax_params)}, state {len(mapped_jax_state)}",
                    f"  torch: params {len(mapped_torch_params)}, state {len(mapped_torch_state)}",
                ]
            ),
            key_differences_report(
                jax_params,
                jax_state,
                mapped_jax_params,
                mapped_jax_state,
                left_name="checkpoint JAX",
                right_name="mapped JAX",
            ),
            key_differences_report(
                torch_params,
                torch_state,
                mapped_torch_params,
                mapped_torch_state,
                left_name="checkpoint torch",
                right_name="mapped torch",
            ),
        ]
    )
