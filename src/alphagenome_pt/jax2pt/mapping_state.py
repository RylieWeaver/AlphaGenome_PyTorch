"""
JAX checkpoint to current alphagenome_pt state_dict mapping.
- It's made fairly explicitly to be easy to read and modify, at the cost of some verbosity and repetition.
"""

from __future__ import annotations

# External
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Mapping, Sequence
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from einops import rearrange

# Internal
from .utils import flatten_nested_dict, jax_to_torch_tensor



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
REUSABLE MODULE MAPPINGS:
"""
def build_linear_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    with_bias: bool = True,
    weight_transform: TransformFn = Rearrange("c1 c2 -> c2 c1"),
) -> list[MapSpec]:
    mappings: list[MapSpec] = []
    if with_bias:
        mappings.append(
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=(f"{jax_name}/b"),
                torch_prefix=torch_prefix,
                torch_keys=(f"{torch_name}.bias"),
                transform=Identity(),
            )
        )
    mappings.append(
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/w"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.weight"),
            transform=weight_transform,
        )
    )
    return mappings


def build_multi_organism_linear_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
) -> list[MapSpec]:
    jax_bias_key = f"{jax_name}/b" if jax_name else "b"
    jax_weight_key = f"{jax_name}/w" if jax_name else "w"
    torch_bias_key = f"{torch_name}.bias" if torch_name else "bias"
    torch_weight_key = f"{torch_name}.weight" if torch_name else "weight"
    return [
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(jax_bias_key),
            torch_prefix=torch_prefix,
            torch_keys=(torch_bias_key),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(jax_weight_key),
            torch_prefix=torch_prefix,
            torch_keys=(torch_weight_key),
            transform=Identity(),
        ),
    ]


def build_norm_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    torch_offset_name: str,
    torch_scale_name: str,
    transform: TransformFn = Identity(),
) -> list[MapSpec]:
    return [
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/offset"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.{torch_offset_name}"),
            transform=transform,
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/scale"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.{torch_scale_name}"),
            transform=transform,
        ),
    ]


def build_batch_norm_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    transform: TransformFn = Rearrange("() () c -> c"),
) -> list[MapSpec]:
    return build_norm_mappings(
        jax_prefix=jax_prefix,
        torch_prefix=torch_prefix,
        jax_name=jax_name,
        torch_name=torch_name,
        torch_offset_name="offset",
        torch_scale_name="scale",
        transform=transform,
    )


def build_layer_norm_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    transform: TransformFn = Identity(),
) -> list[MapSpec]:
    return build_norm_mappings(
        jax_prefix=jax_prefix,
        torch_prefix=torch_prefix,
        jax_name=jax_name,
        torch_name=torch_name,
        torch_offset_name="offset",
        torch_scale_name="scale",
        transform=transform,
    )


def build_batch_norm_state_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    transform: TransformFn = Rearrange("() () c -> c"),
) -> list[MapSpec]:
    return [
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/var_ema"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.var_EMA"),
            transform=transform,
        )
    ]


def _build_conv1d_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
    jax_bias_name: str = "b",
    weight_transform: TransformFn = Rearrange("w c1 c2 -> c2 c1 w"),
) -> list[MapSpec]:
    return [
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/{jax_bias_name}"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.bias"),
            transform=Identity(),
        ),
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/w"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.weight"),
            transform=weight_transform,
        ),
    ]


def build_standardized_conv_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    jax_name: str,
    torch_name: str,
) -> list[MapSpec]:
    conv_mappings = _build_conv1d_mappings(
        jax_prefix,
        torch_prefix,
        jax_name=jax_name,
        torch_name=torch_name,
        jax_bias_name="bias",
    )
    return [
        conv_mappings[0],
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=(f"{jax_name}/scale"),
            torch_prefix=torch_prefix,
            torch_keys=(f"{torch_name}.scale"),
            transform=Rearrange("() () c -> c () ()"),
        ),
        conv_mappings[1],
    ]


def build_conv_mappings(
    jax_prefix: str,
    torch_prefix: str,
    *,
    width: int | None = None,
    standardized: bool = True,
) -> list[MapSpec]:
    mappings: list[MapSpec] = []
    mappings.extend(
        build_batch_norm_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )
    if width == 1:
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name="linear",
                torch_name="conv",
            )
        )
    elif standardized:
        mappings.extend(
            build_standardized_conv_mappings(
                jax_prefix,
                torch_prefix,
                jax_name="standardized_conv1_d",
                torch_name="conv",
            )
        )
    else:
        mappings.extend(
            _build_conv1d_mappings(
                jax_prefix,
                torch_prefix,
                jax_name="conv1_d",
                torch_name="conv",
            )
        )
    return mappings



"""
TOP LEVEL MAPPINGS:
"""
def build_etc_mappings() -> list[MapSpec]:
    mappings = []

    """Org Embedder"""
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
    torch_prefix = f"sequence_encoder.downres_blocks.bin_size_1."
    mappings.extend(
        _build_conv1d_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="conv1_d",
            torch_name="conv1_d",
        )
    )
    mappings.extend(build_conv_mappings(f"{jax_prefix}conv_block/", f"{torch_prefix}conv_block."))

    """ENCODERS / BIN SIZES > 2"""
    jax_indices = ["_0", "_1", "_2", "_3", "_4", "_5"]
    torch_indices = ["2", "4", "8", "16", "32", "64"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_encoder/downres_block{jax_idx}/"
        torch_prefix = f"sequence_encoder.downres_blocks.bin_size_{torch_idx}."
        mappings.extend(build_conv_mappings(f"{jax_prefix}conv_block/", f"{torch_prefix}conv_block1."))
        mappings.extend(build_conv_mappings(f"{jax_prefix}conv_block_1/", f"{torch_prefix}conv_block2."))
    return mappings


def build_transformer_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []
    jax_prefix = f"alphagenome/transformer_tower/"
    torch_prefix = f"transformer_tower.blocks."

    """Transformer Blocks"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]
    torch_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        ### MHA ###
        mappings.extend(
            build_batch_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mha_block{jax_idx}/rms_batch_norm",
                torch_name=f"{torch_idx}.mha.bn1",
            )
        )
        for qkv in ("q", "k", "v"):
            mappings.extend(
                build_linear_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name=f"mha_block{jax_idx}/{qkv}_layer",
                    torch_name=f"{torch_idx}.mha.{qkv}_layer",
                    with_bias=False,
                )
            )
            mappings.extend(
                build_norm_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name=f"mha_block{jax_idx}/norm_{qkv}",
                    torch_name=f"{torch_idx}.mha.norm_{qkv}",
                    torch_offset_name="offset",
                    torch_scale_name="scale",
                )
            )
        mappings.extend(
            build_batch_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mha_block{jax_idx}/rms_batch_norm_1",
                torch_name=f"{torch_idx}.mha.bn2",
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mha_block{jax_idx}/linear_embedding",
                torch_name=f"{torch_idx}.mha.linear_embedding",
            )
        )
        ### Fully Connected ###
        mappings.extend(
            build_batch_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/rms_batch_norm",
                torch_name=f"{torch_idx}.mlp.bn1",
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/linear",
                torch_name=f"{torch_idx}.mlp.fc1",
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/linear_1",
                torch_name=f"{torch_idx}.mlp.fc2",
            )
        )
        mappings.extend(
            build_batch_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/rms_batch_norm_1",
                torch_name=f"{torch_idx}.mlp.bn2",
            )
        )
        ### Attn Bias ###
        mappings.extend(
            build_batch_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"attention_bias_block{jax_idx}/rms_batch_norm",
                torch_name=f"{torch_idx}.attn_bias.bn",
                transform=Rearrange("() () () c -> c"),
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"attention_bias_block{jax_idx}/linear",
                torch_name=f"{torch_idx}.attn_bias.fc",
                with_bias=False,
                weight_transform=Rearrange("h c -> c h"),
            )
        )
    
    """Pair Update Blocks"""
    jax_indices = ["", "_1", "_2", "_3", "_4"]
    torch_indices = ["0", "2", "4", "6", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        ### Seq2Pair ###
        mappings.extend(
            build_layer_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/sequence_to_pair_block/norm_seq2pair",
                torch_name=f"{torch_idx}.pair_update.sequence_to_pair_block.norm",
            )
        )
        for qk in ("q", "k"):
            mappings.extend(
                build_linear_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name=f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_{qk}",
                    torch_name=f"{torch_idx}.pair_update.sequence_to_pair_block.linear_{qk}",
                    with_bias=False,
                )
            )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pos_features",
                torch_name=f"{torch_idx}.pair_update.sequence_to_pair_block.linear_pos_features",
            )
        )
        for qk in ("q", "k"):
            mappings.append(
                MapSpec(
                    jax_prefix=jax_prefix,
                    jax_keys=(f"pair_update_block{jax_idx}/sequence_to_pair_block/{qk}_r_bias"),
                    torch_prefix=torch_prefix,
                    torch_keys=(f"{torch_idx}.pair_update.sequence_to_pair_block.{qk}_r_bias"),
                    transform=Identity(),
                )
            )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_pair",
                torch_name=f"{torch_idx}.pair_update.sequence_to_pair_block.linear_pair",
            )
        )
        for qk in ("q", "k"):
            mappings.extend(
                build_linear_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name=f"pair_update_block{jax_idx}/sequence_to_pair_block/linear_y_{qk}",
                    torch_name=f"{torch_idx}.pair_update.sequence_to_pair_block.linear_y_{qk}",
                    with_bias=False,
                )
            )
        ### Row Attention ###
        mappings.extend(
            build_layer_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/row_attention_block/layer_norm",
                torch_name=f"{torch_idx}.pair_update.row_attention_block.norm",
            )
        )
        for qk in ("k", "q"):
            mappings.extend(
                build_linear_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name=f"pair_update_block{jax_idx}/row_attention_block/linear_{qk}",
                    torch_name=f"{torch_idx}.pair_update.row_attention_block.linear_{qk}",
                    with_bias=False,
                )
            )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/row_attention_block/linear_v",
                torch_name=f"{torch_idx}.pair_update.row_attention_block.linear_v",
            )
        )
        ### Pair MLP ###
        mappings.extend(
            build_layer_norm_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/pair_mlp_block/layer_norm",
                torch_name=f"{torch_idx}.pair_update.pair_mlp_block.norm",
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/pair_mlp_block/linear",
                torch_name=f"{torch_idx}.pair_update.pair_mlp_block.fc1",
            )
        )
        mappings.extend(
            build_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"pair_update_block{jax_idx}/pair_mlp_block/linear_1",
                torch_name=f"{torch_idx}.pair_update.pair_mlp_block.fc2",
            )
        )
    return mappings


def build_decoder_param_mappings() -> list[MapSpec]:
    """DECODER"""
    mappings: list[MapSpec] = []
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6"]
    torch_indices = ["64", "32", "16", "8", "4", "2", "1"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_idx}/"
        torch_prefix = f"sequence_decoder.upres_blocks.bin_size_{torch_idx}."
        mappings.extend(build_conv_mappings(f"{jax_prefix}conv_in/", f"{torch_prefix}conv_in."))
        mappings.extend(
            build_conv_mappings(
                f"{jax_prefix}pointwise_conv_unet_skip/",
                f"{torch_prefix}pointwise_conv_unet_skip.",
                width=1,
            )
        )
        mappings.extend(build_conv_mappings(f"{jax_prefix}conv_out/", f"{torch_prefix}conv_out."))
        mappings.append(
            MapSpec(
                jax_prefix=jax_prefix,
                jax_keys=("residual_scale"),
                torch_prefix=torch_prefix,
                torch_keys=("residual_scale"),
                transform=Rearrange("-> ()"),
            ),
        )
    return mappings


def build_output_embedder_param_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []

    """OUTPUT EMBEDDER T"""
    jax_prefix = "alphagenome/output_embedder/"
    torch_prefix = "output_t."
    mappings.extend(
        build_linear_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="linear",
            torch_name="fc1",
        )
    )
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("org.weight"),
            transform=Identity(),
        ),
    ])
    mappings.extend(
        build_batch_norm_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )

    """OUTPUT EMBEDDER X"""
    jax_prefix = "alphagenome/output_embedder_1/"
    torch_prefix = "output_x."
    mappings.extend(
        build_linear_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="linear",
            torch_name="fc1",
        )
    )
    mappings.extend(
        build_linear_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="linear_1",
            torch_name="fc_skip",
            with_bias=False,
        )
    )
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("org.weight"),
            transform=Identity(),
        ),
    ])
    mappings.extend(
        build_batch_norm_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )

    """OUTPUT PAIR"""
    jax_prefix = "alphagenome/output_pair/"
    torch_prefix = "output_pair."
    mappings.extend([
        MapSpec(
            jax_prefix=jax_prefix,
            jax_keys=("embed/embeddings"),
            torch_prefix=torch_prefix,
            torch_keys=("organism_embed.weight"),
            transform=Identity(),
        ),
    ])
    mappings.extend(
        build_layer_norm_mappings(
            jax_prefix,
            torch_prefix,
            jax_name="layer_norm",
            torch_name="norm",
        )
    )
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
            mappings.extend(
                build_multi_organism_linear_mappings(
                    jax_prefix,
                    torch_prefix,
                    jax_name="multi_organism_linear",
                    torch_name=f"multiorg_linear.{resolution}",
                )
            )
            mappings.extend([
                MapSpec(
                    jax_prefix=jax_prefix,
                    jax_keys=("learnt_scale"),
                    torch_prefix=torch_prefix,
                    torch_keys=(f"residual_scales.{resolution}"),
                    transform=Identity(),
                ),
            ])

    other_track_heads = (
        "contact_maps",
        "splice_sites_classification",
        "splice_sites_usage",
        "splice_sites_junction",
    )
    for head_name in other_track_heads:
        jax_prefix = f"alphagenome/head/{head_name}/multi_organism_linear/"
        torch_prefix = f"_heads.{head_name}.multiorg_linear."
        mappings.extend(
            build_multi_organism_linear_mappings(
                jax_prefix,
                torch_prefix,
                jax_name="",
                torch_name="",
            )
        )

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
    mappings.extend(build_encoder_mappings())
    mappings.extend(build_transformer_param_mappings())
    mappings.extend(build_decoder_param_mappings())
    mappings.extend(build_output_embedder_param_mappings())
    mappings.extend(build_head_param_mappings())
    return mappings


def build_state_mappings() -> list[MapSpec]:
    mappings: list[MapSpec] = []

    """ENCODER"""
    mappings.extend(
        build_batch_norm_state_mappings(
            "alphagenome/sequence_encoder/dna_embedder/conv_block/",
            "sequence_encoder.downres_blocks.bin_size_1.conv_block.",
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )

    jax_indices = ["_0", "_1", "_2", "_3", "_4", "_5"]
    torch_indices = ["2", "4", "8", "16", "32", "64"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_encoder/downres_block{jax_idx}/"
        torch_prefix = f"sequence_encoder.downres_blocks.bin_size_{torch_idx}."
        mappings.extend(
            build_batch_norm_state_mappings(
                f"{jax_prefix}conv_block/",
                f"{torch_prefix}conv_block1.",
                jax_name="rms_batch_norm",
                torch_name="norm",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                f"{jax_prefix}conv_block_1/",
                f"{torch_prefix}conv_block2.",
                jax_name="rms_batch_norm",
                torch_name="norm",
            )
        )

    """TRANSFORMER"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]
    torch_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = "alphagenome/transformer_tower/"
        torch_prefix = f"transformer_tower.blocks.{torch_idx}."
        mappings.extend(
            build_batch_norm_state_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"attention_bias_block{jax_idx}/rms_batch_norm",
                torch_name="attn_bias.bn",
                transform=Rearrange("() () () c -> c"),
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mha_block{jax_idx}/rms_batch_norm",
                torch_name="mha.bn1",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mha_block{jax_idx}/rms_batch_norm_1",
                torch_name="mha.bn2",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/rms_batch_norm",
                torch_name="mlp.bn1",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                jax_prefix,
                torch_prefix,
                jax_name=f"mlp_block{jax_idx}/rms_batch_norm_1",
                torch_name="mlp.bn2",
            )
        )

    """DECODER"""
    jax_indices = ["", "_1", "_2", "_3", "_4", "_5", "_6"]
    torch_indices = ["64", "32", "16", "8", "4", "2", "1"]
    for jax_idx, torch_idx in zip(jax_indices, torch_indices):
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_idx}/"
        torch_prefix = f"sequence_decoder.upres_blocks.bin_size_{torch_idx}."
        mappings.extend(
            build_batch_norm_state_mappings(
                f"{jax_prefix}conv_in/",
                f"{torch_prefix}conv_in.",
                jax_name="rms_batch_norm",
                torch_name="norm",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                f"{jax_prefix}pointwise_conv_unet_skip/",
                f"{torch_prefix}pointwise_conv_unet_skip.",
                jax_name="rms_batch_norm",
                torch_name="norm",
            )
        )
        mappings.extend(
            build_batch_norm_state_mappings(
                f"{jax_prefix}conv_out/",
                f"{torch_prefix}conv_out.",
                jax_name="rms_batch_norm",
                torch_name="norm",
            )
        )

    """OUTPUT EMBEDDERS"""
    mappings.extend(
        build_batch_norm_state_mappings(
            "alphagenome/output_embedder/",
            "output_t.",
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )
    mappings.extend(
        build_batch_norm_state_mappings(
            "alphagenome/output_embedder_1/",
            "output_x.",
            jax_name="rms_batch_norm",
            torch_name="norm",
        )
    )

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
    state_jax_keys, state_torch_keys = mapped_state_keys()
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


def convert_state(
    params: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Convert nested JAX params/state into a PyTorch state_dict fragment."""
    return convert_flat_jax_to_torch(
        flatten_nested_dict(params),
        flatten_nested_dict(state),
    )



"""
CHECK CONVERSION
"""
# Pattern match expected checkpoint/model differences
EXPECTED_JAX_CHECKPOINT_ONLY_PARAM_PATTERNS: tuple[str, ...] = ()
EXPECTED_TORCH_MODEL_ONLY_PARAM_PATTERNS: tuple[str, ...] = (
    "_heads.masked_language_modeling.*",
)
EXPECTED_JAX_CHECKPOINT_ONLY_STATE_PATTERNS: tuple[str, ...] = ()
EXPECTED_TORCH_MODEL_ONLY_STATE_PATTERNS: tuple[str, ...] = ()

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
