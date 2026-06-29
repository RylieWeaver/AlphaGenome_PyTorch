"""Checkpoint conversion from DeepMind to PyTorch model, including params/state, and metadata."""

# Core checkpoint behavior:
# - Load converted DeepMind AlphaGenome PyTorch state into a PyTorch model so
#   users do not need JAX for normal PyTorch use.
# - Download state and metadata from Hugging Face Hub when local files are absent.
# - Define the DeepMind AlphaGenome config hyperparameters used by this package.
#
# Hugging Face loader naming:
# - local_dir: optional directory for visible downloaded files. If None, use
#   Hugging Face's disk cache directly.
# - repo_id: Hugging Face Hub model repo ID.
# - repo_dir: directory inside the Hub repo. Defaults to v{package-version}.
# - fold: DeepMind AlphaGenome checkpoint fold for state loading.
# - *_filename: metadata file names inside repo_dir.
# - token: Hugging Face auth token. If None, huggingface_hub uses env auth.
# - force_download: whether to force a fresh Hub download.
#
# Organism and head loading defaults:
# - Load organism tensors and all compatible heads by default.
# - Skip organism tensors with organisms=False.
# - Skip all heads with heads=False.
# - Prefix-load compatible organism, track, and tissue axes by default.
# - Load selected heads or explicit source-to-target index maps with head_specs.
# - Raise on shape mismatches outside those loadable axes.

from __future__ import annotations

# External
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Literal, NamedTuple
import warnings
from huggingface_hub import hf_hub_download
import torch
from torch import nn

# Internal
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, package_version



##### CONSTANT VARIABLES AND STRICT STRUCTURES #####
FoldName = Literal["all_folds", "fold_0", "fold_1", "fold_2", "fold_3"]
DEFAULT_ALPHAGENOME_REPO_ID = "RylieWeaver/alphagenome-pytorch"
DEFAULT_FOLD = "all_folds"
FOLD_NAMES = (DEFAULT_FOLD, "fold_0", "fold_1", "fold_2", "fold_3")
DEFAULT_CONVERTED_METADATA_FILENAME = "alphagenome_metadata.json"
DEFAULT_RAW_METADATA_FILENAME = "alphagenome_metadata_raw.json"
DEFAULT_METADATA_SUMMARY_FILENAME = "alphagenome_metadata_summary.json"

def fold_filename(fold_name: str) -> str:
    return f"alphagenome_{fold_name}.pt"


# NOTE: Pay special attention to update these 
# keys if/when the module names change
ORGANISM_KEYS = {
    "org_embedder.weight",
    "output_t.org.weight",
    "output_x.org.weight",
    "output_pair.organism_embed.weight",
}

def _is_organism_key(key: str) -> bool:
    return key in ORGANISM_KEYS

# Head tensor loading contract:
# - exact-load "_heads.splice_sites_junction.multiorg_linear.*":
#   shape [organism, hidden, splice_site_channel]
# - pair-map "_heads.splice_sites_junction.*_logits_embeddings":
#   shape [organism, scale_offset, tissue, channel], output axis 2
# - pair-map "_heads.*.multiorg_linear.*":
#   shape [organism, ..., output], output axis -1
# - pair-map "_heads.*.residual_scales.*":
#   shape [organism, output], output axis -1
HEADS_PREFIX = "_heads."
SPLICE_JUNCTION_HEAD_PREFIX = f"{HEADS_PREFIX}splice_sites_junction."
HEAD_EXACT_LOAD_PATTERNS = (
    (SPLICE_JUNCTION_HEAD_PREFIX, ".multiorg_linear."),
)
HEAD_OUTPUT_AXIS_PATTERNS = (
    (SPLICE_JUNCTION_HEAD_PREFIX, "suffix", "_logits_embeddings", 2),
    (HEADS_PREFIX, "contains", ".multiorg_linear.", -1),
    (HEADS_PREFIX, "contains", ".residual_scales.", -1),
)

def _head_output_axis(key: str, ndim: int) -> int | None:
    if ndim < 2:
        return None
    for prefix, contains in HEAD_EXACT_LOAD_PATTERNS:
        if key.startswith(prefix) and contains in key:
            return None
    for prefix, match_type, pattern, output_axis in HEAD_OUTPUT_AXIS_PATTERNS:
        if not key.startswith(prefix):
            continue
        if match_type == "suffix" and key.endswith(pattern):
            return output_axis
        if match_type == "contains" and pattern in key:
            return output_axis
    return None


class CheckpointLoadResult(NamedTuple):
    missing_keys: list[str]
    ignored_missing_keys: list[str]
    unexpected_keys: list[str]


IndexMap = Mapping[int, int]
HeadIndex = tuple[int, int]
HeadIndexMap = Mapping[HeadIndex, HeadIndex]

@dataclass(frozen=True)
class OrganismLoadSpec:
    # Mapping is checkpoint/source organism index -> target/model organism index.
    index_map: IndexMap | None = None

@dataclass(frozen=True)
class HeadLoadSpec:
    # Mapping is (source organism, source output) -> (target organism, target output).
    index_map: HeadIndexMap | None = None



##### LOADERS #####
def _load_metadata_json(metadata_path: Path) -> dict:
    with metadata_path.open() as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Expected metadata JSON object at {metadata_path}, got {type(loaded)!r}"
        )
    return loaded


def _download_hf_file(
    local_dir: Path | None,
    *,
    repo_id: str,
    repo_dir: str | None,
    repo_filename: str,
    token: str | bool | None,
    force_download: bool,
) -> Path:
    repo_dir = repo_dir or f"v{package_version()}"
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_filename,
            subfolder=repo_dir,
            token=token,
            force_download=force_download,
        )
    )
    if local_dir is None:
        return downloaded_path

    local_path = local_dir / repo_filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != local_path.resolve():
        shutil.copy2(downloaded_path, local_path)
    return local_path


def download_deepmind_metadata(
    local_dir: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    repo_filename: str = DEFAULT_CONVERTED_METADATA_FILENAME,
    download_raw_metadata: bool = True,
    raw_repo_filename: str = DEFAULT_RAW_METADATA_FILENAME,
    download_summary_metadata: bool = True,
    summary_repo_filename: str = DEFAULT_METADATA_SUMMARY_FILENAME,
    token: str | bool | None = None,
    force_download: bool = False,
) -> Path:
    """Download DeepMind AlphaGenome metadata JSON files from Hugging Face Hub."""
    local_dir = None if local_dir is None else Path(local_dir).expanduser()
    metadata_path = _download_hf_file(
        local_dir,
        repo_id=repo_id,
        repo_dir=repo_dir,
        repo_filename=repo_filename,
        token=token,
        force_download=force_download,
    )

    if download_raw_metadata:
        _download_hf_file(
            local_dir,
            repo_id=repo_id,
            repo_dir=repo_dir,
            repo_filename=raw_repo_filename,
            token=token,
            force_download=force_download,
        )
    if download_summary_metadata:
        _download_hf_file(
            local_dir,
            repo_id=repo_id,
            repo_dir=repo_dir,
            repo_filename=summary_repo_filename,
            token=token,
            force_download=force_download,
        )

    return metadata_path


def download_deepmind_state(
    local_dir: str | Path | None = None,
    *,
    fold: FoldName = DEFAULT_FOLD,
    download_all_folds: bool = False,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
) -> Path | list[Path]:
    """Download one fold, or every fold with download_all_folds=True."""
    local_dir = None if local_dir is None else Path(local_dir).expanduser()
    folds = FOLD_NAMES if download_all_folds else (fold,)
    paths = [
        _download_hf_file(
            local_dir,
            repo_id=repo_id,
            repo_dir=repo_dir,
            repo_filename=fold_filename(fold_name),
            token=token,
            force_download=force_download,
        )
        for fold_name in folds
    ]
    return paths if download_all_folds else paths[0]



##### INTELLIGENT STATE FILL-IN #####
# - Loads matching tensors directly.
# - Prefix-fills organism, track, and tissue axes when checkpoint/model sizes differ.
# - Uses explicit source-to-target index maps when provided by organism_spec/head_specs.
# - Leaves unfilled target slices at the model's initialized values.
# - Errors on shape mismatches outside the supported fill-in axes.

## Simple Helpers ##
def _validate_keys(
    *,
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    keys: set[str],
) -> None:
    missing_source_keys = keys - set(source_state_dict)
    missing_target_keys = keys - set(target_state_dict)
    errors = []
    if missing_source_keys:
        errors.append(
            "checkpoint is missing "
            f"{sorted(missing_source_keys)}"
        )
    if missing_target_keys:
        errors.append(
            "target model is missing "
            f"{sorted(missing_target_keys)}"
        )
    if errors:
        raise ValueError(
            "Cannot load standalone organism tensors because "
            + "; ".join(errors)
            + ". Pass organisms=False to skip organism-specific modules."
        )


def _normalize_load_dim(dim: int, rank: int) -> int:
    if not -rank <= dim < rank:
        raise IndexError(f"Dimension {dim} is out of range for tensor rank {rank}.")
    return dim % rank


## 1-Dimension ##
def _validate_load_shapes_1D(
    *,
    target: torch.Tensor,
    source: torch.Tensor,
    dim: int,
) -> int:
    if source.dim() != target.dim():
        raise ValueError(
            f"Cannot flexibly load tensors with different ranks: "
            f"checkpoint shape {tuple(source.shape)}, model shape {tuple(target.shape)}."
        )

    dim = _normalize_load_dim(dim, source.dim())
    mismatched_dims = [
        axis
        for axis, (source_size, target_size) in enumerate(zip(source.shape, target.shape))
        if axis != dim and source_size != target_size
    ]
    if mismatched_dims:
        raise ValueError(
            f"Cannot flexibly load tensor with checkpoint shape {tuple(source.shape)} "
            f"into model shape {tuple(target.shape)}. Only dimension {dim} may differ."
        )

    return dim


def _validate_index_map_1D(
    index_map: IndexMap,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    dim: int,
) -> None:
    source_indices = list(index_map.keys())
    target_indices = list(index_map.values())

    if not source_indices:
        raise ValueError("index_map must contain at least one source-to-target index.")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError(f"Target indices must be unique: {target_indices}.")
    if min(source_indices) < 0 or max(source_indices) >= source.size(dim):
        raise IndexError(
            f"Source indices {source_indices} are out of range for checkpoint tensor "
            f"size {source.size(dim)} along dimension {dim}."
        )
    if min(target_indices) < 0 or max(target_indices) >= target.size(dim):
        raise IndexError(
            f"Target indices {target_indices} are out of range for model tensor "
            f"size {target.size(dim)} along dimension {dim}."
        )


def _indexed_load_tensor_1D(
    *,
    target: torch.Tensor,
    source: torch.Tensor,
    index_map: IndexMap,
    dim: int = -1,
) -> torch.Tensor:
    if index_map is None:
        raise ValueError("index_map must be provided for _indexed_load_tensor_1D.")

    dim = _validate_load_shapes_1D(target=target, source=source, dim=dim)
    source_indices = list(index_map.keys())
    target_indices = list(index_map.values())
    _validate_index_map_1D(
        index_map,
        source=source,
        target=target,
        dim=dim,
    )

    target = target.detach().clone()
    source = source.detach().to(device=target.device, dtype=target.dtype)
    for s_idx, t_idx in zip(source_indices, target_indices):
        target.select(dim, t_idx).copy_(source.select(dim, s_idx))

    return target


def _prefix_load_tensor_1D(
    *,
    target: torch.Tensor,
    source: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    dim = _validate_load_shapes_1D(target=target, source=source, dim=dim)
    target = target.detach().clone()
    source = source.detach().to(device=target.device, dtype=target.dtype)

    n = min(source.size(dim), target.size(dim))

    target_slices = [slice(None)] * target.dim()
    source_slices = [slice(None)] * source.dim()
    target_slices[dim] = slice(0, n)
    source_slices[dim] = slice(0, n)

    target[tuple(target_slices)] = source[tuple(source_slices)]
    return target


def _flexible_load_tensor_1D(
    *,
    target: torch.Tensor | None,
    source: torch.Tensor,
    index_map: IndexMap | None = None,
    dim: int = -1,
) -> torch.Tensor:
    # Case 1: target not present, so just give the source
    if target is None:
        return source
    
    # Case 2: explicit index mapping
    if index_map is not None:
        return _indexed_load_tensor_1D(
            target=target,
            source=source,
            index_map=index_map,
            dim=dim,
        )
    
    # Case 3: shapes match, so just copy
    if source.shape == target.shape:
        return source
    
    # Case 4: shapes differ, prefix copy up to the minimum size between the two
    return _prefix_load_tensor_1D(target=target, source=source, dim=dim)


def _flexible_load_key_1D(
    *,
    target_state_dict: Mapping[str, torch.Tensor],
    source_state_dict: Mapping[str, torch.Tensor],
    key: str,
    index_map: IndexMap | None = None,
    dim: int = -1,
) -> torch.Tensor:
    source = source_state_dict[key]
    target = target_state_dict.get(key)
    if target is None:
        raise KeyError(f"Key {key!r} not found in target state dict.")
    return _flexible_load_tensor_1D(
        target=target,
        source=source,
        index_map=index_map,
        dim=dim,
    )


## 2-Dimension ##
def _validate_load_shapes_2D(
    *,
    key: str,
    target: torch.Tensor,
    source: torch.Tensor,
    output_axis: int | None,
) -> int | None:
    if source.ndim != target.ndim:
        raise ValueError(
            f"Cannot load {key!r}: checkpoint rank {source.ndim} "
            f"does not match model rank {target.ndim}."
        )

    if output_axis is None:
        if source.shape != target.shape:
            raise ValueError(
                f"Cannot load {key!r}: checkpoint shape {tuple(source.shape)} "
                f"does not match model shape {tuple(target.shape)}."
            )
        return None

    output_axis = _normalize_load_dim(output_axis, source.ndim)
    flexible_axes = {0, output_axis}

    for axis, (source_size, target_size) in enumerate(zip(source.shape, target.shape)):
        if axis not in flexible_axes and source_size != target_size:
            raise ValueError(
                f"Cannot load {key!r}: dimension {axis} is {source_size} "
                f"in the checkpoint and {target_size} in the model."
            )

    return output_axis


def _prefix_load_tensor_2D(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    output_axis: int,
) -> torch.Tensor:
    target = target.detach().clone()
    source = source.detach().to(device=target.device, dtype=target.dtype)

    source_slices = [slice(None)] * source.ndim
    target_slices = [slice(None)] * target.ndim
    n_organisms = min(source.size(0), target.size(0))
    n_outputs = min(source.size(output_axis), target.size(output_axis))
    source_slices[0] = slice(0, n_organisms)
    target_slices[0] = slice(0, n_organisms)
    source_slices[output_axis] = slice(0, n_outputs)
    target_slices[output_axis] = slice(0, n_outputs)

    target[tuple(target_slices)].copy_(source[tuple(source_slices)])
    return target


def _validate_index_map_2D(
    index_map: HeadIndexMap,
    *,
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    output_axis: int,
) -> None:
    if not index_map:
        raise ValueError(f"{key!r} head index_map must contain at least one source-to-target pair.")

    target_indices = list(index_map.values())
    if len(set(target_indices)) != len(target_indices):
        raise ValueError(f"{key!r} target indices must be unique: {target_indices}.")

    for source_index, target_index in index_map.items():
        if len(source_index) != 2 or len(target_index) != 2:
            raise ValueError(
                f"{key!r} head index_map must use "
                "(source_organism, source_output) -> (target_organism, target_output) pairs."
            )

        source_organism, source_output = source_index
        target_organism, target_output = target_index
        if not 0 <= source_organism < source.size(0):
            raise IndexError(
                f"{key!r} source organism indices must be in [0, {source.size(0)}), "
                f"got {source_organism}."
            )
        if not 0 <= source_output < source.size(output_axis):
            raise IndexError(
                f"{key!r} source output indices must be in [0, {source.size(output_axis)}), "
                f"got {source_output}."
            )
        if not 0 <= target_organism < target.size(0):
            raise IndexError(
                f"{key!r} target organism indices must be in [0, {target.size(0)}), "
                f"got {target_organism}."
            )
        if not 0 <= target_output < target.size(output_axis):
            raise IndexError(
                f"{key!r} target output indices must be in [0, {target.size(output_axis)}), "
                f"got {target_output}."
            )


def _indexed_load_tensor_2D(
    *,
    key: str,
    target: torch.Tensor,
    source: torch.Tensor,
    index_map: HeadIndexMap,
    output_axis: int,
) -> torch.Tensor:
    _validate_index_map_2D(
        index_map,
        key=key,
        source=source,
        target=target,
        output_axis=output_axis,
    )

    target = target.detach().clone()
    source = source.detach().to(device=target.device, dtype=target.dtype)
    for (
        source_organism,
        source_output,
    ), (
        target_organism,
        target_output,
    ) in index_map.items():
        source_slices = [slice(None)] * source.ndim
        target_slices = [slice(None)] * target.ndim
        source_slices[0] = source_organism
        source_slices[output_axis] = source_output
        target_slices[0] = target_organism
        target_slices[output_axis] = target_output
        target[tuple(target_slices)].copy_(source[tuple(source_slices)])

    return target


def _flexible_load_tensor_2D(
    *,
    key: str,
    target: torch.Tensor,
    source: torch.Tensor,
    head_spec: HeadLoadSpec,
) -> torch.Tensor:
    output_axis = _validate_load_shapes_2D(
        key=key,
        target=target,
        source=source,
        output_axis=_head_output_axis(key, source.ndim),
    )
    if output_axis is None:
        return source
    if source.shape == target.shape and head_spec.index_map is None:
        return source
    if head_spec.index_map is None:
        return _prefix_load_tensor_2D(
            source=source,
            target=target,
            output_axis=output_axis,
        )

    return _indexed_load_tensor_2D(
        key=key,
        target=target,
        source=source,
        index_map=head_spec.index_map,
        output_axis=output_axis,
    )


## Head-Specific Helpers ##
def _normalize_organism_load_spec(
    organism_spec: OrganismLoadSpec | IndexMap | None,
) -> OrganismLoadSpec:
    if organism_spec is None:
        return OrganismLoadSpec()
    if isinstance(organism_spec, OrganismLoadSpec):
        return organism_spec
    return OrganismLoadSpec(index_map=dict(organism_spec))


def _normalize_head_load_spec(
    head_spec: HeadLoadSpec | HeadIndexMap | None,
) -> HeadLoadSpec:
    if head_spec is None:
        return HeadLoadSpec()
    if isinstance(head_spec, HeadLoadSpec):
        return head_spec
    return HeadLoadSpec(index_map=dict(head_spec))


def _normalize_head_load_specs(
    head_specs: Mapping[str, HeadLoadSpec | HeadIndexMap] | None,
) -> dict[str, HeadLoadSpec]:
    if head_specs is None:
        return {}
    return {
        head_name: _normalize_head_load_spec(head_spec)
        for head_name, head_spec in head_specs.items()
    }


def _head_names(state_dict: Mapping[str, torch.Tensor]) -> set[str]:
    return {
        _get_head_name(key)
        for key in state_dict
        if key.startswith(HEADS_PREFIX)
    }


def _get_head_name(key: str) -> str:
    return key.removeprefix(HEADS_PREFIX).split(".", 1)[0]


def _head_key_suffixes(
    state_dict: Mapping[str, torch.Tensor],
    head_name: str,
) -> set[str]:
    prefix = f"{HEADS_PREFIX}{head_name}."
    return {
        key.removeprefix(prefix)
        for key in state_dict
        if key.startswith(prefix)
    }


def _flexible_head_keys(
    *,
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    head_specs: Mapping[str, HeadLoadSpec | HeadIndexMap] | None = None,
) -> tuple[set[str], set[str]]:
    target_head_names = _head_names(target_state_dict)
    source_head_names = _head_names(source_state_dict)

    if head_specs is None:
        potential_head_names = set(source_head_names)
    else:
        potential_head_names = set(head_specs)

    # Allow (but warn) source heads not in the target
    for head_name in sorted(potential_head_names - target_head_names):
        warnings.warn(
            f"Skipping source head {head_name!r} because it is not present in the target.",
        )

    # Allow (but warn) target heads not in the source
    if head_specs is None:
        missing_source_head_names = target_head_names - source_head_names
    else:
        missing_source_head_names = (
            potential_head_names & target_head_names
        ) - source_head_names
    for head_name in sorted(missing_source_head_names):
        warnings.warn(
            f"Skipping target head {head_name!r} because it is not present in the source.",
        )

    # Do not allow missing keys for selected heads that exist in both models.
    load_head_names = potential_head_names & source_head_names & target_head_names
    load_source_keys, load_target_keys = set(), set()
    for head_name in sorted(load_head_names):
        source_keys = _head_key_suffixes(source_state_dict, head_name)
        target_keys = _head_key_suffixes(target_state_dict, head_name)
        if source_keys == target_keys:
            prefix = f"{HEADS_PREFIX}{head_name}."
            load_source_keys.update(f"{prefix}{key}" for key in source_keys)
            load_target_keys.update(f"{prefix}{key}" for key in target_keys)
            continue

        missing_target_keys = sorted(source_keys - target_keys)
        missing_source_keys = sorted(target_keys - source_keys)
        errors = []
        if missing_target_keys:
            errors.append(f"target model is missing {missing_target_keys}")
        if missing_source_keys:
            errors.append(f"checkpoint is missing {missing_source_keys}")
        raise ValueError(
            f"Cannot load head {head_name!r} because its checkpoint and target "
            f"tensor keys differ: {'; '.join(errors)}."
        )

    return load_source_keys, load_target_keys


## Head Orchestrators ##
def _flexible_load_deepmind_state(
    *,
    source_state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    organisms: bool = True,
    organism_spec: OrganismLoadSpec | IndexMap | None = None,
    heads: bool = True,
    head_specs: Mapping[str, HeadLoadSpec | HeadIndexMap] | None = None,
    return_ignored_missing_keys: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], set[str]]:
    # NOTE: I don't particularly like creating specs when the user doesn't provide anything.
    # The pros are that it's unified and being able to intake just the index_map dicts is convenient.
    # Another option would be to only normalize if not None, at the cost of universality. Given
    # that the downstream functions work for specs with None index_maps, we stick with this for now.
    organism_spec = _normalize_organism_load_spec(organism_spec)
    normalized_head_specs = _normalize_head_load_specs(head_specs)
    head_source_keys, head_target_keys = set(), set()

    if organisms:
        _validate_keys(
            source_state_dict=source_state_dict,
            target_state_dict=target_state_dict,
            keys=ORGANISM_KEYS,
        )
    if heads:
        head_source_keys, head_target_keys = _flexible_head_keys(
            source_state_dict=source_state_dict,
            target_state_dict=target_state_dict,
            head_specs=head_specs,
        )

    # NOTE: planned_state_dict acts as a flexible subset state_dict of
    # the full source_state_dict after any prefix-mapping, index-mapping,
    # or even ignoring of tensors.
    planned_state_dict: dict[str, torch.Tensor] = {}
    loaded_keys: set[str] = set()
    ignored_missing_keys: set[str] = set()

    # Main Mapping Loop #
    for key, source in source_state_dict.items():
        target = target_state_dict.get(key)

        # Organism Keys
        if _is_organism_key(key):
            if not organisms:
                continue
            planned_state_dict[key] = _flexible_load_key_1D(
                key=key,
                target_state_dict=target_state_dict,
                source_state_dict=source_state_dict,
                index_map=organism_spec.index_map,
                dim=0,
            )
            loaded_keys.add(key)
            continue

        # Head Keys
        if key.startswith(HEADS_PREFIX):
            if key not in head_source_keys:
                continue
            head_name = _get_head_name(key)
            head_spec = normalized_head_specs.get(head_name, HeadLoadSpec())
            planned_state_dict[key] = _flexible_load_tensor_2D(
                key=key,
                target=target,
                source=source,
                head_spec=head_spec,
            )
            loaded_keys.add(key)
            continue

        if target is not None and source.shape != target.shape:
            raise ValueError(
                f"Cannot load {key!r}: checkpoint shape {tuple(source.shape)} "
                f"does not match model shape {tuple(target.shape)}."
            )

        # Add to subset state_dict for direct loading into the target
        planned_state_dict[key] = source
        if target is not None:
            loaded_keys.add(key)

    # Collect missing keys that are ignored due to organisms=False or heads=False.
    for key in target_state_dict:
        if key.startswith(HEADS_PREFIX) and key not in head_target_keys:
            ignored_missing_keys.add(key)
        if _is_organism_key(key) and not organisms:
            ignored_missing_keys.add(key)

    if return_ignored_missing_keys:
        return planned_state_dict, ignored_missing_keys
    return planned_state_dict


def load_deepmind_state(
    model: nn.Module,
    *,
    local_dir: str | Path | None = None,
    local_filename: str | None = None,
    organisms: bool = True,
    organism_spec: OrganismLoadSpec | IndexMap | None = None,
    heads: bool = True,
    head_specs: Mapping[str, HeadLoadSpec | HeadIndexMap] | None = None,
    fold: FoldName = DEFAULT_FOLD,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
    map_location: str | torch.device = "cpu",
    assign: bool = True,
):
    explicit_local_filename = local_filename is not None
    local_filename = local_filename or fold_filename(fold)
    if local_dir is None and explicit_local_filename:
        local_dir = Path.cwd()
    local_dir = None if local_dir is None else Path(local_dir).expanduser()
    resolved_checkpoint_path = (
        None
        if local_dir is None
        else local_dir / local_filename
    )

    if resolved_checkpoint_path is None or not resolved_checkpoint_path.exists() or force_download:
        downloaded_path = download_deepmind_state(
            local_dir,
            fold=fold,
            repo_id=repo_id,
            repo_dir=repo_dir,
            token=token,
            force_download=force_download,
        )
        if resolved_checkpoint_path is None:
            resolved_checkpoint_path = downloaded_path
        elif downloaded_path.resolve() != resolved_checkpoint_path.resolve():
            resolved_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, resolved_checkpoint_path)

    state_dict = torch.load(resolved_checkpoint_path, map_location=map_location)
    model_state_dict = model.state_dict()
    state_dict, ignored_missing_keys = _flexible_load_deepmind_state(
        source_state_dict=state_dict,
        target_state_dict=model_state_dict,
        organisms=organisms,
        organism_spec=organism_spec,
        heads=heads,
        head_specs=head_specs,
        return_ignored_missing_keys=True,
    )

    load_result = model.load_state_dict(state_dict, strict=False, assign=assign)
    missing_keys = [
        key
        for key in load_result.missing_keys
        if (
            key not in ignored_missing_keys and
            not key.endswith("._track_means")
        )
    ]
    return CheckpointLoadResult(
        missing_keys=missing_keys,
        ignored_missing_keys=list(ignored_missing_keys),
        unexpected_keys=list(load_result.unexpected_keys),
    )



##### CORE YIELDING FUNCTIONS #####
# (metadata, config, state)
def deepmind_metadata(
    metadata_dir: str | Path | None = None,
    *,
    metadata_filename: str = DEFAULT_CONVERTED_METADATA_FILENAME,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
) -> dict:
    metadata_dir = None if metadata_dir is None else Path(metadata_dir).expanduser()
    metadata_path = None if metadata_dir is None else metadata_dir / metadata_filename

    if metadata_path is not None and metadata_path.exists() and not force_download:
        return _load_metadata_json(metadata_path)

    metadata_path = download_deepmind_metadata(
        metadata_dir,
        repo_id=repo_id,
        repo_dir=repo_dir,
        repo_filename=metadata_filename,
        download_raw_metadata=False,
        download_summary_metadata=False,
        token=token,
        force_download=force_download,
    )
    return _load_metadata_json(metadata_path)


def deepmind_config(metadata: dict | None = None):
    if metadata is None:
        metadata = deepmind_metadata()

    return AlphaGenomeConfig(
        max_seq_len=1_048_576,
        num_channels=768,
        channel_increment=128,
        transformer_layers=9,
        num_q_heads=8,
        num_kv_heads=1,
        qk_head_dim=128,
        v_head_dim=192,
        pair_channels=128,
        pair_heads=32,
        pos_channels=64,
        transformer_mlp_ratio=2,
        embedder_mlp_ratio=2,
        num_splice_sites=512,
        splice_site_channels=768,
        metadata=metadata,
    )


def deepmind_model(
    device: str | torch.device = "cpu",
    metadata: dict | None = None,
    *,
    load_state: bool = False,
    local_dir: str | Path | None = None,
    local_filename: str | None = None,
    organisms: bool = True,
    organism_spec: OrganismLoadSpec | IndexMap | None = None,
    heads: bool = True,
    head_specs: Mapping[str, HeadLoadSpec | HeadIndexMap] | None = None,
    fold: FoldName = DEFAULT_FOLD,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
    map_location: str | torch.device = "cpu",
    assign: bool = True,
):
    if metadata is None:
        metadata = deepmind_metadata(
            repo_id=repo_id,
            repo_dir=repo_dir,
            token=token,
            force_download=force_download,
        )
    model = AlphaGenome(deepmind_config(metadata=metadata))
    if load_state:
        load_deepmind_state(
            model,
            local_dir=local_dir,
            local_filename=local_filename,
            organisms=organisms,
            organism_spec=organism_spec,
            heads=heads,
            head_specs=head_specs,
            fold=fold,
            repo_id=repo_id,
            repo_dir=repo_dir,
            token=token,
            force_download=force_download,
            map_location=map_location,
            assign=assign,
        )
    return model.to(torch.device(device))



##### ALIASES #####
# We generally don't support backward-compatibility but make
# some effort here for the high-level checkpoint function calls.
def deepmind_alphagenome_metadata(*args, **kwargs) -> dict:
    return deepmind_metadata(*args, **kwargs)


def download_alphagenome_metadata(*args, **kwargs) -> Path:
    return download_deepmind_metadata(*args, **kwargs)


def download_alphagenome_checkpoint(
    output_path: str | Path | None = None,
    *,
    fold: FoldName = DEFAULT_FOLD,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    repo_dir: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
) -> Path:
    # Backwards-compatibility shim for older callers that passed a destination
    # directory or destination checkpoint file as the first argument.
    output_path = None if output_path is None else Path(output_path).expanduser()
    local_dir = (
        output_path.parent
        if output_path is not None and output_path.suffix
        else output_path
    )
    downloaded_path = download_deepmind_state(
        local_dir,
        fold=fold,
        repo_id=repo_id,
        repo_dir=repo_dir,
        token=token,
        force_download=force_download,
    )
    if output_path is not None and output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded_path.resolve() != output_path.resolve():
            shutil.copy2(downloaded_path, output_path)
        return output_path
    return downloaded_path


def load_alphagenome_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path | None = None,
    *,
    heads: bool = True,
    organisms: bool = True,
    **kwargs,
):
    # Backwards-compatibility shim for older callers that passed a checkpoint
    # file or directory as the second positional argument.
    if checkpoint_path is None:
        return load_deepmind_state(
            model,
            heads=heads,
            organisms=organisms,
            **kwargs,
        )

    path = Path(checkpoint_path).expanduser()
    if path.exists() and path.is_dir():
        local_dir = path
        local_filename = kwargs.pop("local_filename", fold_filename(DEFAULT_FOLD))
    else:
        local_dir = path.parent
        local_filename = path.name

    return load_deepmind_state(
        model,
        local_dir=local_dir,
        local_filename=local_filename,
        heads=heads,
        organisms=organisms,
        **kwargs,
    )


def deepmind_alphagenome_config(*args, **kwargs):
    return deepmind_config(*args, **kwargs)


def deepmind_alphagenome_model(*args, **kwargs):
    return deepmind_model(*args, **kwargs)


def official_alphagenome_metadata(*args, **kwargs) -> dict:
    return deepmind_metadata(*args, **kwargs)


def official_alphagenome_config(*args, **kwargs):
    return deepmind_config(*args, **kwargs)

def official_alphagenome_model(*args, **kwargs):
    return deepmind_model(*args, **kwargs)



### DEPRECATED ALIASED FUNCTIONS AND THEIR USES ###
# def download_alphagenome_checkpoint(
#     output_path: str | Path,
#     *,
#     repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
#     filename: str = DEFAULT_ALPHAGENOME_CHECKPOINT,
#     token: str | bool | None = None,
#     force_download: bool = False,
# ) -> Path:
#     """Download the converted AlphaGenome PyTorch checkpoint from Hugging Face Hub.

#     Args:
#         output_path: Destination file path, or an existing directory. If a directory
#             is provided, the checkpoint is saved under ``filename`` inside it.
#         repo_id: Hugging Face Hub model repo ID.
#         filename: File name in the Hub repo.
#         token: Hugging Face auth token. If None, huggingface_hub uses the env token.
#         force_download: Whether to force a fresh download from the Hub.

#     Returns:
#         Path to the saved checkpoint file.
#     """
#     try:
#         from huggingface_hub import hf_hub_download
#     except ImportError as exc:
#         raise ImportError(
#             "download_alphagenome_checkpoint requires huggingface_hub. "
#             "Install it with `pip install huggingface_hub` or "
#             "`pip install alphagenome_pt[hub]`."
#         ) from exc

#     output_path = Path(output_path)
#     if output_path.exists() and output_path.is_dir():
#         destination = output_path / filename
#     elif output_path.suffix:
#         destination = output_path
#     else:
#         destination = output_path / filename

#     downloaded_path = Path(
#         hf_hub_download(
#             repo_id=repo_id,
#             filename=filename,
#             token=token,
#             force_download=force_download,
#         )
#     )

#     destination.parent.mkdir(parents=True, exist_ok=True)
#     if downloaded_path.resolve() != destination.resolve():
#         shutil.copy2(downloaded_path, destination)

#     return destination

# def load_alphagenome_checkpoint(
#     model: nn.Module,
#     checkpoint_path: str | Path,
#     *,
#     heads: bool = False,
#     organisms: bool = True,
#     repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
#     filename: str = DEFAULT_ALPHAGENOME_CHECKPOINT,
#     token: str | bool | None = None,
#     force_download: bool = False,
#     map_location: str | torch.device = "cpu",
#     assign: bool = True,
# ):
#     """Load converted official AlphaGenome weights into a PyTorch model.

#     If ``checkpoint_path`` does not exist, the checkpoint is downloaded there
#     with :func:`download_alphagenome_checkpoint`.

#     Args:
#         model: PyTorch AlphaGenome model.
#         checkpoint_path: Local checkpoint file path, or directory destination.
#         heads: If False, skip all ``_heads.*`` tensors so users can load the
#             official trunk with custom downstream heads.
#         organisms: If False, skip organism embedding tensors so users can load
#             the official trunk into models with different organism metadata.
#         repo_id: Hugging Face Hub model repo ID.
#         filename: File name in the Hub repo.
#         token: Hugging Face auth token. If None, huggingface_hub uses env auth.
#         force_download: Whether to force a fresh download from the Hub.
#         map_location: ``torch.load`` map location.
#         assign: Passed through to ``model.load_state_dict``.

#     Returns:
#         The ``load_state_dict`` incompatible-keys result.
#     """
#     checkpoint_path = Path(checkpoint_path)
#     if checkpoint_path.exists() and checkpoint_path.is_dir():
#         resolved_checkpoint_path = checkpoint_path / filename
#     elif checkpoint_path.suffix:
#         resolved_checkpoint_path = checkpoint_path
#     else:
#         resolved_checkpoint_path = checkpoint_path / filename

#     if not resolved_checkpoint_path.exists() or force_download:
#         resolved_checkpoint_path = download_alphagenome_checkpoint(
#             checkpoint_path,
#             repo_id=repo_id,
#             filename=filename,
#             token=token,
#             force_download=force_download,
#         )

#     state_dict = torch.load(resolved_checkpoint_path, map_location=map_location)
#     skipped_head_keys: set[str] = set()
#     skipped_organism_keys: set[str] = set()
#     if not heads:
#         skipped_head_keys = {
#             key
#             for key in model.state_dict()
#             if key.startswith("_heads.")
#         }
#         state_dict = {
#             key: value
#             for key, value in state_dict.items()
#             if not key.startswith("_heads.")
#         }
#     if not organisms:
#         skipped_organism_keys = {
#             key
#             for key in model.state_dict()
#             if _is_organism_embedding_key(key)
#         }
#         state_dict = {
#             key: value
#             for key, value in state_dict.items()
#             if not _is_organism_embedding_key(key)
#         }

#     load_result = model.load_state_dict(state_dict, strict=False, assign=assign)
#     missing_keys = [
#         key
#         for key in load_result.missing_keys
#         if (
#             key not in skipped_head_keys and
#             key not in skipped_organism_keys and
#             not key.endswith("._track_means")
#         )
#     ]
#     return CheckpointLoadResult(
#         missing_keys=missing_keys,
#         unexpected_keys=list(load_result.unexpected_keys),
#     )

# load_result = load_alphagenome_checkpoint(
#     model,
#     checkpoint_path,
#     heads=True,       # keep released output heads
#     organisms=True,	  # keep released human/mouse organism parameters
#     map_location="cpu",
# )
