from __future__ import annotations

# External Imports
from pathlib import Path
import shutil
from collections.abc import Mapping
from typing import NamedTuple

# Internal Imports
import torch
from torch import nn

from .model import AlphaGenomeConfig


DEFAULT_ALPHAGENOME_REPO_ID = "RylieWeaver/alphagenome-pytorch"
DEFAULT_ALPHAGENOME_CHECKPOINT = "alphagenome_converted_state_dict.pt"


class CheckpointLoadResult(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


def official_alphagenome_metadata() -> dict:
    """Return metadata matching the converted official AlphaGenome checkpoint."""
    def head(num_tracks: int) -> dict:
        return {
            "num_tracks": [num_tracks, num_tracks],
            "means": [[1.0] * num_tracks, [1.0] * num_tracks],
        }

    return {
        "organisms": ["human", "mouse"],
        "heads": {
            "atac": head(256),
            "dnase": head(384),
            "procap": head(128),
            "cage": head(640),
            "rna_seq": head(768),
            "chip_tf": head(1664),
            "chip_histone": head(1152),
            "contact_maps": head(28),
            "splice_sites_classification": head(5),
            "splice_sites_usage": head(734),
            "splice_sites_junction": {
                "num_tissues": [367, 367],
                "means": [[1.0] * 367, [1.0] * 367],
            },
        },
    }


def official_alphagenome_config(metadata: Mapping | None = None) -> AlphaGenomeConfig:
    """Return the checkpoint-compatible official AlphaGenome model config.

    Args:
        metadata: Optional metadata override. Pass task-specific metadata here
            when loading the trunk but using different downstream heads.
    """
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
        metadata=metadata if metadata is not None else official_alphagenome_metadata(),
    )


def download_alphagenome_checkpoint(
    output_path: str | Path,
    *,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    filename: str = DEFAULT_ALPHAGENOME_CHECKPOINT,
    token: str | bool | None = None,
    force_download: bool = False,
) -> Path:
    """Download the converted AlphaGenome PyTorch checkpoint from Hugging Face Hub.

    Args:
        output_path: Destination file path, or an existing directory. If a directory
            is provided, the checkpoint is saved under ``filename`` inside it.
        repo_id: Hugging Face Hub model repo ID.
        filename: File name in the Hub repo.
        token: Hugging Face auth token. If None, huggingface_hub uses the env token.
        force_download: Whether to force a fresh download from the Hub.

    Returns:
        Path to the saved checkpoint file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "download_alphagenome_checkpoint requires huggingface_hub. "
            "Install it with `pip install huggingface_hub` or "
            "`pip install alphagenome_pt[hub]`."
        ) from exc

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        destination = output_path / filename
    elif output_path.suffix:
        destination = output_path
    else:
        destination = output_path / filename

    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            force_download=force_download,
        )
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != destination.resolve():
        shutil.copy2(downloaded_path, destination)

    return destination


def load_alphagenome_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    heads: bool = False,
    repo_id: str = DEFAULT_ALPHAGENOME_REPO_ID,
    filename: str = DEFAULT_ALPHAGENOME_CHECKPOINT,
    token: str | bool | None = None,
    force_download: bool = False,
    map_location: str | torch.device = "cpu",
    assign: bool = True,
):
    """Load converted official AlphaGenome weights into a PyTorch model.

    If ``checkpoint_path`` does not exist, the checkpoint is downloaded there
    with :func:`download_alphagenome_checkpoint`.

    Args:
        model: PyTorch AlphaGenome model.
        checkpoint_path: Local checkpoint file path, or directory destination.
        heads: If False, skip all ``_heads.*`` tensors so users can load the
            official trunk with custom downstream heads.
        repo_id: Hugging Face Hub model repo ID.
        filename: File name in the Hub repo.
        token: Hugging Face auth token. If None, huggingface_hub uses env auth.
        force_download: Whether to force a fresh download from the Hub.
        map_location: ``torch.load`` map location.
        assign: Passed through to ``model.load_state_dict``.

    Returns:
        The ``load_state_dict`` incompatible-keys result.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        resolved_checkpoint_path = checkpoint_path / filename
    elif checkpoint_path.suffix:
        resolved_checkpoint_path = checkpoint_path
    else:
        resolved_checkpoint_path = checkpoint_path / filename

    if not resolved_checkpoint_path.exists() or force_download:
        resolved_checkpoint_path = download_alphagenome_checkpoint(
            checkpoint_path,
            repo_id=repo_id,
            filename=filename,
            token=token,
            force_download=force_download,
        )

    state_dict = torch.load(resolved_checkpoint_path, map_location=map_location)
    skipped_head_keys: set[str] = set()
    if not heads:
        skipped_head_keys = {
            key
            for key in model.state_dict()
            if key.startswith("_heads.")
        }
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("_heads.")
        }

    load_result = model.load_state_dict(state_dict, strict=False, assign=assign)
    missing_keys = [
        key
        for key in load_result.missing_keys
        if key not in skipped_head_keys and not key.endswith("._track_means")
    ]
    return CheckpointLoadResult(
        missing_keys=missing_keys,
        unexpected_keys=list(load_result.unexpected_keys),
    )
