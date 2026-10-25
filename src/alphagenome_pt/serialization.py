"""Checkpoint weights in safetensors and metadata in JSON."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _save_state_dict(state_dict, path):
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in state_dict.items()
    }
    save_file(tensors, str(path), metadata={"format": "pt"})


def _load_state_dict(path, *, map_location="cpu"):
    if Path(path).suffix == ".pt":
        return torch.load(path, map_location=map_location, weights_only=True)
    return load_file(str(path), device=str(map_location))


def _metadata_json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Metadata value of type {type(value).__name__} is not JSON serializable")


def _save_metadata(metadata, path):
    with Path(path).open("w") as f:
        json.dump(metadata, f, default=_metadata_json_default, indent=4)
        f.write("\n")


def _load_metadata(path):
    if Path(path).suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    with Path(path).open() as f:
        return json.load(f)
