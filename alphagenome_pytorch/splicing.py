# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports""" 
# General
import numpy as np

# Torch
import torch
import torch.nn as nn

# AlphaGenome



def top_k_splice_sites(
    x: torch.Tensor,
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> torch.Tensor:
    """Returns the top k splice sites from the predictions.

    Args:
      - x: Array of shape [B, S, 5] containing splice site predictions 
        (donor +ve, acceptor +ve, donor -ve, acceptor -ve, other).
      - k: Number of top splice sites to return.
      - pad_to_length: Pad the output to this length.
      - threshold: Threshold to filter out low confidence splice sites.
    """
    B, S, _ = x.size()
    device = x.device
    values, positions = torch.topk(x[..., :4], k=k, dim=1)  # both [B, k, 4]
    fill_int = S  # any value greater than S-1 (so that any valid index so ends up at the end after sort)
    if threshold > 0:
        # Fill positions where values < threshold with fill_int 
        positions = torch.where(values < threshold, torch.tensor(fill_int, dtype=torch.int32, device=device), positions)
    positions, _ = torch.sort(positions, dim=1)
    if threshold > 0:
        # Replace fill_int back to -1 after sorting
        positions = torch.where(positions == fill_int, torch.tensor(-1, dtype=torch.int32, device=device), positions)
    positions = positions.permute(0, 2, 1).to(torch.int32)
    if positions.shape[2] < pad_to_length:
        padding_shape = (B, 4, pad_to_length - positions.shape[2])
        padding = torch.full(
            padding_shape, -1, dtype=torch.int32, device=device  # NOTE: Using -1 as padding value
        )
        positions = torch.cat([positions, padding], dim=2)
    return positions  # [B, 4, pad_to_length]


def generate_splice_site_positions(
    ref: torch.Tensor,
    alt: torch.Tensor | None,
    splice_sites: torch.Tensor | None,
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> torch.Tensor:
    """
    Returns the top k splice sites from predictions and (true) splice sites.
    This function can take alt probabilities and true splice sites in addition to ref 
    so that multiple sources of splice sites can be combined for inference.
    """

    if alt is not None:
        ref = torch.maximum(ref, alt)
    if splice_sites is not None:
        ref = torch.maximum(ref, splice_sites)
    return top_k_splice_sites(
        ref, k=k, pad_to_length=pad_to_length, threshold=threshold
    )  # [B, 4, pad_to_length]
