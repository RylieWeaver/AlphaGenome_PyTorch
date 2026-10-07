# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# External
import torch
import torch.distributed as dist

# Internal
from .distributed import all_gather



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
    ref: torch.Tensor,                  # [B, S_local, 5] (SP) or [B, S, 5]
    alt: torch.Tensor | None,           # [B, S_local, 5] (SP) or [B, S, 5]
    splice_sites: torch.Tensor | None,  # [B, S_local, 5] (SP) or [B, S, 5]
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
    sequence_group=None,
) -> torch.Tensor:
    """
    Returns the top k splice sites from predictions and (true) splice sites.
    This function can take alt probabilities and true splice sites in addition to ref
    so that multiple sources of splice sites can be combined for inference.

    If ``sequence_group`` is provided, inputs contain local sequence shards. The
    full sequence is collected, SP rank 0 selects positions, and those positions
    are broadcast to every rank in the group.
    """

    # Take max scores over all possible sources
    if alt is not None:
        ref = torch.maximum(ref, alt)                              # [B, S_local, 5] (SP) or [B, S, 5]
    if splice_sites is not None:
        ref = torch.maximum(ref, splice_sites)                     # [B, S_local, 5] (SP) or [B, S, 5]

    # SP-gather scores
    if sequence_group is not None:
        ref = all_gather(ref, dim=1, group=sequence_group)         # [B, S, 5]

    # topk does not guarantee stable indices for ties, so one SP rank selects.
    if sequence_group is None or dist.get_rank(sequence_group) == 0:
        positions = top_k_splice_sites(
            ref, k=k, pad_to_length=pad_to_length, threshold=threshold
        ).contiguous()                                             # [B, 4, K]
    else:
        positions = torch.empty(
            (ref.shape[0], 4, max(k, pad_to_length)),
            dtype=torch.int32,
            device=ref.device,
        )                                                          # [B, 4, K]

    if sequence_group is not None:
        dist.broadcast(
            positions,
            src=dist.get_global_rank(sequence_group, 0),
            group=sequence_group,
        )                                                          # [B, 4, K] on every SP rank
    return positions                                               # [B, 4, K]
