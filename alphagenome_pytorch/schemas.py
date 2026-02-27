# Provenance: PyTorch port of AlphaGenome (Google LLC) code (Apache-2.0). Modified by Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
from typing import Union
from dataclasses import dataclass

# Torch
import torch

# AlphaGenome
from .metadata import Metadata
from . import bundles



@dataclass
class DataBatch:
    """Input batch for the model."""

    dna_sequence: list[str] | None = None                       # [B, S, V]
    mlm: torch.Tensor | None = None                             # [B, S]
    organism_index: torch.Tensor | None = None                  # [B]
    atac: torch.Tensor | None = None                            # [B, S, C_ATAC]
    atac_mask: torch.Tensor | None = None                       # [B, #S, C_ATAC]
    dnase: torch.Tensor | None = None                           # [B, S, C_DNASE]
    dnase_mask: torch.Tensor | None = None                      # [B, #S, C_DNASE]
    procap: torch.Tensor | None = None                          # [B, S, C_PROCAP]
    procap_mask: torch.Tensor | None = None                     # [B, #S, C_PROCAP]
    chip_histone: torch.Tensor | None = None                    # [B, S//128, C_CHIP_HISTONE]
    chip_histone_mask: torch.Tensor | None = None               # [B, #S//128, C_CHIP_HISTONE]
    chip_tf: torch.Tensor | None = None                         # [B, S//128, C_CHIP_TF]
    chip_tf_mask: torch.Tensor | None = None                    # [B, #S//128, C_CHIP_TF]
    rna_seq: torch.Tensor | None = None                         # [B, S, C_RNA_SEQ]
    rna_seq_mask: torch.Tensor | None = None                    # [B, #S, C_RNA_SEQ]
    rna_seq_strand: torch.Tensor | None = None                  # [B, 1, C_RNA_SEQ]
    cage: torch.Tensor | None = None                            # [B, S, C_CAGE]
    cage_mask: torch.Tensor | None = None                       # [B, #S, C_CAGE]
    contact_maps: torch.Tensor | None = None                    # [B, S//2048, S//2048, C_CONTACT_MAPS]
    contact_maps_mask: torch.Tensor | None = None               # [B, #S//2048, #S//2048, C_CONTACT_MAPS]
    splice_site_positions: torch.Tensor | None = None           # [B, 4, D]
    splice_sites: torch.Tensor | None = None                    # [B, S, C_SPLICE_SITES]
    splice_site_usage: torch.Tensor | None = None               # [B, S, C_SPLICE_SITE_USAGE]
    splice_site_usage_mask: torch.Tensor | None = None          # [B, #S, C_SPLICE_SITE_USAGE]
    splice_site_junction: torch.Tensor | None = None            # [B, D, A, C_SPLICE_SITE_junction]
    splice_site_junction_mask: torch.Tensor | None = None       # [B, #D, #A, C_SPLICE_SITE_junction]

    def get_organism_index(self) -> torch.Tensor:               # [B]
        """Returns the organism index data."""
        if self.organism_index is None:
            raise ValueError('Organism index data is not present in the batch.')
        return self.organism_index

    def get_genome_tracks(
        self,
        bundle: bundles.BundleName
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the genome tracks data for the given bundle if present."""
        match bundle:
            case bundles.BundleName.ATAC:
                data, mask = self.atac, self.atac_mask
            case bundles.BundleName.DNASE:
                data, mask = self.dnase, self.dnase_mask
            case bundles.BundleName.PROCAP:
                data, mask = self.procap, self.procap_mask
            case bundles.BundleName.CAGE:
                data, mask = self.cage, self.cage_mask
            case bundles.BundleName.RNA_SEQ:
                data, mask = self.rna_seq, self.rna_seq_mask
            case bundles.BundleName.CHIP_TF:
                data, mask = self.chip_tf, self.chip_tf_mask
            case bundles.BundleName.CHIP_HISTONE:
                data, mask = self.chip_histone, self.chip_histone_mask
            case _:
                raise ValueError(
                    f'Unknown bundle name: {bundle!r}. Is it a genome tracks bundle?'
                )

        if data is None or mask is None:
            raise ValueError(f'{bundle.name!r} data is not present in the batch.')
        return data, mask
    
    def to(self, device: torch.device):
        """Moves all tensors in the batch to the specified device."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
        return self


@dataclass
class Channels:
    """
    AlphaGenome num channels at different resolutions.
    """
    
    channels_1bp: int | None = None
    channels_128bp: int | None = None
    channels_pair: int | None = None

    def get_num_channels(self, resolution: int) -> int | None:
        if resolution == 128:
            return self.channels_128bp
        elif resolution == 1:
            return self.channels_1bp
        else:
            raise ValueError(f'Unsupported resolution: {resolution}')
