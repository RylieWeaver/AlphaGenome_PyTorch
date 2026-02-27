# Provenance: Derived from AlphaGenome (Google LLC) Apache-2.0 code; translated to PyTorch and adjusted dict structure. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
import json
from typing import Union
from pathlib import Path

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# AlphaGenome


class Metadata():
    """
    Wraps a metadata dictionary with helper functions. The metadata dictionary MUST have the nested 
    structure of (organism/tasks --> head --> dict) where each head corresponds to a task/output type, namely: 
        - [atac, dnase, procap, rna_seq, chip_tf, chip_histone, contact_maps, splice_sites_classification, 
            splice_sites_usage, splice_sites_junction]

    NOTE: The metadata dictionary requires "num_tracks" for all heads except
    "splice_sites_junction", which requires "num_tissues" instead.

    Shape Naming Conventions:
    - O: Number of organisms
    - C: Number of channels
    - T: Number of tracks (or tissues for splice_sites_junction head)

    The num_tracks field indicates how many tracks of data are available for each organism for that head. In 
    the example below, there are 200 RNA-seq tracks for human and 150 for mouse, 100 CAGE tracks for human 
    and 0 for mouse. The means field indicates the means for each track for each organism, which are used for
    scaling the data in prediction (scaled = raw / mean). It must be padded to the maximum number of tracks 
    across orgnanisms for that head (using 1.0 for padding is recommended because that will have no effect on 
    scaling, however any mean value can be used because the loss is masked anyways). The track mask field is 
    a boolean mask indicating which tracks are valid for each organism, which are used to mask out the loss. 
    For example, if we have 200 RNA-seq tracks for human and 150 for mouse, then the last (200 - 150) = 50 
    entries of the track mask corresponding to mouse will be 0.
    
    Example metadata dictionary:
    {
        "organisms": ["human", "mouse"],
        "tasks": {
            "rna_seq": {
                "num_tracks": [200, 150],                               # [O]
                "means": [[0.1, 0.05, ...], [...]],                     # [O, max(T_rna)]
                "track_mask": [[1, 1, ..., 0], [1, 1, ..., 0]],         # [O, max(T_rna)]
            },
            "cage": {
                "num_tracks": [100, 0],                                 # [O]
                "means": [[0.2, 0.1, ...], [0, 0, ...]],                # [O, max(T_cage)]
                "track_mask": [[1, 1, ..., 0], [0, 0, ..., 0]],         # [O, max(T_cage)]
            },
            "splice_sites_junction": {
                "num_tissues": [25, 25],                                # [O]
                "means": [[0.3, 0.25, ...], [0.35, 0.3, ...]],  # [O, max(T_ss)]
            },
        },
    }

    NOTE: The names of the tasks (e.g. rna_seq) MUST be consistent with the names in heads.py for
    the HeadName class.


    """
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def get_organism(self, organism_index: int) -> str:
        return self.metadata["organisms"][organism_index]
    
    def get_organism_index(self, organism: str) -> int:
        return self.metadata["organisms"].index(organism)
    
    def get_heads(self) -> list[str]:
        """
        Returns the list of head names in the metadata. We assume that all organisms have the same heads, so we
        can just look at the first organism.
        """
        return list(self.metadata["heads"].keys())

    def get_num_organisms(self) -> int:
        """
        Returns the number of organisms in the metadata. Since the organisms
        are the top-level keys, this is simply the length of the metadata dictionary.
        """
        return len(self.metadata["organisms"])
    
    def get_organisms(self) -> list[str]:
        """
        Returns the list of organisms in the metadata.
        """
        return list(self.metadata["organisms"])

    def get_num_tracks_organism(self, organism: Union[str, int], head_name: str) -> int:
        """Returns the number of tracks for a given organism and head."""
        if head_name == "splice_sites_junction":
            raise ValueError("Use get_num_tissues_organism for splice_sites_junction head.")
        if isinstance(organism, str):
            organism = self.get_organism_index(organism)
        if head_name not in self.metadata["heads"]:
            return 0
        return self.metadata["heads"][head_name]["num_tracks"][organism]
    
    def get_num_tissues_organism(self, organism: Union[str, int], head_name: str) -> int:
        """Returns the number of tissues for a given organism and head."""
        if head_name != "splice_sites_junction":
            raise ValueError("get_num_tissues_organism is only for splice_sites_junction head.")
        if isinstance(organism, str):
            organism = self.get_organism_index(organism)
        if "splice_sites_junction" not in self.metadata["heads"]:
            return 0
        return self.metadata["heads"][head_name]["num_tissues"][organism]
    
    def get_num_tracks(self, head_name: str) -> int:
        if head_name == "splice_sites_junction":
            raise ValueError("Use get_num_tissues for splice_sites_junction head.")
        if head_name == "masked_language_modeling":
            raise ValueError("The masked_language_modeling head does not have num_tracks.")
        return max(self.metadata["heads"][head_name]["num_tracks"])
    
    def get_num_tissues(self, head_name: str) -> int:
        if head_name != "splice_sites_junction":
            raise ValueError("get_num_tissues is only for splice_sites_junction head.")
        return max(self.metadata["heads"][head_name]["num_tissues"])

    def get_means_organism(self, organism: Union[str, int], head_name: str) -> list[float]:
        """Returns the means for a given organism and head."""
        if isinstance(organism, str):
            organism = self.get_organism_index(organism)
        if head_name not in self.metadata["heads"]:
            return []
        if "means" not in self.metadata["heads"][head_name]:
            return []
        return self.metadata["heads"][head_name]["means"][organism]
    
    def get_means(self, head_name: str) -> torch.Tensor:
        """Returns the means for a given organism and head."""
        if head_name not in self.metadata["heads"]:
            return []
        if "means" not in self.metadata["heads"][head_name]:
            return []
        return self.metadata["heads"][head_name]["means"]
    
    def get_track_mask_organism(self, organism: Union[str, int], head_name: str) -> torch.Tensor:
        """
        Returns a boolean mask of shape [T] indicating which tracks are valid.
        This mask indicates whether the tracks for that organism were padded and
        are masked out in the loss computation.
        """
        if head_name == "splice_sites_junction":
            raise ValueError("Use get_tissue_mask_organism for splice_sites_junction head.")
        if head_name == "masked_language_modeling":
            raise ValueError("The masked_language_modeling head does not have a track mask.")
        if isinstance(organism, str):
            organism = self.get_organism_index(organism)
        if head_name not in self.metadata["heads"]:
            return torch.tensor([], dtype=torch.bool)
        if "track_mask" not in self.metadata["heads"][head_name]:
            max_tracks = self.get_num_tracks(head_name)
            num_tracks = self.get_num_tracks_organism(organism, head_name)
            track_mask = torch.zeros(max_tracks, dtype=torch.bool)
            track_mask[:num_tracks] = 1
            return track_mask
        return self.metadata["heads"][head_name]["track_mask"][organism]
    
    def get_tissue_mask_organism(self, organism: Union[str, int], head_name: str) -> torch.Tensor:
        """
        Returns a boolean mask of shape [T] indicating which tissues are valid.
        This mask indicates whether the tissues for that organism were padded and
        are masked out in the loss computation.
        """
        if head_name != "splice_sites_junction":
            raise ValueError("get_tissue_mask_organism is only for splice_sites_junction head.")
        if isinstance(organism, str):
            organism = self.get_organism_index(organism)
        if head_name not in self.metadata["heads"]:
            return torch.tensor([], dtype=torch.bool)
        if "tissue_mask" not in self.metadata["heads"][head_name]:
            max_tissues = self.get_num_tissues(head_name)
            num_tissues = self.get_num_tissues_organism(organism, head_name)
            tissue_mask = torch.zeros(max_tissues, dtype=torch.bool)
            tissue_mask[:num_tissues] = 1
            return tissue_mask
        return self.metadata["heads"][head_name]["tissue_mask"][organism]
    
    def get_track_mask(self, head_name: str) -> torch.Tensor:
        """
        Returns a boolean mask of shape [O, T] indicating which tracks are valid 
        for each organism.
        """
        if head_name == "splice_sites_junction":
            raise ValueError("Use get_tissue_mask for splice_sites_junction head.")
        if head_name not in self.metadata["heads"]:
            return torch.tensor([], dtype=torch.bool)
        if head_name == "masked_language_modeling":
            raise ValueError("The masked_language_modeling head does not have a track mask.")
        if "track_mask" not in self.metadata["heads"][head_name]:
            num_organisms = self.get_num_organisms()
            max_tracks = self.get_num_tracks(head_name)
            track_mask = torch.zeros((num_organisms, max_tracks), dtype=torch.bool)
            for organism_index in range(num_organisms):
                organism = self.get_organism(organism_index)
                track_mask[organism_index] = self.get_track_mask_organism(organism, head_name)
            return track_mask
        return self.metadata["heads"][head_name]["track_mask"]
    
    def get_tissue_mask(self, head_name: str) -> torch.Tensor:
        """
        Returns a boolean mask of shape [O, T] indicating which tissues are valid 
        for each organism.
        """
        if head_name != "splice_sites_junction":
            raise ValueError("get_tissue_mask is only for splice_sites_junction head.")
        if head_name not in self.metadata["heads"]:
            return torch.tensor([], dtype=torch.bool)
        if "tissue_mask" not in self.metadata["heads"][head_name]:
            num_organisms = self.get_num_organisms()
            max_tissues = self.get_num_tissues(head_name)
            tissue_mask = torch.zeros((num_organisms, max_tissues), dtype=torch.bool)
            for organism_index in range(num_organisms):
                organism = self.get_organism(organism_index)
                tissue_mask[organism_index] = self.get_tissue_mask_organism(organism, head_name)
            return tissue_mask
        return self.metadata["heads"][head_name]["tissue_mask"]
    
    def get_multiorg_track_mask(self, head_name: str, organism_index: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of shape [B, T] indicating which tracks are valid for each sample in the batch.
        This is used to create the track mask for the loss computation when we have a batch of samples from
        multiple organisms. The organism_index tensor indicates which organism each sample in the batch belongs to.
        """
        if head_name == "splice_sites_junction":
            raise ValueError("Use get_multiorg_tissue_mask for splice_sites_junction head.")
        if head_name == "masked_language_modeling":
            raise ValueError("The masked_language_modeling head does not have a track mask.")
        track_mask = self.get_track_mask(head_name)     # [O, T]
        return track_mask[organism_index]               # [B, T]
    
    def get_multiorg_tissue_mask(self, head_name: str, organism_index: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask of shape [B, T] indicating which tissues are valid for each sample in the batch.
        This is used to create the tissue mask for the loss computation when we have a batch of samples from
        multiple organisms. The organism_index tensor indicates which organism each sample in the batch belongs to.
        """
        if head_name != "splice_sites_junction":
            raise ValueError("get_multiorg_tissue_mask is only for splice_sites_junction head.")
        tissue_mask = self.get_tissue_mask(head_name)   # [O, T]
        return tissue_mask[organism_index]              # [B, T]

    def make_track_mask(self, head_name: str) -> torch.Tensor:
        """
        Creates a boolean mask of shape [O, T] indicating which tracks are valid for each organism.
        This is used to create the track mask if it is not provided in the metadata.
        """
        if head_name == "splice_sites_junction":
            raise ValueError("Use make_tissue_mask for splice_sites_junction head.")
        if head_name == "masked_language_modeling":
            raise ValueError("The masked_language_modeling head does not have a track mask.")
        track_mask = self.get_track_mask(head_name)
        self.metadata["heads"][head_name]["track_mask"] = track_mask
    
    def make_tissue_mask(self, head_name: str) -> torch.Tensor:
        """
        Creates a boolean mask of shape [O, T] indicating which tissues are valid for each organism.
        This is used to create the tissue mask if it is not provided in the metadata.
        """
        if head_name != "splice_sites_junction":
            raise ValueError("make_tissue_mask is only for splice_sites_junction head.")
        tissue_mask = self.get_tissue_mask(head_name)
        self.metadata["heads"][head_name]["tissue_mask"] = tissue_mask

    def make_all_masks(self):
        """
        Creates all track and tissue masks for all heads in the metadata.
        """
        for head_name in self.get_heads():
            if head_name == "splice_sites_junction":
                self.make_tissue_mask(head_name)
            elif head_name == "masked_language_modeling":
                continue
            else:
                self.make_track_mask(head_name)

    def to_dict(self) -> dict:
        """Returns the underlying metadata dictionary."""
        return self.metadata
