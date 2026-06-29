"""Utilities for tests/docs to generate synthetic objects (mostly data and metadata)"""
# External
from collections.abc import Sequence
import warnings

import torch
import torch.nn.functional as F

# Internal
from .bundles import BundleName
from .heads import HeadName
from .metadata import Metadata
from .model import AlphaGenome, AlphaGenomeConfig
from .schemas import DataBatch



##### CONSTANTS #####
GENOME_TRACK_HEADS = {
    HeadName.ATAC.value,
    HeadName.DNASE.value,
    HeadName.PROCAP.value,
    HeadName.CAGE.value,
    HeadName.RNA_SEQ.value,
    HeadName.CHIP_TF.value,
    HeadName.CHIP_HISTONE.value,
}
TRACK_HEADS = {
    HeadName.CONTACT_MAPS.value,
    HeadName.SPLICE_SITES_CLASSIFICATION.value,
    HeadName.SPLICE_SITES_USAGE.value,
}
TISSUE_HEADS = {
    HeadName.SPLICE_SITES_JUNCTION.value,
}
DEFAULT_SYNTHETIC_NUM_TRACKS = 4
SPLICE_SITES_CLASSIFICATION_NUM_CLASSES = 5


### METADATA ###
def _synthetic_organism_names(num_organisms: int) -> list[str]:
    return [f"organism_{i}" for i in range(num_organisms)]


def _synthetic_num_tracks(
    num_organisms: int,
    *,
    num_tracks: int | None,
) -> list[int]:
    num_tracks = DEFAULT_SYNTHETIC_NUM_TRACKS if num_tracks is None else num_tracks
    if num_tracks < 0:
        raise ValueError("num_tracks must be non-negative.")
    return [num_tracks] * num_organisms             # [O]


def _synthetic_means(num_tracks: Sequence[int], low: float = 3.0, high: float = 10.0) -> torch.Tensor:
    all_means = []
    max_tracks = max(num_tracks)
    for num in num_tracks:
        means = torch.empty(num).uniform_(low, high)
        means = F.pad(means, (0, max_tracks - num), value=1.0)
        all_means.append(means)
    return torch.stack(all_means, dim=0)                 # [O, T]


def synthetic_metadata_track_mask(num_tracks: Sequence[int]) -> torch.Tensor:
    max_tracks = max(num_tracks)
    track_mask = torch.zeros((len(num_tracks), max_tracks), dtype=torch.bool)
    for organism_index, n_tracks in enumerate(num_tracks):
        track_mask[organism_index, :n_tracks] = True
    return track_mask                                    # [O, T]


def _synthetic_head(
    head_name: str,
    *,
    num_organisms: int,
    num_tracks: int | None,
) -> dict:
    if head_name == HeadName.MASKED_LANGUAGE_MODELING.value:
        if num_tracks is not None:
            warnings.warn(
                "Ignoring num_tracks for masked language modeling metadata.",
                stacklevel=2,
            )
        return {}
    if head_name == HeadName.SPLICE_SITES_CLASSIFICATION.value:
        if (
            num_tracks is not None
            and num_tracks != SPLICE_SITES_CLASSIFICATION_NUM_CLASSES
        ):
            warnings.warn(
                "Ignoring num_tracks for splice site classification metadata; "
                f"using {SPLICE_SITES_CLASSIFICATION_NUM_CLASSES} classes.",
                stacklevel=2,
            )
        num_tracks = [SPLICE_SITES_CLASSIFICATION_NUM_CLASSES] * num_organisms
        return {
            "num_tracks": num_tracks,
            "track_mask": synthetic_metadata_track_mask(num_tracks),
        }

    num_tracks = _synthetic_num_tracks(
        num_organisms,
        num_tracks=num_tracks,
    )
    if head_name in GENOME_TRACK_HEADS:
        return {
            "num_tracks": num_tracks,
            "means": _synthetic_means(num_tracks),
            "track_mask": synthetic_metadata_track_mask(num_tracks),
        }
    if head_name in TRACK_HEADS:
        return {
            "num_tracks": num_tracks,
            "track_mask": synthetic_metadata_track_mask(num_tracks),
        }
    if head_name in TISSUE_HEADS:
        return {"num_tissues": num_tracks}

    raise ValueError(f"Unsupported head {head_name!r}.")


def _synthetic_head_names(heads: Sequence[HeadName] | None) -> tuple[HeadName, ...]:
    if heads is None:
        return ()

    selected = list(dict.fromkeys(heads))
    
    # NOTE: SPLICE_SITES_JUNCTION requires SPLICE_SITES_CLASSIFICATION
    if (
        HeadName.SPLICE_SITES_JUNCTION in selected
        and HeadName.SPLICE_SITES_CLASSIFICATION not in selected
    ):
        selected.append(HeadName.SPLICE_SITES_CLASSIFICATION)
    return tuple(selected)


def _synthetic_heads(
    heads: Sequence[HeadName] | None,
    *,
    num_organisms: int,
    num_tracks: int | None,
) -> dict:
    return {
        head_name: _synthetic_head(
            head_name,
            num_organisms=num_organisms,
            num_tracks=num_tracks,
        )
        for head_name in (head.value for head in _synthetic_head_names(heads))
    }


def synthetic_metadata(
    heads: Sequence[HeadName] | None = None,
    *,
    num_organisms: int = 2,
    organisms: Sequence[str] | None = None,
    num_tracks: int | None = None,
) -> Metadata:
    if organisms is None:
        organisms = _synthetic_organism_names(num_organisms)
    else:
        num_organisms = len(organisms)

    heads = _synthetic_heads(
        heads,
        num_organisms=num_organisms,
        num_tracks=num_tracks,
    )
    metadata = Metadata(
        metadata={
            "organisms": list(organisms),
            "heads": heads,
        }
    )
    metadata.make_all_masks()
    return metadata



##### DATA #####
def _head_resolution(head_name: str) -> int:
    return BundleName(head_name).get_resolution()


def synthetic_organism_index(batch_size: int, num_organisms: int = 2) -> torch.Tensor:
    return torch.tensor([i % num_organisms for i in range(batch_size)], dtype=torch.long)       # [B]


def synthetic_dna_sequence(batch_size: int, seq_len: int) -> torch.Tensor:
    bases = torch.randint(0, 4, (batch_size, seq_len), dtype=torch.long)
    return F.one_hot(bases, num_classes=4).to(torch.float32)                                    # [B, S, 4]


def synthetic_mlm(batch_size: int, seq_len: int, vocab_size: int = 5) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)                # [B, S]


def synthetic_poisson_tracks(means: torch.Tensor, organism_index: torch.Tensor, length: int) -> torch.Tensor:
    means = torch.as_tensor(means, dtype=torch.float32)
    rates = means[organism_index].unsqueeze(1).repeat(1, length, 1)                             # [B, S, T]
    return torch.poisson(rates).to(torch.float32)                                               # [B, S, T]


# NOTE: Metadata masks account for per-organism padding. Sample masks account for
# presence/absence of observed data in a given sample.
def synthetic_sample_track_mask(
    batch_size: int,
    num_tracks: int,
    *,
    ndim: int = 3,
    present_probability: float = 1.0,
) -> torch.Tensor:
    shape = (batch_size, *([1] * (ndim - 2)), num_tracks)
    if present_probability >= 1.0:
        return torch.ones(shape, dtype=torch.bool)                                               # [B, ..., T]
    return torch.rand(shape) < present_probability                                               # [B, ..., T]


def synthetic_genome_tracks(
    metadata: Metadata,
    head_name: str,
    organism_index: torch.Tensor,
    length: int,
) -> torch.Tensor:
    return synthetic_poisson_tracks(
        metadata.get_means(head_name),
        organism_index,
        length,
    )


# NOTE: allowed values are -1 (reverse), 0 (unknown or paired), 1 (forward)
def synthetic_rna_seq_strand(batch_size: int, num_tracks: int) -> torch.Tensor:
    return torch.randint(-1, 2, (batch_size, 1, num_tracks), dtype=torch.int32)          # [B, 1, T]


# Contact maps are continuous pairwise interaction targets trained with MSE.
def synthetic_contact_maps(
    batch_size: int,
    pair_len: int,
    num_tracks: int,
) -> torch.Tensor:
    return torch.rand(batch_size, pair_len, pair_len, num_tracks)                            # [B, S//2048, S//2048, T]


# Splice site classification is one class label per base, represented one-hot.
def synthetic_splice_sites(batch_size: int, seq_len: int, num_classes: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch_size, seq_len), dtype=torch.long)
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)                  # [B, S, C]


# Splice site usage is a continuous per-base, per-track target.
def synthetic_splice_site_usage(batch_size: int, seq_len: int, num_tracks: int) -> torch.Tensor:
    return torch.rand(batch_size, seq_len, num_tracks)                                       # [B, S, T]


# Junction prediction consumes donor/acceptor candidate positions.
def synthetic_splice_site_positions(
    batch_size: int,
    seq_len: int,
    num_splice_sites: int,
) -> torch.Tensor:
    return torch.randint(
        0,
        seq_len,
        (batch_size, 4, num_splice_sites),
        dtype=torch.long,
    )                                                                                       # [B, 4, K]


# Junction targets are count-like donor/acceptor pair observations per tissue and strand.
def synthetic_splice_junctions(
    batch_size: int,
    num_splice_sites: int,
    num_tissues: int,
) -> torch.Tensor:
    rates = 4.0 * torch.ones(
        batch_size,
        num_splice_sites,
        num_splice_sites,
        2 * num_tissues,
    )                                                                                       # [B, K, K, 2*T]
    return torch.poisson(rates).to(torch.float32)                                           # [B, K, K, 2*T]


def synthetic_batch(
    metadata: Metadata | None = None,
    *,
    batch_size: int = 2,
    seq_len: int = 8192,
    num_splice_sites: int = 2,
) -> DataBatch:
    if metadata is None:
        metadata = synthetic_metadata(heads=tuple(HeadName))

    organism_index = synthetic_organism_index(
        batch_size,
        num_organisms=metadata.get_num_organisms(),
    )
    batch_kwargs = {
        "dna_sequence": synthetic_dna_sequence(batch_size, seq_len),
        "organism_index": organism_index,
    }

    for head_name in metadata.get_heads():
        if head_name == HeadName.MASKED_LANGUAGE_MODELING.value:
            batch_kwargs["mlm"] = synthetic_mlm(batch_size, seq_len)
        elif head_name in GENOME_TRACK_HEADS:
            resolution = _head_resolution(head_name)
            length = seq_len // resolution
            num_tracks = metadata.get_num_tracks(head_name)
            batch_kwargs[head_name] = synthetic_genome_tracks(
                metadata,
                head_name,
                organism_index,
                length,
            )
            batch_kwargs[f"{head_name}_mask"] = synthetic_sample_track_mask(
                batch_size,
                num_tracks,
            )
            if head_name == HeadName.RNA_SEQ.value:
                batch_kwargs["rna_seq_strand"] = synthetic_rna_seq_strand(
                    batch_size,
                    num_tracks,
                )
        elif head_name == HeadName.CONTACT_MAPS.value:
            pair_len = seq_len // _head_resolution(head_name)
            num_tracks = metadata.get_num_tracks(head_name)
            batch_kwargs["contact_maps"] = synthetic_contact_maps(
                batch_size,
                pair_len,
                num_tracks,
            )
            batch_kwargs["contact_maps_mask"] = synthetic_sample_track_mask(
                batch_size,
                num_tracks,
                ndim=4,
            )
        elif head_name == HeadName.SPLICE_SITES_CLASSIFICATION.value:
            batch_kwargs["splice_sites"] = synthetic_splice_sites(
                batch_size,
                seq_len,
                metadata.get_num_tracks(head_name),
            )
        elif head_name == HeadName.SPLICE_SITES_USAGE.value:
            batch_kwargs["splice_site_usage"] = synthetic_splice_site_usage(
                batch_size,
                seq_len,
                metadata.get_num_tracks(head_name),
            )
            batch_kwargs["splice_site_usage_mask"] = synthetic_sample_track_mask(
                batch_size,
                metadata.get_num_tracks(head_name),
            )
        elif head_name == HeadName.SPLICE_SITES_JUNCTION.value:
            num_tissues = metadata.get_num_tissues(head_name)
            batch_kwargs["splice_site_positions"] = synthetic_splice_site_positions(
                batch_size,
                seq_len,
                num_splice_sites,
            )
            batch_kwargs["splice_junctions"] = synthetic_splice_junctions(
                batch_size,
                num_splice_sites,
                num_tissues,
            )
            batch_kwargs["splice_junctions_mask"] = synthetic_sample_track_mask(
                batch_size,
                num_tissues,
                ndim=4,
            )

    return DataBatch(**batch_kwargs)


### MODEL ###
def small_alphagenome(metadata: Metadata | None = None, **cfg_overrides) -> AlphaGenome:
    if metadata is None:
        metadata = synthetic_metadata()

    cfg_kwargs = {
        "max_seq_len": 2048*8,
        "num_channels": 64,
        "transformer_layers": 3,
        "metadata": metadata,
    }
    cfg_kwargs.update(cfg_overrides)
    return AlphaGenome(AlphaGenomeConfig(**cfg_kwargs))
