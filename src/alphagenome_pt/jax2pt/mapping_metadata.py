"""Map public AlphaGenome JAX metadata into alphagenome_pt metadata."""

from __future__ import annotations

from typing import Any
import warnings

# Internal
from .utils import get_output_metadata


GENOME_TRACK_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
)

OTHER_TRACK_HEADS = (
    "contact_maps",
    "splice_sites",
    "splice_site_usage",
)

# Public metadata attributes use output/schema names, while model heads use
# JAX HeadName/checkpoint names.
OUTPUT_METADATA_TO_HEAD = {
    "contact_maps": "contact_maps",
    "splice_sites": "splice_sites_classification",
    "splice_site_usage": "splice_sites_usage",
    "splice_junctions": "splice_sites_junction",
}



def _get_metadata_records(metadata: Any) -> list[dict[str, Any]]:
    if isinstance(metadata, dict):
        return metadata["records"]
    return metadata.to_dict(orient="records")


def _is_padding_track(name: Any) -> bool:
    return isinstance(name, str) and name.strip().casefold() == "padding"


def _pad_list(values: list[Any], length: int, pad_value: Any) -> list[Any]:
    return values + [pad_value] * (length - len(values))


def _get_track_mask(metadata: Any) -> list[bool]:
    return [
        not _is_padding_track(record["name"])
        for record in _get_metadata_records(metadata)
    ]


def _get_track_means(
    metadata: Any,
    *,
    head_name: str,
    organism_name: str,
    mean_column: str,
) -> list[float]:
    means = []
    for i, record in enumerate(_get_metadata_records(metadata)):
        name = record["name"]
        if _is_padding_track(name):
            means.append(1.0)
            continue

        value = record.get(mean_column)
        if value is None:
            warnings.warn(
                f"Missing {mean_column!r} for non-padding AlphaGenome metadata "
                f"row: head={head_name!r}, organism={organism_name!r}, "
                f"index={i}, name={name!r}. Defaulting to 1.0.",
            )
            means.append(1.0)
            continue

        try:
            means.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not convert {mean_column!r} value to float: {value!r}"
            ) from exc

    return means


def _track_entry(
    track_metadata_by_org: list[tuple[str, Any]],
    *,
    head_name: str,
    mean_column: str | None = None,
) -> dict[str, Any]:
    num_tracks = [
        len(_get_metadata_records(metadata))
        for _, metadata in track_metadata_by_org
    ]
    max_tracks = max(num_tracks)

    track_mask = []
    if mean_column is not None:
        means = []

    for organism_name, metadata in track_metadata_by_org:
        mask = _get_track_mask(metadata)
        track_mask.append(_pad_list(mask, max_tracks, False))
        if mean_column is not None:
            _means = _get_track_means(
                metadata,
                head_name=head_name,
                organism_name=organism_name,
                mean_column=mean_column,
            )
            means.append(_pad_list(_means, max_tracks, 1.0))

    entry = {
        "num_tracks": num_tracks,
        "track_mask": track_mask,
    }
    if mean_column is not None:
        entry["means"] = means

    return entry


def _splice_junction_entry(track_metadata_by_org: list[tuple[str, Any]]) -> dict[str, Any]:
    num_tissues = [
        len(_get_metadata_records(metadata))
        for _, metadata in track_metadata_by_org
    ]
    max_tissues = max(num_tissues)
    return {
        "num_tissues": num_tissues,
        "tissue_mask": [
            _pad_list(_get_track_mask(metadata), max_tissues, False)
            for _, metadata in track_metadata_by_org
        ],
    }


def convert_metadata(
    loaded: list[tuple[str, Any]],  # list of (organism_name, organism_metadata)
) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for head_name in GENOME_TRACK_HEADS:
        heads[head_name] = _track_entry(
            [
                (organism_name, get_output_metadata(metadata, head_name))
                for organism_name, metadata in loaded
            ],
            head_name=head_name,
            mean_column="nonzero_mean",
        )

    for metadata_name in OTHER_TRACK_HEADS:
        head_name = OUTPUT_METADATA_TO_HEAD[metadata_name]
        heads[head_name] = _track_entry(
            [
                (organism_name, get_output_metadata(metadata, metadata_name))
                for organism_name, metadata in loaded
            ],
            head_name=head_name,
        )

    head_name = OUTPUT_METADATA_TO_HEAD["splice_junctions"]
    heads[head_name] = _splice_junction_entry(
        [
            (organism_name, get_output_metadata(metadata, "splice_junctions"))
            for organism_name, metadata in loaded
        ]
    )

    return {
        "organisms": [name for name, _ in loaded],
        "heads": heads,
    }


def summarize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    heads = {}
    for head_name, head in metadata["heads"].items():
        if "num_tracks" in head:
            summary = {
                "num_tracks": head["num_tracks"],
            }
            if "track_mask" in head:
                summary["padding_tracks"] = [
                    len(mask) - sum(mask)
                    for mask in head["track_mask"]
                ]
            heads[head_name] = summary
            continue

        if "num_tissues" in head:
            summary = {
                "num_tissues": head["num_tissues"],
            }
            if "tissue_mask" in head:
                summary["padding_tissues"] = [
                    len(mask) - sum(mask)
                    for mask in head["tissue_mask"]
                ]
            heads[head_name] = summary
            continue

        heads[head_name] = dict(head)

    return {
        "organisms": metadata["organisms"],
        "heads": heads,
    }
