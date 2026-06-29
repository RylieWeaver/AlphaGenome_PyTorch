# External
from types import SimpleNamespace

import pytest

# Internal
from alphagenome_pt import Metadata
from alphagenome_pt.jax2pt.mapping_metadata import (
    GENOME_TRACK_HEADS,
    OTHER_TRACK_HEADS,
    OUTPUT_METADATA_TO_HEAD,
    convert_metadata,
    summarize_metadata,
)


def _raw_metadata(
    *,
    genome_table: dict,
    other_table: dict,
    junction_table: dict,
) -> SimpleNamespace:
    data = {
        head_name: genome_table
        for head_name in GENOME_TRACK_HEADS
    }
    data.update({
        head_name: other_table
        for head_name in OTHER_TRACK_HEADS
    })
    data["splice_junctions"] = junction_table
    return SimpleNamespace(**data)


DUMMY_RAW_METADATA = [
    (
        "human",
        _raw_metadata(
            genome_table={
                "records": [
                    {"name": "track_a", "nonzero_mean": 0.5},
                    {"name": "Padding"},
                    {"name": "track_b", "nonzero_mean": 2.0},
                ],
            },
            other_table={
                "records": [
                    {"name": "contact_a"},
                    {"name": "Padding"},
                ],
            },
            junction_table={
                "records": [
                    {"name": "tissue_a"},
                    {"name": "Padding"},
                ],
            },
        ),
    ),
    (
        "mouse",
        _raw_metadata(
            genome_table={
                "records": [
                    {"name": "Padding"},
                    {"name": "track_c", "nonzero_mean": 3.0},
                ],
            },
            other_table={
                "records": [
                    {"name": "Padding"},
                    {"name": "contact_b"},
                ],
            },
            junction_table={
                "records": [
                    {"name": "Padding"},
                    {"name": "Padding"},
                ],
            },
        ),
    ),
]

DUMMY_GENOME_HEAD = {
    "num_tracks": [3, 2],
    "track_mask": [
        [True, False, True],
        [False, True, False],
    ],
    "means": [
        [0.5, 1.0, 2.0],
        [1.0, 3.0, 1.0],
    ],
}

DUMMY_OTHER_TRACK_HEAD = {
    "num_tracks": [2, 2],
    "track_mask": [
        [True, False],
        [False, True],
    ],
}

DUMMY_SPLICE_JUNCTION_HEAD = {
    "num_tissues": [2, 2],
    "tissue_mask": [
        [True, False],
        [False, False],
    ],
}

DUMMY_CONVERTED_METADATA = {
    "organisms": ["human", "mouse"],
    "heads": {
        **{
            head_name: DUMMY_GENOME_HEAD
            for head_name in GENOME_TRACK_HEADS
        },
        **{
            OUTPUT_METADATA_TO_HEAD[head_name]: DUMMY_OTHER_TRACK_HEAD
            for head_name in OTHER_TRACK_HEADS
        },
        OUTPUT_METADATA_TO_HEAD["splice_junctions"]: DUMMY_SPLICE_JUNCTION_HEAD,
    },
}

DUMMY_SUMMARY_METADATA = {
    "organisms": ["human", "mouse"],
    "heads": {
        **{
            head_name: {
                "num_tracks": [3, 2],
                "padding_tracks": [1, 2],
            }
            for head_name in GENOME_TRACK_HEADS
        },
        **{
            OUTPUT_METADATA_TO_HEAD[head_name]: {
                "num_tracks": [2, 2],
                "padding_tracks": [1, 1],
            }
            for head_name in OTHER_TRACK_HEADS
        },
        OUTPUT_METADATA_TO_HEAD["splice_junctions"]: {
            "num_tissues": [2, 2],
            "padding_tissues": [1, 2],
        },
    },
}


def test_convert_metadata_matches_expected_converted_metadata():
    assert convert_metadata(DUMMY_RAW_METADATA) == DUMMY_CONVERTED_METADATA


def test_converted_metadata_is_loadable_by_metadata_class():
    metadata = Metadata(DUMMY_CONVERTED_METADATA)

    assert metadata.get_num_tracks_organism("human", "atac") == 3
    assert metadata.get_num_tracks("atac") == 3
    assert metadata.get_num_tissues_organism("mouse", "splice_sites_junction") == 2
    assert metadata.get_num_tissues("splice_sites_junction") == 2


def test_convert_metadata_warns_when_nonpadding_mean_is_missing():
    human = _raw_metadata(
        genome_table={"records": [{"name": "track_a"}]},
        other_table={"records": [{"name": "contact_a"}]},
        junction_table={"records": [{"name": "tissue_a"}]},
    )

    with pytest.warns(UserWarning, match="Missing 'nonzero_mean'"):
        converted = convert_metadata([("human", human)])
    assert converted["heads"]["atac"]["means"] == [[1.0]]


def test_summarize_metadata_matches_expected_summary_metadata():
    assert summarize_metadata(DUMMY_CONVERTED_METADATA) == DUMMY_SUMMARY_METADATA
