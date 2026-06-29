# External
from collections.abc import Mapping
import json
import os

import pytest
import torch

# Internal
from alphagenome_pt import (
    DEFAULT_CONVERTED_METADATA_FILENAME,
    DEFAULT_FOLD,
    DEFAULT_METADATA_SUMMARY_FILENAME,
    DEFAULT_RAW_METADATA_FILENAME,
    FOLD_NAMES,
    HeadLoadSpec,
    HeadName,
    OrganismLoadSpec,
    deepmind_config,
    deepmind_metadata,
    deepmind_model,
    download_alphagenome_checkpoint,
    download_deepmind_metadata,
    download_deepmind_state,
    fold_filename,
    load_alphagenome_checkpoint,
    load_deepmind_state,
    small_alphagenome,
    synthetic_metadata,
)
from alphagenome_pt.checkpoint import (
    HEADS_PREFIX,
    ORGANISM_KEYS,
    _download_hf_file,
    _flexible_load_deepmind_state,
)



##### DOWNLOADING #####
@pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST") != "1",
    reason="Set ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 to download the HF checkpoint.",
)
def test_download_deepmind_state_all_folds_from_hf(tmp_path):
    pytest.importorskip("huggingface_hub")

    checkpoint_paths = download_deepmind_state(
        tmp_path,
        download_all_folds=True,
    )

    assert checkpoint_paths == [tmp_path / fold_filename(fold) for fold in FOLD_NAMES]
    for checkpoint_path in checkpoint_paths:
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0


@pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST") != "1",
    reason="Set ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 to download the HF metadata.",
)
def test_download_deepmind_metadata_from_hf(tmp_path):
    pytest.importorskip("huggingface_hub")

    metadata_path = download_deepmind_metadata(
        tmp_path,
    )

    assert metadata_path == tmp_path / DEFAULT_CONVERTED_METADATA_FILENAME
    for filename in (
        DEFAULT_CONVERTED_METADATA_FILENAME,
        DEFAULT_RAW_METADATA_FILENAME,
        DEFAULT_METADATA_SUMMARY_FILENAME,
    ):
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("repo_filename", "contents"),
    [
        ("state.pt", "checkpoint"),
        ("metadata.json", '{"organisms": ["human"], "heads": {}}'),
    ],
)
def test_download_hf_file_copies_cached_file_to_local_dir(
    monkeypatch,
    tmp_path,
    repo_filename,
    contents,
):
    cache_dir = tmp_path / "cache"
    local_dir = tmp_path / "local"
    cached_file = cache_dir / "repo" / "v_test" / repo_filename
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text(contents)

    def fake_hf_hub_download(**kwargs):
        assert kwargs["filename"] == repo_filename
        assert kwargs["subfolder"] == "v_test"
        assert "local_dir" not in kwargs
        return str(cached_file)

    monkeypatch.setattr(
        "alphagenome_pt.checkpoint.hf_hub_download",
        fake_hf_hub_download,
    )

    path = _download_hf_file(
        local_dir,
        repo_id="repo",
        repo_dir="v_test",
        repo_filename=repo_filename,
        token=None,
        force_download=False,
    )

    assert path == local_dir / repo_filename
    assert path.read_text() == contents



##### FLEXIBLE LOADING #####
def _save_state(
    tmp_path,
    state_dict: dict[str, torch.Tensor],
    filename: str | None = None,
):
    path = tmp_path / (filename or fold_filename(DEFAULT_FOLD))
    torch.save(state_dict, path)
    return path


def _fill_model_state(model: torch.nn.Module, value: float) -> None:
    for tensor in model.state_dict().values():
        if torch.is_floating_point(tensor):
            tensor.fill_(value)


# NOTE: this gives every entry a unique values, which is useful
# for robustly testing equivalence of tensors after loading/rearranging.
def _set_arange(state_dict: dict[str, torch.Tensor], key: str) -> None:
    state_dict[key] = torch.arange(
        state_dict[key].numel(),
        dtype=state_dict[key].dtype,
    ).reshape_as(state_dict[key])


EXAMPLE_ALWAYS_LOADED_KEYS = (
    "sequence_encoder.downres_blocks.bin_size_1.conv1_d.weight",
    "sequence_encoder.downres_blocks.bin_size_2.conv_block1.conv.bias",
    "transformer_tower.blocks.0.mha.q_layer.weight",
    "sequence_decoder.upres_blocks.bin_size_1.conv_out.conv.bias",
    "output_t.fc1.weight",
)
RNA_SEQ_HEAD = HeadName.RNA_SEQ
RNA_SEQ_HEAD_KEYS = (
    f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.multiorg_linear.1.weight",
    f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.multiorg_linear.1.bias",
    f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.residual_scales.1",
)
RNA_SEQ_MULTIORG_WEIGHT_KEY = f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.multiorg_linear.1.weight"
RNA_SEQ_MULTIORG_BIAS_KEY = f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.multiorg_linear.1.bias"
RNA_SEQ_RESIDUAL_SCALES_KEY = f"{HEADS_PREFIX}{RNA_SEQ_HEAD.value}.residual_scales.1"
MLM_HEAD_NAME = HeadName.MASKED_LANGUAGE_MODELING.value
CONTACT_MAPS_HEAD_NAME = HeadName.CONTACT_MAPS.value


def _get_state_dict(
    model_or_state_dict: torch.nn.Module | Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if isinstance(model_or_state_dict, torch.nn.Module):
        return model_or_state_dict.state_dict()
    return model_or_state_dict


def _assert_equal_keys(
    model_or_state_dict1: torch.nn.Module | Mapping[str, torch.Tensor],
    model_or_state_dict2: torch.nn.Module | Mapping[str, torch.Tensor],
    keys: set[str] | tuple[str, ...],
) -> None:
    model1_state = _get_state_dict(model_or_state_dict1)
    model2_state = _get_state_dict(model_or_state_dict2)
    for key in keys:
        assert torch.equal(model1_state[key], model2_state[key])


def _assert_unequal_keys(
    model_or_state_dict1: torch.nn.Module | Mapping[str, torch.Tensor],
    model_or_state_dict2: torch.nn.Module | Mapping[str, torch.Tensor],
    keys: set[str] | tuple[str, ...],
) -> None:
    model1_state = _get_state_dict(model_or_state_dict1)
    model2_state = _get_state_dict(model_or_state_dict2)
    for key in keys:
        assert not torch.equal(model1_state[key], model2_state[key])



## 1-Dimensional ##
# (organism)
def test_load_deepmind_state_skips_organism_tensors(tmp_path):
    source_model = small_alphagenome()
    target_model = small_alphagenome()

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)
    target_organism_state = {
        key: target_model.state_dict()[key].clone()
        for key in ORGANISM_KEYS
    }

    _save_state(tmp_path, source_model.state_dict())
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        organisms=False,
    )

    _assert_equal_keys(source_model, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    _assert_unequal_keys(source_model, target_model, ORGANISM_KEYS)
    # NOTE: We also check that the target was unaltered in general
    for key in ORGANISM_KEYS:
        assert torch.equal(target_model.state_dict()[key], target_organism_state[key])
        assert key not in load_result.missing_keys
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_prefix_loads_organism_tensors(tmp_path):
    source_model = small_alphagenome(synthetic_metadata(num_organisms=2))
    # NOTE: The target model has fewer organisms than the source (must prefix fill!)
    target_model = small_alphagenome(synthetic_metadata(num_organisms=1))

    source_state = source_model.state_dict()
    for key in ORGANISM_KEYS:
        _set_arange(source_state, key)

    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        heads=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    for key in ORGANISM_KEYS:
        assert torch.equal(target_model.state_dict()[key][0], source_state[key][0])
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_uses_explicit_organism_index_map(tmp_path):
    source_model = small_alphagenome(synthetic_metadata(num_organisms=5))
    target_model = small_alphagenome(synthetic_metadata(num_organisms=7))

    source_state = source_model.state_dict()
    for key in ORGANISM_KEYS:
        _set_arange(source_state, key)
    target_organism_state = {
        key: target_model.state_dict()[key].clone()
        for key in ORGANISM_KEYS
    }

    _save_state(tmp_path, source_state)
    # NOTE: structure of the index_map is {source_index: target_index}
    # This mapping should robustly test the mapping by:
    # - differing the shapes
    # - target idx without a source idx 
    # - source idx without a target idx
    index_map = {
        0: 1,
        1: 0,
        2: 3,
        4: 5,
    }
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        organism_spec=OrganismLoadSpec(index_map=index_map),
    )

    # Check Mapped
    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    for source_index, target_index in index_map.items():
        for key in ORGANISM_KEYS:
            assert torch.equal(
                target_model.state_dict()[key][target_index],
                source_state[key][source_index],
            )
    # Check Unmapped
    target_indices = set(range(next(iter(target_organism_state.values())).size(0)))
    unmapped_target_indices = target_indices - set(index_map.values())
    for target_index in unmapped_target_indices:
        for key in ORGANISM_KEYS:
            assert torch.equal(
                target_model.state_dict()[key][target_index],
                target_organism_state[key][target_index],
            )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


# NOTE: It's easy to change a key while accidentally not uploading checkpoint behavior.
# This test will error loudly to prevent that from happening.
@pytest.mark.parametrize("missing_key", sorted(ORGANISM_KEYS))
@pytest.mark.parametrize("missing_from", ("source", "target"))
def test_load_deepmind_state_rejects_missing_organism_keys(missing_from, missing_key):
    source_state = dict(small_alphagenome().state_dict())
    target_state = dict(small_alphagenome().state_dict())

    if missing_from == "source":
        del source_state[missing_key]
        expected_error = "checkpoint is missing"
    else:
        del target_state[missing_key]
        expected_error = "target model is missing"

    with pytest.raises(ValueError, match="Pass organisms=False") as exc_info:
        _flexible_load_deepmind_state(
            source_state_dict=source_state,
            target_state_dict=target_state,
            heads=False,
        )

    assert expected_error in str(exc_info.value)
    assert missing_key in str(exc_info.value)


## 2-Dimensional ##
# (heads)
def test_load_deepmind_state_heads_false_skips_head_specs(tmp_path):
    metadata = synthetic_metadata(heads=(RNA_SEQ_HEAD,))
    source_model = small_alphagenome(metadata)
    target_model = small_alphagenome(metadata)

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)
    target_head_state = {
        key: target_model.state_dict()[key].clone()
        for key in target_model.state_dict()
        if key.startswith(HEADS_PREFIX)
    }

    source_state = source_model.state_dict()
    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        heads=False,
        head_specs={RNA_SEQ_HEAD.value: HeadLoadSpec()},
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    for key, value in target_head_state.items():
        assert torch.equal(target_model.state_dict()[key], value)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_empty_head_specs_loads_no_heads(tmp_path):
    metadata = synthetic_metadata(heads=(RNA_SEQ_HEAD,))
    source_model = small_alphagenome(metadata)
    target_model = small_alphagenome(metadata)

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)
    target_head_state = {
        key: target_model.state_dict()[key].clone()
        for key in target_model.state_dict()
        if key.startswith(HEADS_PREFIX)
    }

    source_state = source_model.state_dict()
    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        head_specs={},
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    for key, value in target_head_state.items():
        assert torch.equal(target_model.state_dict()[key], value)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_prefix_loads_into_smaller_head(tmp_path):
    source_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=2, num_tracks=4),
    )
    target_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=2, num_tracks=2),
    )

    source_state = source_model.state_dict()
    for key in RNA_SEQ_HEAD_KEYS:
        _set_arange(source_state, key)

    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        heads=True,
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert torch.equal(
        target_model._heads[RNA_SEQ_HEAD.value].multiorg_linear["1"].weight,
        source_state[RNA_SEQ_MULTIORG_WEIGHT_KEY][:, :, :2],
    )
    assert torch.equal(
        target_model._heads[RNA_SEQ_HEAD.value].multiorg_linear["1"].bias,
        source_state[RNA_SEQ_MULTIORG_BIAS_KEY][:, :2],
    )
    assert torch.equal(
        target_model._heads[RNA_SEQ_HEAD.value].residual_scales["1"],
        source_state[RNA_SEQ_RESIDUAL_SCALES_KEY][:, :2],
    )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_prefix_loads_into_larger_head_and_keeps_extra_tracks(tmp_path):
    source_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=1, num_tracks=2),
    )
    target_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=1, num_tracks=4),
    )

    _fill_model_state(target_model, -1.0)
    target_extra_tracks = target_model._heads[RNA_SEQ_HEAD.value].multiorg_linear["1"].weight[:, :, 2:].clone()

    source_state = source_model.state_dict()
    for key in RNA_SEQ_HEAD_KEYS:
        _set_arange(source_state, key)

    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert torch.equal(
        target_model._heads[RNA_SEQ_HEAD.value].multiorg_linear["1"].weight[:, :, :2],
        source_state[RNA_SEQ_MULTIORG_WEIGHT_KEY],
    )
    assert torch.equal(
        target_model._heads[RNA_SEQ_HEAD.value].multiorg_linear["1"].weight[:, :, 2:],
        target_extra_tracks,
    )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_uses_explicit_head_index_map(tmp_path):
    source_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=5, num_tracks=5),
    )
    target_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=7, num_tracks=7),
    )

    _fill_model_state(target_model, -1.0)
    source_state = source_model.state_dict()
    for key in RNA_SEQ_HEAD_KEYS:
        _set_arange(source_state, key)
    target_state = {
        key: target_model.state_dict()[key].clone()
        for key in RNA_SEQ_HEAD_KEYS
    }

    # NOTE: structure is {(source_organism, source_track): (target_organism, target_track)}.
    # This mapping tests different source/target shapes, crossed organism/track positions,
    # unmapped target positions, and unused source positions.
    index_map = {
        (0, 4): (6, 0),
        (1, 3): (0, 5),
        (2, 1): (3, 2),
        (4, 0): (1, 6),
    }

    _save_state(tmp_path, source_state)
    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        head_specs={
            RNA_SEQ_HEAD.value: HeadLoadSpec(
                index_map=index_map,
            )
        },
        organisms=False,
    )

    target_after = target_model.state_dict()
    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)

    # Check Mapped
    for (source_organism, source_track), (target_organism, target_track) in index_map.items():
        assert torch.equal(
            target_after[RNA_SEQ_MULTIORG_WEIGHT_KEY][target_organism, :, target_track],
            source_state[RNA_SEQ_MULTIORG_WEIGHT_KEY][source_organism, :, source_track],
        )
        assert torch.equal(
            target_after[RNA_SEQ_MULTIORG_BIAS_KEY][target_organism, target_track],
            source_state[RNA_SEQ_MULTIORG_BIAS_KEY][source_organism, source_track],
        )
        assert torch.equal(
            target_after[RNA_SEQ_RESIDUAL_SCALES_KEY][target_organism, target_track],
            source_state[RNA_SEQ_RESIDUAL_SCALES_KEY][source_organism, source_track],
        )

    # Check Unmapped
    target_indices = {
        (organism_index, track_index)
        for organism_index in range(target_state[RNA_SEQ_MULTIORG_BIAS_KEY].size(0))
        for track_index in range(target_state[RNA_SEQ_MULTIORG_BIAS_KEY].size(1))
    }
    unmapped_target_indices = target_indices - set(index_map.values())
    for target_organism, target_track in unmapped_target_indices:
        assert torch.equal(
            target_after[RNA_SEQ_MULTIORG_WEIGHT_KEY][target_organism, :, target_track],
            target_state[RNA_SEQ_MULTIORG_WEIGHT_KEY][target_organism, :, target_track],
        )
        assert torch.equal(
            target_after[RNA_SEQ_MULTIORG_BIAS_KEY][target_organism, target_track],
            target_state[RNA_SEQ_MULTIORG_BIAS_KEY][target_organism, target_track],
        )
        assert torch.equal(
            target_after[RNA_SEQ_RESIDUAL_SCALES_KEY][target_organism, target_track],
            target_state[RNA_SEQ_RESIDUAL_SCALES_KEY][target_organism, target_track],
        )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_rejects_duplicate_target_indices(tmp_path):
    source_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=2, num_tracks=4),
    )
    target_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=1, num_tracks=2),
    )

    source_state = source_model.state_dict()
    _save_state(tmp_path, source_state)

    with pytest.raises(ValueError, match="target indices must be unique"):
        load_deepmind_state(
            target_model,
            local_dir=tmp_path,
            head_specs={
                RNA_SEQ_HEAD.value: HeadLoadSpec(
                    index_map={
                        (0, 0): (0, 0),
                        (0, 1): (0, 0),
                    },
                )
            },
            organisms=False,
        )


def test_load_deepmind_state_rejects_out_of_range_indices(tmp_path):
    source_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=2, num_tracks=4),
    )
    target_model = small_alphagenome(
        synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_organisms=1, num_tracks=2),
    )

    source_state = source_model.state_dict()
    _save_state(tmp_path, source_state)

    with pytest.raises(IndexError, match="source output indices"):
        load_deepmind_state(
            target_model,
            local_dir=tmp_path,
            head_specs={RNA_SEQ_HEAD.value: HeadLoadSpec(index_map={(0, 4): (0, 0)})},
            organisms=False,
        )


def test_load_deepmind_state_rejects_non_loadable_shape_mismatch(tmp_path):
    source_model = small_alphagenome(synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_tracks=2))
    target_model = small_alphagenome(synthetic_metadata(heads=(RNA_SEQ_HEAD,), num_tracks=2))

    source_state = source_model.state_dict()
    weight_key = RNA_SEQ_MULTIORG_WEIGHT_KEY
    source_state[weight_key] = torch.ones(
        source_state[weight_key].shape[0],
        source_state[weight_key].shape[1] + 1,
        source_state[weight_key].shape[2],
    )
    _save_state(tmp_path, source_state)

    with pytest.raises(ValueError, match="dimension 1"):
        load_deepmind_state(
            target_model,
            local_dir=tmp_path,
            organisms=False,
        )



### METADATA LOADING ###
def test_deepmind_config_accepts_custom_metadata():
    metadata = {
        "organisms": ["human"],
        "heads": {MLM_HEAD_NAME: {}},
    }

    cfg = deepmind_config(metadata=metadata)

    assert cfg.max_seq_len == 1_048_576
    assert cfg.metadata.get_num_organisms() == 1
    assert cfg.metadata.get_organisms() == ["human"]
    assert cfg.metadata.get_heads() == [MLM_HEAD_NAME]
    assert cfg.metadata.metadata["heads"][MLM_HEAD_NAME] == {}


def test_deepmind_metadata_loads_local_converted_metadata(tmp_path):
    # NOTE: checks that the function loads the converted metadata if it exists
    metadata = {
        "organisms": ["human", "mouse"],
        "heads": {CONTACT_MAPS_HEAD_NAME: {"num_tracks": [28, 28]}},
    }
    metadata_path = tmp_path / DEFAULT_CONVERTED_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata))

    assert deepmind_metadata(metadata_dir=tmp_path) == metadata


def test_deepmind_metadata_downloads_hf_metadata(monkeypatch, tmp_path):
    # NOTE: checks that the function downloads the metadata if it does not exist locally
    metadata = {
        "organisms": ["human", "mouse"],
        "heads": {CONTACT_MAPS_HEAD_NAME: {"num_tracks": [28, 28]}},
    }
    downloaded_filenames = []

    def fake_download_hf_file(local_dir, *, repo_id, repo_dir, repo_filename, token, force_download):
        downloaded_filenames.append(repo_filename)
        path = (tmp_path if local_dir is None else local_dir) / repo_filename
        path.write_text(json.dumps(metadata))
        return path

    monkeypatch.setattr(
        "alphagenome_pt.checkpoint._download_hf_file",
        fake_download_hf_file,
    )

    loaded = deepmind_metadata(metadata_dir=tmp_path)

    assert loaded == metadata
    assert downloaded_filenames == [DEFAULT_CONVERTED_METADATA_FILENAME]
    assert (tmp_path / DEFAULT_CONVERTED_METADATA_FILENAME).exists()
    assert not (tmp_path / DEFAULT_RAW_METADATA_FILENAME).exists()


### HIGH-LEVEL LOADING ###
def test_load_deepmind_state_accepts_local_filename(tmp_path):
    source_model = small_alphagenome()
    target_model = small_alphagenome()

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)

    source_state = source_model.state_dict()
    local_filename = "tmp_state.pt"
    _save_state(tmp_path, source_state, local_filename)

    load_result = load_deepmind_state(
        target_model,
        local_dir=tmp_path,
        local_filename=local_filename,
        heads=False,
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert load_result.unexpected_keys == []


def test_load_deepmind_state_uses_cwd_for_local_filename_without_local_dir(tmp_path, monkeypatch):
    source_model = small_alphagenome()
    target_model = small_alphagenome()

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)

    source_state = source_model.state_dict()
    local_filename = "tmp_state.pt"
    _save_state(tmp_path, source_state, local_filename)

    monkeypatch.chdir(tmp_path)
    load_result = load_deepmind_state(
        target_model,
        local_filename=local_filename,
        heads=False,
        organisms=False,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert load_result.unexpected_keys == []


def test_load_deepmind_model_can_load_state(monkeypatch, tmp_path):
    # NOTE: we could do this test without monkeypatching the small
    # model, but using the full model size would be slow
    def fake_alphagenome(config):
        return small_alphagenome(config.metadata)

    monkeypatch.setattr("alphagenome_pt.checkpoint.AlphaGenome", fake_alphagenome)

    metadata = {
        "organisms": ["human"],
        "heads": {},
    }
    source_model = small_alphagenome(deepmind_config(metadata=metadata).metadata)
    _fill_model_state(source_model, 1.0)

    source_state = source_model.state_dict()
    local_filename = "tmp_state.pt"
    _save_state(tmp_path, source_state, local_filename)

    model = deepmind_model(
        metadata=metadata,
        load_state=True,
        local_dir=tmp_path,
        local_filename=local_filename,
    )

    assert isinstance(model, torch.nn.Module)
    _assert_equal_keys(source_state, model, EXAMPLE_ALWAYS_LOADED_KEYS)


### BACKWARDS-COMPATIBILITY ALIASES ###
def test_download_alphagenome_checkpoint_accepts_output_file(monkeypatch, tmp_path):
    downloaded_filenames = []

    def fake_download_deepmind_state(local_dir=None, **kwargs):
        downloaded_filenames.append(fold_filename(kwargs["fold"]))
        path = local_dir / fold_filename(kwargs["fold"])
        path.write_text("checkpoint")
        return path

    monkeypatch.setattr(
        "alphagenome_pt.checkpoint.download_deepmind_state",
        fake_download_deepmind_state,
    )

    output_path = tmp_path / "old_checkpoint_name.pt"
    path = download_alphagenome_checkpoint(output_path)

    assert path == output_path
    assert path.read_text() == "checkpoint"
    assert downloaded_filenames == [fold_filename(DEFAULT_FOLD)]


def test_load_alphagenome_checkpoint_accepts_checkpoint_file(tmp_path):
    source_model = small_alphagenome()
    target_model = small_alphagenome()

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)

    source_state = source_model.state_dict()
    checkpoint_path = _save_state(tmp_path, source_state, "old_checkpoint_name.pt")

    load_result = load_alphagenome_checkpoint(
        target_model,
        checkpoint_path,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert load_result.unexpected_keys == []


def test_load_alphagenome_checkpoint_accepts_checkpoint_directory(tmp_path):
    source_model = small_alphagenome()
    target_model = small_alphagenome()

    _fill_model_state(source_model, 1.0)
    _fill_model_state(target_model, -1.0)

    source_state = source_model.state_dict()
    _save_state(tmp_path, source_state)

    load_result = load_alphagenome_checkpoint(
        target_model,
        tmp_path,
    )

    _assert_equal_keys(source_state, target_model, EXAMPLE_ALWAYS_LOADED_KEYS)
    assert load_result.unexpected_keys == []
