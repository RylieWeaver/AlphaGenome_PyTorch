# State-dict contract coverage in this file is adapted from
# genomicsxai/alphagenome-pytorch,
# tests/integration/test_checkpoint_roundtrip.py (Apache-2.0). Existing
# configuration and model save/load tests are original work.

# External
import pytest
import torch

# Internal
from alphagenome_pt import (
    AlphaGenome,
    AlphaGenomeConfig,
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import (
    DNA_SEQUENCE,
    assert_finite_metric_tree,
    assert_predictions_close,
    build_small_model,
    build_small_model_with_batch,
)


@pytest.mark.parametrize(
    "cfg_overrides",
    [
        {
            "num_channels": 48,
            "transformer_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 1,
            "qk_head_dim": 12,
            "v_head_dim": 13,
        },
        {
            "num_channels": 72,
            "channel_increment": 12,
            "transformer_layers": 2,
            "num_q_heads": 6,
            "num_kv_heads": 2,
            "qk_head_dim": 12,
            "v_head_dim": 12,
            "pair_channels": 12,
            "dropout": 0.1,
        },
        {
            "num_channels": 96,
            "channel_increment": 16,
            "transformer_layers": 1,
            "num_q_heads": 8,
            "num_kv_heads": 4,
            "qk_head_dim": 16,
            "v_head_dim": 16,
            "pair_channels": 20,
            "num_splice_sites": 2,
            "splice_site_channels": 64,
        },
    ],
)
def test_hyperparameter_smoke(cfg_overrides: dict):
    metadata = synthetic_metadata(
        (HeadName.MASKED_LANGUAGE_MODELING,),
        organisms=("human",),
    )
    model = small_alphagenome(metadata, **cfg_overrides)

    batch = synthetic_batch(metadata, batch_size=1, seq_len=model.max_seq_len)

    result = model.loss(batch)
    result.total.backward()

    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.parameters()
    )


@pytest.mark.parametrize("qk_head_dim", (0, 11))
def test_qk_head_dim_must_be_positive_and_even(qk_head_dim: int):
    metadata = synthetic_metadata(
        (HeadName.MASKED_LANGUAGE_MODELING,),
        organisms=("human",),
    )

    with pytest.raises(
        AssertionError,
        match="qk_head_dim must be positive and even for RoPE",
    ):
        small_alphagenome(metadata, qk_head_dim=qk_head_dim)


@pytest.mark.parametrize("v_head_dim", (-1, 0))
def test_v_head_dim_must_be_positive(v_head_dim: int):
    metadata = synthetic_metadata(
        (HeadName.MASKED_LANGUAGE_MODELING,),
        organisms=("human",),
    )

    with pytest.raises(AssertionError, match="v_head_dim must be positive"):
        small_alphagenome(metadata, v_head_dim=v_head_dim)


def test_config_save_load_roundtrip(tmp_path):
    metadata = synthetic_metadata(
        (HeadName.MASKED_LANGUAGE_MODELING, HeadName.RNA_SEQ),
        organisms=("human",),
    )
    model = small_alphagenome(metadata)
    cfg = model.cfg

    cfg_path = tmp_path / "config.json"
    metadata_path = tmp_path / "metadata.pt"
    cfg.save(cfg_path, metadata_path)
    loaded = AlphaGenomeConfig.load(cfg_path, metadata_path)

    assert loaded.max_seq_len == cfg.max_seq_len
    assert loaded.num_channels == cfg.num_channels
    assert loaded.channel_increment == cfg.channel_increment
    assert loaded.transformer_layers == cfg.transformer_layers
    assert loaded.num_q_heads == cfg.num_q_heads
    assert loaded.num_kv_heads == cfg.num_kv_heads
    assert loaded.qk_head_dim == cfg.qk_head_dim
    assert loaded.v_head_dim == cfg.v_head_dim
    assert loaded.pair_channels == cfg.pair_channels
    assert loaded.pair_heads == cfg.pair_heads
    assert loaded.sync_bn == cfg.sync_bn
    assert loaded.dtype_policy == cfg.dtype_policy
    assert loaded.metadata.get_heads() == cfg.metadata.get_heads()
    assert torch.equal(
        loaded.metadata.metadata["heads"]["rna_seq"]["means"],
        cfg.metadata.metadata["heads"]["rna_seq"]["means"],
    )
    assert torch.equal(
        loaded.metadata.metadata["heads"]["rna_seq"]["track_mask"],
        cfg.metadata.metadata["heads"]["rna_seq"]["track_mask"],
    )

    loaded_model = AlphaGenome(loaded)
    loaded_model.load_state_dict(model.state_dict())
    batch = synthetic_batch(metadata, batch_size=1, seq_len=model.max_seq_len)

    model.zero_grad(set_to_none=True)
    result = model.loss(batch)
    result.total.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    loaded_model.zero_grad(set_to_none=True)
    loaded_result = loaded_model.loss(batch)
    loaded_result.total.backward()
    loaded_grads = {
        name: param.grad.detach().clone()
        for name, param in loaded_model.named_parameters()
        if param.grad is not None
    }

    assert torch.allclose(result.total, loaded_result.total)
    result_leaves = dict(result.tree.iter_leaves())
    loaded_leaves = dict(loaded_result.tree.iter_leaves())
    assert result_leaves.keys() == loaded_leaves.keys()
    for path, leaf in result_leaves.items():
        assert torch.allclose(leaf.value, loaded_leaves[path].value)
    assert grads.keys() == loaded_grads.keys()
    for key, value in grads.items():
        assert torch.allclose(value, loaded_grads[key])
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)


def test_model_save_load_roundtrip(tmp_path):
    metadata = synthetic_metadata(
        (HeadName.RNA_SEQ,),
        organisms=("human",),
        num_tracks=2,
    )
    model = small_alphagenome(
        metadata,
        max_seq_len=2_048,
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
    )
    model.eval()

    model_dir = tmp_path / "nested" / "model"
    model.save(model_dir)
    loaded_model = AlphaGenome.load(model_dir)
    loaded_model.eval()

    assert (model_dir / "config.json").is_file()
    assert (model_dir / "metadata.pt").is_file()
    assert (model_dir / "model.pt").is_file()

    model_config = model.cfg.to_dict().copy()
    loaded_config = loaded_model.cfg.to_dict().copy()
    model_config.pop("metadata")
    loaded_config.pop("metadata")
    assert loaded_config == model_config
    assert next(loaded_model.parameters()).device == torch.device("cpu")
    assert loaded_model.metadata.get_organisms() == model.metadata.get_organisms()
    assert loaded_model.metadata.get_heads() == model.metadata.get_heads()
    torch.testing.assert_close(
        loaded_model.metadata.metadata["heads"]["rna_seq"]["means"],
        model.metadata.metadata["heads"]["rna_seq"]["means"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        loaded_model.metadata.metadata["heads"]["rna_seq"]["track_mask"],
        model.metadata.metadata["heads"]["rna_seq"]["track_mask"],
        rtol=0,
        atol=0,
    )

    model_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    assert loaded_state.keys() == model_state.keys()
    for key, value in model_state.items():
        torch.testing.assert_close(loaded_state[key], value, rtol=0, atol=0)

    with torch.inference_mode():
        expected = model.predict(DNA_SEQUENCE)
        actual = loaded_model.predict(DNA_SEQUENCE)
    assert_predictions_close(actual, expected)


class TestStateDictContracts:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_state_dict_reload_preserves_parameters_and_buffers(self, tmp_path):
        torch.manual_seed(42)
        source, _ = build_small_model()
        torch.manual_seed(999)
        target, _ = build_small_model()

        path = tmp_path / "model.pt"
        torch.save(source.state_dict(), path)
        target.load_state_dict(torch.load(path, weights_only=True))

        for (name, expected), (_, actual) in zip(
            source.named_parameters(), target.named_parameters()
        ):
            torch.testing.assert_close(
                actual, expected, atol=0, rtol=0, msg=f"parameter {name}"
            )

        persistent = set(source.state_dict())
        source_buffers = dict(source.named_buffers())
        for name, actual in target.named_buffers():
            if name in persistent:
                torch.testing.assert_close(
                    actual,
                    source_buffers[name],
                    atol=0,
                    rtol=0,
                    msg=f"buffer {name}",
                )

    def test_embeddings_are_identical_after_state_dict_reload(self, tmp_path):
        torch.manual_seed(42)
        source, batch = build_small_model_with_batch()
        source.eval()
        with torch.no_grad():
            before = source.embed(batch)

        path = tmp_path / "model.pt"
        torch.save(source.state_dict(), path)

        torch.manual_seed(999)
        target, _ = build_small_model()
        target.load_state_dict(torch.load(path, weights_only=True))
        target.eval()
        with torch.no_grad():
            after = target.embed(batch)

        for name in ("embeddings_1bp", "embeddings_128bp", "embeddings_pair"):
            torch.testing.assert_close(
                getattr(after, name), getattr(before, name), atol=0, rtol=0
            )

    def test_trunk_loads_when_head_keys_are_missing(self):
        torch.manual_seed(42)
        source, _ = build_small_model()
        trunk = {
            key: value
            for key, value in source.state_dict().items()
            if not key.startswith("_heads.")
        }

        torch.manual_seed(999)
        target, _ = build_small_model()
        missing, unexpected = target.load_state_dict(trunk, strict=False)

        assert all(key.startswith("_heads.") for key in missing)
        assert not unexpected
        source_parameters = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            if not name.startswith("_heads."):
                torch.testing.assert_close(
                    parameter,
                    source_parameters[name],
                    atol=0,
                    rtol=0,
                    msg=f"trunk parameter {name}",
                )

    def test_expected_state_dict_prefixes_are_present(self):
        model, _ = build_small_model()
        keys = set(model.state_dict())
        for prefix in (
            "sequence_encoder.",
            "transformer_tower.",
            "sequence_decoder.",
            "_heads.",
        ):
            assert any(key.startswith(prefix) for key in keys), prefix

    def test_metadata_buffers_are_not_in_state_dict(self):
        model, _ = build_small_model()
        persistent = set(model.state_dict())
        non_persistent = [
            name
            for name, _ in model.named_buffers()
            if name not in persistent
        ]
        assert non_persistent
        assert all(
            name.rsplit(".", 1)[1]
            in ("_track_means", "_track_mask", "_tissue_mask")
            for name in non_persistent
        ), non_persistent
