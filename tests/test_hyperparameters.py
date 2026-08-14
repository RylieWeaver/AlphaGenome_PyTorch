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
from .helpers import DNA_SEQUENCE, assert_finite_metric_tree, assert_predictions_close


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
