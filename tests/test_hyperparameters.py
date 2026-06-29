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
from .helpers import assert_finite_scalars


@pytest.mark.parametrize(
    "cfg_overrides",
    [
        {
            "num_channels": 48,
            "transformer_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 1,
            "qk_head_dim": 12,
            "v_head_dim": 12,
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

    total_loss, scalars, _ = model.loss(batch)
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.parameters()
    )


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
    loss, scalars, _ = model.loss(batch)
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    loaded_model.zero_grad(set_to_none=True)
    loaded_loss, loaded_scalars, _ = loaded_model.loss(batch)
    loaded_loss.backward()
    loaded_grads = {
        name: param.grad.detach().clone()
        for name, param in loaded_model.named_parameters()
        if param.grad is not None
    }

    assert torch.allclose(loss, loaded_loss)
    assert scalars.keys() == loaded_scalars.keys()
    for key, value in scalars.items():
        assert torch.allclose(value, loaded_scalars[key])
    assert grads.keys() == loaded_grads.keys()
    for key, value in grads.items():
        assert torch.allclose(value, loaded_grads[key])
    assert torch.isfinite(loss)
    assert_finite_scalars(scalars)
