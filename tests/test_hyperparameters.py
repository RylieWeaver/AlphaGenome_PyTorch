import pytest
import torch

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig
from alphagenome_pytorch.schemas import DataBatch

from .helpers import (
    assert_finite_scalars,
    build_metadata,
    build_small_model,
    make_means,
    random_dna_batch,
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
    metadata = build_metadata({"masked_language_modeling": {}}, organisms=("human",))
    model = build_small_model(metadata, **cfg_overrides)

    batch_size = 1
    seq_len = model.input_seq_len
    organism_index = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.randint(0, 5, (batch_size, seq_len), dtype=torch.long)
    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        mlm=labels,
    )

    total_loss, scalars, _ = model.loss(batch)
    total_loss.backward()

    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )

    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
    assert grad_norm > 0.0


def test_config_save_load_roundtrip(tmp_path):
    metadata = build_metadata(
        {
            "masked_language_modeling": {},
            "rna_seq": {
                "num_tracks": [4],
                "means": make_means([4]),
            },
        },
        organisms=("human",),
    )
    cfg = AlphaGenomeConfig(
        input_seq_len=2048,
        num_channels=80,
        channel_increment=10,
        transformer_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        qk_head_dim=20,
        v_head_dim=20,
        pair_channels=16,
        pair_heads=8,
        metadata=metadata,
    )

    cfg_path = tmp_path / "config.json"
    metadata_path = tmp_path / "metadata.pt"
    cfg.save(cfg_path, metadata_path)
    loaded = AlphaGenomeConfig.load(cfg_path, metadata_path)

    assert loaded.input_seq_len == cfg.input_seq_len
    assert loaded.num_channels == cfg.num_channels
    assert loaded.channel_increment == cfg.channel_increment
    assert loaded.transformer_layers == cfg.transformer_layers
    assert loaded.num_q_heads == cfg.num_q_heads
    assert loaded.num_kv_heads == cfg.num_kv_heads
    assert loaded.qk_head_dim == cfg.qk_head_dim
    assert loaded.v_head_dim == cfg.v_head_dim
    assert loaded.pair_channels == cfg.pair_channels
    assert loaded.pair_heads == cfg.pair_heads
    assert loaded.metadata.get_heads() == cfg.metadata.get_heads()
    assert torch.equal(
        loaded.metadata.metadata["heads"]["rna_seq"]["means"],
        cfg.metadata.metadata["heads"]["rna_seq"]["means"],
    )
    assert torch.equal(
        loaded.metadata.metadata["heads"]["rna_seq"]["track_mask"],
        cfg.metadata.metadata["heads"]["rna_seq"]["track_mask"],
    )


def test_config_save_load_allows_model_reconstruction(tmp_path):
    metadata = build_metadata(
        {"masked_language_modeling": {}},
        organisms=("human",),
    )
    cfg = AlphaGenomeConfig(
        input_seq_len=2048,
        num_channels=64,
        transformer_layers=1,
        metadata=metadata,
    )

    cfg_path = tmp_path / "config.json"
    metadata_path = tmp_path / "metadata.pt"
    cfg.save(cfg_path, metadata_path)
    loaded_cfg = AlphaGenomeConfig.load(cfg_path, metadata_path)
    model = AlphaGenome(loaded_cfg)

    batch_size = 1
    seq_len = model.input_seq_len
    organism_index = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.randint(0, 5, (batch_size, seq_len), dtype=torch.long)
    batch = DataBatch(
        dna_sequence=random_dna_batch(batch_size, seq_len),
        organism_index=organism_index,
        mlm=labels,
    )

    total_loss, scalars, _ = model.loss(batch)
    assert torch.isfinite(total_loss)
    assert_finite_scalars(scalars)
