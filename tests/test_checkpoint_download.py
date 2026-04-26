import os

import pytest
import torch

from alphagenome_pt import (
    DEFAULT_ALPHAGENOME_CHECKPOINT,
    download_alphagenome_checkpoint,
    load_alphagenome_checkpoint,
    official_alphagenome_config,
    official_alphagenome_metadata,
)
import alphagenome_pt.checkpoint as checkpoint_module


@pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST") != "1",
    reason="Set ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 to download the HF checkpoint.",
)
def test_download_alphagenome_checkpoint_from_hf(tmp_path):
    pytest.importorskip("huggingface_hub")

    checkpoint_path = download_alphagenome_checkpoint(tmp_path)

    assert checkpoint_path.exists()
    assert checkpoint_path.name == DEFAULT_ALPHAGENOME_CHECKPOINT
    assert checkpoint_path.stat().st_size > 0


class _ToyAlphaGenome(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(2, 2, bias=False)
        self._heads = torch.nn.ModuleDict({
            "toy": torch.nn.Linear(2, 1, bias=False)
        })
        self.custom = torch.nn.Module()
        self.custom.register_buffer("_track_means", torch.ones(2))


def test_load_alphagenome_checkpoint_skips_heads(tmp_path):
    model = _ToyAlphaGenome()
    for param in model.parameters():
        torch.nn.init.zeros_(param)

    checkpoint = {
        "trunk.weight": torch.ones_like(model.trunk.weight),
        "_heads.toy.weight": torch.ones_like(model._heads["toy"].weight),
    }
    checkpoint_path = tmp_path / DEFAULT_ALPHAGENOME_CHECKPOINT
    torch.save(checkpoint, checkpoint_path)

    load_result = load_alphagenome_checkpoint(
        model,
        checkpoint_path,
        heads=False,
    )

    assert torch.equal(model.trunk.weight, checkpoint["trunk.weight"])
    assert torch.equal(model._heads["toy"].weight, torch.zeros_like(model._heads["toy"].weight))
    assert "_heads.toy.weight" not in load_result.missing_keys
    assert "custom._track_means" not in load_result.missing_keys
    assert load_result.unexpected_keys == []


def test_load_alphagenome_checkpoint_downloads_when_missing(tmp_path, monkeypatch):
    model = _ToyAlphaGenome()
    checkpoint = {"trunk.weight": torch.ones_like(model.trunk.weight)}
    downloaded_path = tmp_path / DEFAULT_ALPHAGENOME_CHECKPOINT
    torch.save(checkpoint, downloaded_path)

    def fake_download(output_path, **kwargs):
        return downloaded_path

    monkeypatch.setattr(
        checkpoint_module,
        "download_alphagenome_checkpoint",
        fake_download,
    )

    load_alphagenome_checkpoint(model, tmp_path / "missing.pt", heads=False)

    assert torch.equal(model.trunk.weight, checkpoint["trunk.weight"])


def test_official_alphagenome_config_accepts_custom_metadata():
    metadata = {
        "organisms": ["human"],
        "heads": {"masked_language_modeling": {}},
    }

    cfg = official_alphagenome_config(metadata=metadata)

    assert cfg.max_seq_len == 1_048_576
    assert cfg.metadata.metadata is metadata
    assert "contact_maps" in official_alphagenome_metadata()["heads"]
