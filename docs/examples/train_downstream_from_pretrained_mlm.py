# External
import torch

# Internal
from alphagenome_pt import (
    AlphaGenome,
    AlphaGenomeConfig,
    HeadName,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)
from utils import example_train, resolve_device


"""
Pretrain the MLM head on synthetic data, then fine-tune downstream heads.
"""


def set_enabled_heads(model: AlphaGenome, enabled_heads: set[str]) -> None:
    for head_name in model.metadata.get_heads():
        model.metadata.metadata["heads"][head_name]["enabled"] = head_name in enabled_heads


if __name__ == "__main__":
    device = resolve_device()

    metadata = synthetic_metadata(
        heads=(
            HeadName.MASKED_LANGUAGE_MODELING,
            HeadName.RNA_SEQ,
            HeadName.CAGE,
            HeadName.ATAC,
            HeadName.DNASE,
            HeadName.PROCAP,
            HeadName.CHIP_TF,
            HeadName.CHIP_HISTONE,
            HeadName.CONTACT_MAPS,
            HeadName.SPLICE_SITES_CLASSIFICATION,
            HeadName.SPLICE_SITES_USAGE,
            HeadName.SPLICE_SITES_JUNCTION,
        ),
        num_organisms=2,
        num_tracks=8,
    )
    model = small_alphagenome(
        metadata,
        num_channels=96,
    ).to(device)
    batch = synthetic_batch(metadata, seq_len=model.max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    print("Starting MLM pretraining...")
    set_enabled_heads(model, {HeadName.MASKED_LANGUAGE_MODELING.value})
    example_train(model, [batch], optimizer, steps=1000)

    model.cfg.save(
        cfg_path="AG_mlm_pretrained_config.json",
        metadata_path="AG_mlm_pretrained_metadata.pt",
    )
    torch.save(model.state_dict(), "AG_mlm_pretrained.pt")
    print("Finished MLM pretraining. Starting downstream fine-tuning...")

    loaded_cfg = AlphaGenomeConfig.load(
        cfg_path="AG_mlm_pretrained_config.json",
        metadata_path="AG_mlm_pretrained_metadata.pt",
    )
    model = AlphaGenome(loaded_cfg).to(device)
    model.load_state_dict(torch.load("AG_mlm_pretrained.pt", map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    downstream_heads = set(model.metadata.get_heads()) - {HeadName.MASKED_LANGUAGE_MODELING.value}
    set_enabled_heads(model, downstream_heads)
    batch = synthetic_batch(
        model.metadata,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    ).to(device)
    example_train(model, [batch], optimizer, steps=1000)

    print("SUCCESS")
