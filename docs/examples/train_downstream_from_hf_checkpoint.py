# External
import torch

# Internal
from alphagenome_pt import HeadName, deepmind_model, synthetic_batch, synthetic_metadata
from utils import example_train, resolve_device


"""
Fine-tune custom downstream heads from the converted DeepMind checkpoint.
The checkpoint trunk is loaded, but the heads are skipped.
"""


if __name__ == "__main__":
    device = resolve_device()

    metadata = synthetic_metadata(
        heads=(
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

    model = deepmind_model(
        device=device,
        metadata=metadata.metadata,
        load_state=True,
        heads=False,
    )

    batch = synthetic_batch(
        model.metadata,
        batch_size=1,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    example_train(model, [batch], optimizer, steps=1000)

    print("SUCCESS")
