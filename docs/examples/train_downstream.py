# Internal
from alphagenome_pt import HeadName, small_alphagenome, synthetic_metadata
from utils import example_train_val_test, resolve_device


"""
Train AlphaGenome downstream heads on synthetic data.
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
    model = small_alphagenome(
        metadata,
        num_channels=96,
    ).to(device)

    example_train_val_test(
        metadata=metadata,
        model=model,
        max_steps=1000,
        lr=1e-5,
    )
    print("SUCCESS")
