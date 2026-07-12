# Shared model builders for the tests in this package. Not adapted from any
# upstream source.

from alphagenome_pt import (
    HeadName,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)

ALL_HEADS = tuple(HeadName)

# The junction head is the one head that legitimately receives no gradient from a
# randomly initialised model: it only predicts at positions the classification
# head flags as splice sites, and an untrained classifier flags none. genomicsxai/alphagenome-pytorch
# excludes it from its gradient sweep for the same reason.
HEADS_WITHOUT_JUNCTION = tuple(
    h for h in HeadName if h is not HeadName.SPLICE_SITES_JUNCTION
)


def build(heads=ALL_HEADS, **cfg):
    metadata = synthetic_metadata(heads)
    return small_alphagenome(metadata, **cfg), metadata


def build_with_batch(heads=ALL_HEADS, batch_size=2, **cfg):
    model, metadata = build(heads, **cfg)
    # num_splice_sites must match the model, or the junction head's loss will
    # raise on a shape mismatch. synthetic_batch defaults to 2; the model uses 8.
    batch = synthetic_batch(
        metadata,
        batch_size=batch_size,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )
    return model, batch
