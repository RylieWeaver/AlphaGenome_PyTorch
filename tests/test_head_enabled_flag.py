# External
import torch

# Internal
from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_metric_tree


def test_enabled_flag_skips_disabled_heads():
    metadata = synthetic_metadata((HeadName.RNA_SEQ, HeadName.ATAC))
    heads = metadata.metadata["heads"]
    heads["rna_seq"]["enabled"] = False
    heads["atac"]["enabled"] = True
    model = small_alphagenome(metadata)

    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert "atac" in result.predictions
    assert "rna_seq" not in result.predictions
    assert set(result.tree.head_loss_totals()) == {"atac"}
