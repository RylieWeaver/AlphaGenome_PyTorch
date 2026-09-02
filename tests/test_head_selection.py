# Adapted from genomicsxai/alphagenome-pytorch, tests/unit/test_model_heads.py
# (Apache-2.0). No golden numbers.
#
# Deviations from the source file:
#   - genomicsxai/alphagenome-pytorch selects heads with a heads=("atac",) argument to forward(); alphagenome_pt
#     selects them through the metadata "enabled" flag, so these are rewritten
#     against that API
#   - enabled-flag coverage originally lived in tests/test_head_enabled_flag.py
#     and is grouped into TestHeadSelection here

import pytest
import torch

from alphagenome_pt import HeadName, small_alphagenome, synthetic_batch, synthetic_metadata

from .helpers import (
    ALL_HEADS,
    assert_finite_metric_tree,
    build_small_model_with_batch,
)

class TestHeadSelection:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_enabled_flag_skips_disabled_heads(self):
        metadata = synthetic_metadata((HeadName.RNA_SEQ, HeadName.ATAC))
        heads = metadata.metadata["heads"]
        heads[HeadName.RNA_SEQ.value]["enabled"] = False
        heads[HeadName.ATAC.value]["enabled"] = True
        model = small_alphagenome(metadata)

        batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
        result = model.loss(batch, return_predictions=True)

        assert result.predictions is not None
        assert torch.isfinite(result.total)
        assert_finite_metric_tree(result.tree)
        assert HeadName.ATAC.value in result.predictions
        assert HeadName.RNA_SEQ.value not in result.predictions
        assert set(result.tree.head_loss_totals()) == {HeadName.ATAC.value}

    def test_only_requested_heads_are_built(self):
        metadata = synthetic_metadata((HeadName.ATAC, HeadName.DNASE))
        model = small_alphagenome(metadata)
        batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
        predictions = model(batch)

        assert HeadName.ATAC.value in predictions
        assert HeadName.DNASE.value in predictions
        # Anything not asked for must be absent, not silently computed.
        assert HeadName.CAGE.value not in predictions
        assert HeadName.RNA_SEQ.value not in predictions

    def test_single_head_model_has_one_head(self):
        metadata = synthetic_metadata((HeadName.ATAC,))
        model = small_alphagenome(metadata)
        predictions = model(
            synthetic_batch(metadata, seq_len=model.max_seq_len)
        )
        assert list(predictions) == [HeadName.ATAC.value]

    def test_all_heads_can_be_requested(self):
        model, batch = build_small_model_with_batch(ALL_HEADS)
        predictions = model(batch)
        expected = {h.value for h in ALL_HEADS}
        assert set(predictions) == expected

    def test_selecting_fewer_heads_makes_a_smaller_model(self):
        # If head selection did not actually skip construction, the parameter
        # count would not move and the memory saving would be a lie.
        one, _ = build_small_model_with_batch((HeadName.ATAC,))
        many, _ = build_small_model_with_batch(ALL_HEADS)
        assert (sum(p.numel() for p in one.parameters())
                < sum(p.numel() for p in many.parameters()))

    def test_unknown_head_name_is_rejected(self):
        with pytest.raises((KeyError, ValueError, AttributeError)):
            synthetic_metadata(("not_a_real_head",))
