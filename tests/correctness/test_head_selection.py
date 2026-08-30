# Adapted from genomicsxai/alphagenome-pytorch, tests/unit/test_model_heads.py
# (Apache-2.0). No golden numbers.
#
# Deviations from the source file:
#   - genomicsxai/alphagenome-pytorch selects heads with a heads=("atac",) argument to forward(); alphagenome_pt
#     selects them through the metadata "enabled" flag, so these are rewritten
#     against that API
#   - alphagenome_pt already has tests/test_head_enabled_flag.py covering part of this;
#     these are the cases it does not cover

import pytest
import torch

from alphagenome_pt import HeadName, small_alphagenome, synthetic_batch, synthetic_metadata

from .helpers import ALL_HEADS, build_with_batch

pytestmark = pytest.mark.integration


class TestHeadSelection:
    def test_only_requested_heads_are_built(self):
        metadata = synthetic_metadata((HeadName.ATAC, HeadName.DNASE))
        model = small_alphagenome(metadata)
        batch = synthetic_batch(metadata, seq_len=model.max_seq_len)
        predictions = model(batch)

        assert "atac" in predictions
        assert "dnase" in predictions
        # Anything not asked for must be absent, not silently computed.
        assert "cage" not in predictions
        assert "rna_seq" not in predictions

    def test_single_head_model_has_one_head(self):
        metadata = synthetic_metadata((HeadName.ATAC,))
        model = small_alphagenome(metadata)
        predictions = model(
            synthetic_batch(metadata, seq_len=model.max_seq_len)
        )
        assert list(predictions) == ["atac"]

    def test_all_heads_can_be_requested(self):
        model, batch = build_with_batch(ALL_HEADS)
        predictions = model(batch)
        expected = {h.value for h in ALL_HEADS}
        assert set(predictions) == expected

    def test_selecting_fewer_heads_makes_a_smaller_model(self):
        # If head selection did not actually skip construction, the parameter
        # count would not move and the memory saving would be a lie.
        one, _ = build_with_batch((HeadName.ATAC,))
        many, _ = build_with_batch(ALL_HEADS)
        assert (sum(p.numel() for p in one.parameters())
                < sum(p.numel() for p in many.parameters()))

    def test_unknown_head_name_is_rejected(self):
        with pytest.raises((KeyError, ValueError, AttributeError)):
            synthetic_metadata(("not_a_real_head",))
