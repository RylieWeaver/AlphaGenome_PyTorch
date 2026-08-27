# External
import pytest
import torch

# Internal
from alphagenome_pt import HeadName, small_alphagenome, synthetic_metadata


HEADS_WITH_METADATA_MASKS = tuple(
    head
    for head in HeadName
    if head is not HeadName.MASKED_LANGUAGE_MODELING
)


@pytest.mark.parametrize("head", HEADS_WITH_METADATA_MASKS)
def test_explicit_metadata_mask_reaches_head(head: HeadName):
    metadata = synthetic_metadata((head,), num_organisms=1)
    head_metadata = metadata.metadata["heads"][head.value]
    mask_name = (
        "tissue_mask"
        if head is HeadName.SPLICE_SITES_JUNCTION
        else "track_mask"
    )
    mask_width = torch.as_tensor(head_metadata[mask_name]).shape[-1]
    provided_mask = (torch.arange(mask_width) % 2 == 0).unsqueeze(0)
    head_metadata[mask_name] = provided_mask

    model = small_alphagenome(metadata, max_seq_len=2_048)
    actual_mask = model._heads[head.value]._track_mask
    expected_mask = (
        torch.cat([provided_mask, provided_mask], dim=-1)
        if head is HeadName.SPLICE_SITES_JUNCTION
        else provided_mask
    )

    torch.testing.assert_close(actual_mask, expected_mask)


def test_metadata_can_toggle_configured_heads():
    metadata = synthetic_metadata((HeadName.RNA_SEQ, HeadName.ATAC))
    model = small_alphagenome(metadata, max_seq_len=2_048)
    model.eval()
    heads = model.metadata.metadata["heads"]
    sequence = "ACGT" * 512

    heads[HeadName.RNA_SEQ.value]["enabled"] = False
    heads[HeadName.ATAC.value]["enabled"] = True
    assert set(model.predict(sequence)) == {HeadName.ATAC.value}

    heads[HeadName.RNA_SEQ.value]["enabled"] = True
    heads[HeadName.ATAC.value]["enabled"] = False
    assert set(model.predict(sequence)) == {HeadName.RNA_SEQ.value}
