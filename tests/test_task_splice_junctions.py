# External
import pytest
import torch

# Internal
import alphagenome_pt.losses as loss_functions
from alphagenome_pt import (
    HeadName,
    synthetic_batch,
    synthetic_metadata,
    small_alphagenome,
)
from .helpers import assert_finite_metric_tree


def test_splice_junctions_head():
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    model = small_alphagenome(metadata)

    batch = synthetic_batch(
        metadata,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )

    result = model.loss(
        batch,
        return_predictions=True,
    )

    assert result.predictions is not None
    assert torch.isfinite(result.total)
    assert_finite_metric_tree(result.tree)
    assert "splice_sites_junction" in result.predictions
    assert (
        result.predictions["splice_sites_junction"]["predictions"].shape
        == batch.splice_junctions.shape
    )
    assert torch.equal(
        result.predictions["splice_sites_junction"]["splice_site_positions"],
        batch.splice_site_positions,
    )


def test_splice_junction_loss_tree_combines_loss_components(monkeypatch):
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    model = small_alphagenome(metadata)
    batch = synthetic_batch(
        metadata,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )

    components = {
        "total_count": torch.tensor(2.0),
        "ratio": torch.tensor(3.0),
    }
    monkeypatch.setattr(
        loss_functions,
        "poisson_loss",
        lambda **_: components["total_count"],
    )
    monkeypatch.setattr(
        loss_functions,
        "cross_entropy_loss",
        lambda **_: components["ratio"],
    )

    result = model.loss(batch)

    # Both the acceptor and donor contribute one count and one ratio loss.
    # The official head weight applies to every term, while its internal
    # total-count weight applies only to the count terms.
    expected = 0.2 * 2 * (
        components["ratio"] + 0.2 * components["total_count"]
    )
    torch.testing.assert_close(
        result.tree.total_loss("splice_sites_junction"),
        expected,
    )


def test_explicit_positions_do_not_require_classification_predictions():
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    metadata.metadata["heads"][
        HeadName.SPLICE_SITES_CLASSIFICATION.value
    ]["enabled"] = False
    model = small_alphagenome(metadata)
    # NOTE: Synthetic batch will include the splice site positions
    # which are passed and used by the splice junctions head.
    batch = synthetic_batch(
        metadata,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )

    result = model.loss(batch, return_predictions=True)

    assert result.predictions is not None
    assert HeadName.SPLICE_SITES_JUNCTION.value in result.predictions
    assert HeadName.SPLICE_SITES_CLASSIFICATION.value not in result.predictions


@pytest.mark.parametrize(
    "junction_mask",
    (
        torch.tensor([True, False, True, False]).reshape(1, 1, 1, 4),
        torch.tensor(
            [True, False, True, False, False, True, False, True]
        ).reshape(1, 1, 1, 8),
    ),
)
def test_splice_junction_target_mask_does_not_change_prediction_mask(junction_mask):
    head_name = HeadName.SPLICE_SITES_JUNCTION.value
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    model = small_alphagenome(metadata)
    batch = synthetic_batch(
        metadata,
        batch_size=1,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )

    batch.splice_junctions_mask = junction_mask
    masked_predictions = model.predict(batch)

    batch.splice_junctions_mask = torch.ones_like(junction_mask)
    unmasked_predictions = model.predict(batch)

    assert torch.equal(
        masked_predictions[head_name]["splice_junction_mask"],
        unmasked_predictions[head_name]["splice_junction_mask"],
    )


def test_splice_junction_metadata_mask_limits_predictions():
    head_name = HeadName.SPLICE_SITES_JUNCTION.value
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    metadata_tissue_mask = torch.tensor([True, True, False, True])
    metadata.metadata["heads"][head_name]["tissue_mask"][0] = (
        metadata_tissue_mask
    )
    model = small_alphagenome(metadata)
    batch = synthetic_batch(
        metadata,
        batch_size=1,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )
    batch.splice_junctions_mask = None

    prediction_mask = model.predict(batch)[head_name]["splice_junction_mask"]
    metadata_track_mask = torch.cat(
        [metadata_tissue_mask, metadata_tissue_mask]
    )
    invalid_tracks = ~metadata_track_mask

    assert not prediction_mask[..., invalid_tracks].any()


@pytest.mark.parametrize(
    ("mask_shape", "message"),
    (
        ((1, 1, 1, 3), "one channel per tissue"),
        ((1, 1, 4), "must have shape"),
    ),
)
def test_splice_junction_mask_rejects_invalid_shapes(mask_shape, message):
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    model = small_alphagenome(metadata)
    batch = synthetic_batch(
        metadata,
        batch_size=1,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )
    batch.splice_junctions_mask = torch.ones(mask_shape, dtype=torch.bool)

    with pytest.raises(ValueError, match=message):
        model.loss(batch)


@pytest.mark.parametrize(
    ("dtype", "is_integer"),
    (
        (torch.int16, True),
        (torch.int32, True),
        (torch.int64, True),
        (torch.float32, False),
        (torch.bool, False),
    ),
)
def test_splice_positions_require_integer_indices(
    dtype: torch.dtype,
    is_integer: bool,
):
    metadata = synthetic_metadata(
        (HeadName.SPLICE_SITES_CLASSIFICATION, HeadName.SPLICE_SITES_JUNCTION)
    )
    model = small_alphagenome(metadata)
    batch = synthetic_batch(
        metadata,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )
    batch.splice_site_positions = batch.splice_site_positions.to(dtype=dtype)

    if is_integer:
        predictions = model.predict(batch)
        assert (
            predictions["splice_sites_junction"]["splice_site_positions"].dtype
            == torch.long
        )
    else:
        with pytest.raises(TypeError, match="integer indices"):
            model.predict(batch)
