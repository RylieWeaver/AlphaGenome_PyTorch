# External
import torch

# Internal
from alphagenome_pt import (
    HeadName,
    MetricTree,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)


DNA_SEQUENCE = "ACGT" * 512
ALL_HEADS = tuple(HeadName)


def seeded_randn(
    shape: tuple[int, ...] = (1, 16, 2, 64),
    seed: int = 42,
) -> torch.Tensor:
    """Return a reproducible normally distributed tensor."""
    torch.manual_seed(seed)
    return torch.randn(shape)


def assert_healthy_gradient(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Assert that a tensor received a finite, non-vanishing gradient."""
    assert tensor.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(tensor.grad).all(), f"{name}: non-finite gradient"
    norm = tensor.grad.norm().item()
    # Treat gradients at or below 1e-12 as vanished in practice.
    assert norm > 1e-12, f"{name}: gradient vanished ({norm})"
    assert norm < 1e8, f"{name}: gradient exploded ({norm})"


def build_small_model(heads=ALL_HEADS, **cfg):
    """Build a small model and its matching synthetic metadata."""
    metadata = synthetic_metadata(heads)
    return small_alphagenome(metadata, **cfg), metadata


def build_small_model_with_batch(heads=ALL_HEADS, batch_size=2, **cfg):
    """Build a small model and a shape-compatible synthetic batch."""
    model, metadata = build_small_model(heads, **cfg)
    batch = synthetic_batch(
        metadata,
        batch_size=batch_size,
        seq_len=model.max_seq_len,
        num_splice_sites=model.num_splice_sites,
    )
    return model, batch


def assert_predictions_close(
    actual: dict[str, dict[str, torch.Tensor]],
    expected: dict[str, dict[str, torch.Tensor]],
) -> None:
    assert actual.keys() == expected.keys()
    for head_name in actual:
        assert actual[head_name].keys() == expected[head_name].keys()
        for output_name in actual[head_name]:
            torch.testing.assert_close(
                actual[head_name][output_name],
                expected[head_name][output_name],
            )


def assert_finite_metric_tree(tree: MetricTree) -> None:
    for path, leaf in tree.iter_leaves():
        assert torch.isfinite(leaf.value), path
