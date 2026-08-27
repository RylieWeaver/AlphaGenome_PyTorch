# External
import torch

# Internal
from alphagenome_pt import MetricTree


DNA_SEQUENCE = "ACGT" * 512


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
