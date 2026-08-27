# External
import pytest
import torch

# Internal
from alphagenome_pt import LossLeaf, MetricTree



##### LEAF TESTS #####
def test_loss_leaf_accepts_float_and_tensor():
    assert torch.equal(LossLeaf(1.5).value, torch.tensor(1.5))

    value = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    assert LossLeaf(value).value is value


def test_loss_leaf_rejects_invalid_values():
    with pytest.raises(TypeError, match="torch.Tensor or float"):
        LossLeaf(1)
    for dtype in (torch.int64, torch.bool, torch.complex64):
        with pytest.raises(TypeError, match="floating point"):
            LossLeaf(torch.tensor(1, dtype=dtype))
    with pytest.raises(ValueError, match="scalar tensor"):
        LossLeaf(torch.ones(2))



##### SMALL TREE TESTS #####
def test_metric_tree_rejects_invalid_children():
    with pytest.raises(TypeError, match="children must be a mapping"):
        MetricTree([])
    with pytest.raises(ValueError, match="cannot be empty"):
        MetricTree({})


def test_metric_tree_rejects_invalid_path_names_and_nodes():
    with pytest.raises(ValueError, match="non-empty strings"):
        MetricTree({"": LossLeaf(1.0)}).leaf_paths()
    with pytest.raises(ValueError, match="non-empty strings"):
        MetricTree({"head": LossLeaf(1.0), 1: LossLeaf(2.0)}).leaf_paths()
    with pytest.raises(TypeError, match="LossLeaf objects or mappings"):
        MetricTree({"head": object()}).leaf_paths()


def test_metric_tree_rejects_invalid_paths_and_empty_branches():
    tree = MetricTree({"head": {"term": LossLeaf(torch.tensor(1.0))}})
    with pytest.raises(KeyError, match="No metrics at path"):
        tree.total_loss("missing")
    with pytest.raises(KeyError, match="continues beyond a leaf"):
        tree.total_loss("head", "term", "part")

    empty_tree = MetricTree({"head": {}})
    with pytest.raises(ValueError, match="Metric branch .* is empty"):
        empty_tree.total_loss("head")


def test_metric_tree_keeps_nested_children_and_recomputes_leaf_paths():
    # NOTE: Add a leaf and make sure the tree recomputes
    # .leaf_paths() and .total_loss() correctly.
    tree = MetricTree({
        "head": {
            "second": LossLeaf(2.0),
        },
    })
    branch = tree.children["head"]
    assert isinstance(branch, dict)
    branch["first"] = LossLeaf(1.0)

    assert tree.leaf_paths() == (
        ("head", "first"),
        ("head", "second"),
    )
    assert torch.equal(tree.total_loss(), torch.tensor(3.0))


def test_metric_tree_to_dict_preserves_hierarchy_and_tensors():
    first = torch.tensor(1.0, requires_grad=True)
    tree = MetricTree({
        "rna_seq": {
            "second": LossLeaf(torch.tensor(2.0)),
            "first": LossLeaf(first),
        },
        "contact_maps": {"loss": LossLeaf(torch.tensor(3.0))},
    })

    values = tree.to_dict()

    assert list(values) == ["contact_maps", "rna_seq"]
    rna_values = values["rna_seq"]
    assert isinstance(rna_values, dict)
    assert list(rna_values) == ["first", "second"]
    assert rna_values["first"] is first
    assert values is not tree.children
    assert rna_values is not tree.children["rna_seq"]

    rna_values["first"].backward()
    assert torch.equal(first.grad, torch.tensor(1.0))


def test_metric_tree_total_loss_is_differentiable():
    c1 = 2.0
    c2 = 1.5
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)
    tree = MetricTree({
        "head": {
            "first": LossLeaf(c1 * first),
            "second": LossLeaf(c2 * second),
        },
    })

    tree.total_loss().backward()

    assert torch.equal(first.grad, torch.tensor(c1))
    assert torch.equal(second.grad, torch.tensor(c2))


def test_metric_tree_detach_preserves_values_and_paths():
    value = torch.tensor(2.0, requires_grad=True)
    tree = MetricTree({"head": {"term": LossLeaf(value)}})

    detached = tree.detach()
    original_total = tree.total_loss()
    detached_total = detached.total_loss()

    assert detached.leaf_paths() == tree.leaf_paths()
    assert torch.equal(detached_total, original_total)
    assert original_total.requires_grad
    assert not detached_total.requires_grad


def test_metric_tree_detach_returns_new_children():
    tree = MetricTree({"head": {"term": LossLeaf(1.0)}})

    detached = tree.detach()
    detached_head = detached.children["head"]
    assert isinstance(detached_head, dict)
    detached_head["new"] = LossLeaf(2.0)

    assert detached.leaf_paths() == (
        ("head", "new"),
        ("head", "term"),
    )
    assert tree.leaf_paths() == (("head", "term"),)


def test_metric_tree_add_detaches_by_default_and_can_preserve_gradients():
    c1 = 2.0
    c2 = 1.5
    left_value = torch.tensor(1.0, requires_grad=True)
    right_value = torch.tensor(2.0, requires_grad=True)
    left = MetricTree({"head": {"term": LossLeaf(c1 * left_value)}})
    right = MetricTree({"head": {"term": LossLeaf(c2 * right_value)}})

    detached = left.add(right)
    assert not detached.total_loss().requires_grad
    assert detached.total_loss().grad_fn is None
    assert left_value.grad is None
    assert right_value.grad is None

    attached = left.add(right, detach=False)
    attached.total_loss().backward()

    assert torch.equal(left_value.grad, torch.tensor(c1))
    assert torch.equal(right_value.grad, torch.tensor(c2))


def test_metric_tree_add_returns_new_children():
    left = MetricTree({"head": {"term": LossLeaf(1.0)}})
    right = MetricTree({"head": {"term": LossLeaf(2.0)}})

    result = left.add(right)
    result_head = result.children["head"]
    assert isinstance(result_head, dict)
    result_head["new"] = LossLeaf(3.0)

    assert result.leaf_paths() == (
        ("head", "new"),
        ("head", "term"),
    )
    expected_input_paths = (("head", "term"),)
    assert left.leaf_paths() == expected_input_paths
    assert right.leaf_paths() == expected_input_paths


def test_metric_tree_add_requires_identical_paths():
    left = MetricTree({
        "rna_seq": {
            "1bp": {"loss": LossLeaf(1.0)},
        },
    })
    right = MetricTree({
        "rna_seq": {
            "128bp": {"loss": LossLeaf(2.0)},
        },
    })

    with pytest.raises(ValueError, match="identical paths"):
        left.add(right)


def test_metric_tree_add_rejects_leaf_branch_conflicts():
    left = MetricTree({"head": {"loss": LossLeaf(1.0)}})
    right = MetricTree({"head": {"loss": {"part": LossLeaf(2.0)}}})

    with pytest.raises(ValueError, match="shape conflict"):
        left.add(right)



##### FULL TREE TESTS #####
def _all_head_metric_tree(scale: float = 1.0) -> MetricTree:
    # Use every loss-tree shape currently returned by a model head. Insert
    # branches out of order so this also checks canonical traversal order.

    # NOTE: Scale lets us define separate trees while being able to
    # leverage this helper.
    def leaf(value: float) -> LossLeaf:
        return LossLeaf(scale * value)

    # NOTE: Only one of each head type
    # (RNA-seq is the representative for genome track heads).
    return MetricTree({
        "splice_sites_junction": {
            "total_counts": {
                "donor": leaf(12.0),
                "acceptor": leaf(11.0),
            },
            "ratios": {
                "donor": leaf(10.0),
                "acceptor": leaf(9.0),
            },
        },
        "rna_seq": {
            "1bp": {
                "total_count": leaf(6.0),
                "positional": leaf(5.0),
            },
            "128bp": {
                "total_count": leaf(8.0),
                "positional": leaf(7.0),
            },
        },
        "splice_sites_usage": {"binary_cross_entropy": leaf(4.0)},
        "contact_maps": {"mse": leaf(1.0)},
        "splice_sites_classification": {"cross_entropy": leaf(3.0)},
        "masked_language_modeling": {"cross_entropy": leaf(2.0)},
    })


_ALL_LEAF_PATHS = (
    ("contact_maps", "mse"),
    ("masked_language_modeling", "cross_entropy"),
    ("rna_seq", "128bp", "positional"),
    ("rna_seq", "128bp", "total_count"),
    ("rna_seq", "1bp", "positional"),
    ("rna_seq", "1bp", "total_count"),
    ("splice_sites_classification", "cross_entropy"),
    ("splice_sites_junction", "ratios", "acceptor"),
    ("splice_sites_junction", "ratios", "donor"),
    ("splice_sites_junction", "total_counts", "acceptor"),
    ("splice_sites_junction", "total_counts", "donor"),
    ("splice_sites_usage", "binary_cross_entropy"),
)

def test_metric_tree_leaf_paths_are_canonical():
    assert _all_head_metric_tree().leaf_paths() == _ALL_LEAF_PATHS


_ALL_PREFIX_TOTALS = {
    (): 78.0,
    ("contact_maps",): 1.0,
    ("contact_maps", "mse"): 1.0,
    ("masked_language_modeling",): 2.0,
    ("masked_language_modeling", "cross_entropy"): 2.0,
    ("rna_seq",): 26.0,
    ("rna_seq", "128bp"): 15.0,
    ("rna_seq", "128bp", "positional"): 7.0,
    ("rna_seq", "128bp", "total_count"): 8.0,
    ("rna_seq", "1bp"): 11.0,
    ("rna_seq", "1bp", "positional"): 5.0,
    ("rna_seq", "1bp", "total_count"): 6.0,
    ("splice_sites_usage",): 4.0,
    ("splice_sites_usage", "binary_cross_entropy"): 4.0,
    ("splice_sites_classification",): 3.0,
    ("splice_sites_classification", "cross_entropy"): 3.0,
    ("splice_sites_junction",): 42.0,
    ("splice_sites_junction", "ratios"): 19.0,
    ("splice_sites_junction", "ratios", "acceptor"): 9.0,
    ("splice_sites_junction", "ratios", "donor"): 10.0,
    ("splice_sites_junction", "total_counts"): 23.0,
    ("splice_sites_junction", "total_counts", "acceptor"): 11.0,
    ("splice_sites_junction", "total_counts", "donor"): 12.0,
}
_ALL_PREFIX_PATHS = frozenset(_ALL_PREFIX_TOTALS)


def test_all_head_metric_tree_data_is_complete():
    tree = _all_head_metric_tree()
    leaves = tuple(tree.iter_leaves())
    leaf_paths = tuple(path for path, _ in leaves)
    prefix_paths = {
        path[:depth]
        for path in leaf_paths
        for depth in range(len(path) + 1)
    }

    assert leaf_paths == _ALL_LEAF_PATHS
    assert prefix_paths == _ALL_PREFIX_PATHS
    for prefix, expected in _ALL_PREFIX_TOTALS.items():
        values = [
            leaf.value
            for path, leaf in leaves
            if path[:len(prefix)] == prefix
        ]
        torch.testing.assert_close(sum(values), torch.tensor(expected))


def test_metric_tree_loss_totals():
    tree = _all_head_metric_tree()

    # Include the root, every head, every intermediate branch, and every leaf.
    for prefix, expected in _ALL_PREFIX_TOTALS.items():
        torch.testing.assert_close(
            tree.total_loss(*prefix),
            torch.tensor(expected),
        )

    head_totals = tree.head_loss_totals()
    expected_head_names = tuple(
        sorted(
            prefix[0]
            for prefix in _ALL_PREFIX_TOTALS
            if len(prefix) == 1
        )
    )
    # tuple() over a dictionary returns its keys
    assert tuple(head_totals) == expected_head_names
    for head_name, total in head_totals.items():
        torch.testing.assert_close(
            total,
            torch.tensor(_ALL_PREFIX_TOTALS[(head_name,)]),
        )


def test_metric_tree_add_returns_matching_sum():
    left_scale = 1.5
    right_scale = 2.5
    total_scale = left_scale + right_scale
    left = _all_head_metric_tree(scale=left_scale)
    right = _all_head_metric_tree(scale=right_scale)

    result = left.add(right)

    for tree in (left, right, result):
        assert tree.leaf_paths() == _ALL_LEAF_PATHS

    for prefix, expected_base in _ALL_PREFIX_TOTALS.items():
        expected_left = left_scale * expected_base
        torch.testing.assert_close(
            left.total_loss(*prefix),
            torch.tensor(expected_left),
        )
        expected_right = right_scale * expected_base
        torch.testing.assert_close(
            right.total_loss(*prefix),
            torch.tensor(expected_right),
        )
        expected_total = total_scale * expected_base
        torch.testing.assert_close(
            result.total_loss(*prefix),
            torch.tensor(expected_total),
        )
