# Losses and the Metric Tree

{bdg-warning}`Evolving API`

`model(batch, mode="loss")` computes the built-in losses for every enabled head and organizes their component terms in a `MetricTree`. It returns a `LossOutput` whose `total` is the scalar to optimize:

```{code-block} python
model.train()
optimizer.zero_grad(set_to_none=True)

output = model(batch, mode="loss")
output.total.backward()
optimizer.step()
```

`model(batch, mode="loss")` and `model.loss(batch)` are equivalent on
`AlphaGenome` and the package's parallel execution wrappers. Both preserve the current
model mode and gradient context.

:::{admonition} Required Batch State
:class: important

At least one head must be enabled, and the `DataBatch` must contain a target
for every enabled head. See
[Target and Mask Fields](../background/data-and-metadata.md#target-and-mask-fields)
for target and mask behavior.
:::


## LossOutput

| Field | Contents |
| --- | --- |
| `total` | Scalar tensor equal to `tree.total_loss()` |
| `tree` | `MetricTree` containing the additive loss terms from every enabled head |
| `predictions` | Prediction dictionary when `return_predictions=True` |
| `embeddings` | `Embeddings` when `return_embeddings=True` |

See [Predictions and Embeddings](predictions-and-embeddings.md) for their contents and shapes:

```{code-block} python
output = model(
    batch,
    mode="loss",
    return_predictions=True,
    return_embeddings=True,
)
predictions = output.predictions  # Nested dictionary of enabled-head outputs
embeddings = output.embeddings    # 1-bp, 128-bp, and pair representations
```

## LossLeaf

A `LossLeaf(numerator, denominator=1.0)` stores the sum and count for one mean
loss term. Combining these across ranks gives the correct global mean, even
when ranks have different numbers of valid targets.

| Member | Meaning |
| --- | --- |
| `numerator` | Sum of losses over valid targets, including any term weighting |
| `denominator` | Number of valid targets included in that sum. Does not track gradients |
| `value` | Scalar tensor equal to `numerator / denominator` when the denominator is positive, otherwise zero |

The numerator retains its autograd history. The denominator is converted to
the numerator's device and dtype, then detached. For example:

```{code-block} python
import torch
from alphagenome_pt import LossLeaf

errors = torch.tensor([1.0, 3.0], requires_grad=True)
leaf = LossLeaf(errors.sum(), errors.numel())

print(leaf.numerator.item())    # 4.0
print(leaf.denominator.item())  # 2.0
print(leaf.value.item())        # 2.0
leaf.value.backward()
```

`leaf.add(other)` sums both numerators and both denominators and detaches
by default (pass `detach=False` to retain the numerator computation graph).
As a result, losses with different counts combine as
`(numerator_a + numerator_b) / (denominator_a + denominator_b)` as desired
for the true global mean.

## MetricTree

Loss mode returns a `LossOutput` whose `tree` has one top-level branch for
each enabled head. The structure beneath each head depends on its loss, and
every path ends in a `LossLeaf`. Tree totals sum the `.value` of each
selected leaf.

:::{note}
When an availability mask fully excludes a loss term, its leaf has zero
numerator/denominator/value. We keep that leaf in the tree so paths stay
consistent across batches and ranks. Other ranks may still contribute valid
values when the statistics are reduced.
:::

:::{container} long-table

| Method | Behavior |
| --- | --- |
| `total_loss()` | Returns the total anywhere in the tree, from the root to an individual leaf |
| `head_loss_totals()` | Returns a dictionary of totals by head |
| `iter_leaves()` | Iterates over `(path, LossLeaf)` pairs |
| `leaf_paths()` | Returns all leaf paths in canonical order |
| `to_dict()` | Converts the hierarchy to nested dictionaries of tensors |
| `detach()` | Returns a new tree without autograd history |
| `add()` | Sums corresponding leaf numerators and denominators and detaches by default. Pass `detach=False` to retain computation graphs |
| `distributed_reduce(group)` | Returns a new tree with corresponding leaf numerators and denominators summed across the process group, preserving numerator autograd history. |

:::

### Loss Totals

`total_loss()` returns the total for the whole tree or any branch or leaf:

```{code-block} python
tree = output.tree

model_total = tree.total_loss()
rna_total = tree.total_loss("rna_seq")
rna_128bp_total = tree.total_loss("rna_seq", "128bp")
positional = tree.total_loss("rna_seq", "128bp", "positional")
```

`head_loss_totals()` returns one total for every enabled head:

```{code-block} python
head_totals = tree.head_loss_totals()  # dict[str, torch.Tensor]
```

### Conversion to Dictionary

`to_dict()` returns nested dictionaries of each leaf's `.value`, preserving
autograd history:

```{code-block} python
values = tree.to_dict()
positional = values["rna_seq"]["128bp"]["positional"]
```

### Detach and Accumulate

`add()` sums each leaf's numerator and denominator across two batches and 
detaches by default. `detach()` removes autograd history from an individual
tree. Use them to accumulate logging statistics without retaining computation
graphs:

```{code-block} python
accumulated = None

for batch in batches:
    tree = model(batch, mode="loss").tree
    accumulated = (
        tree.detach()
        if accumulated is None
        else accumulated.add(tree)
    )

mean_head_losses = accumulated.head_loss_totals()
```

:::{admonition} Aggregation Semantics
:class: note

Each accumulated leaf's value is its summed numerator divided by its summed
denominator. This preserves each loss term's weighting when valid-target
counts differ between batches.
:::

### From Existing Predictions

If predictions are already available, compute their losses without another model pass:

```{code-block} python
batch = model.as_data_batch(batch)
predictions = model(batch, mode="predict")
tree = model.metric_tree_from_predictions(predictions, batch)
```

`metric_tree_from_predictions()` does not run the model or prepare the batch.
It computes each enabled head's losses from the supplied predictions and the
normalized batch's organism index, targets, and masks. It does not detach
predictions, so gradients are retained when the predictions come from a
gradient-tracked model call.

### Tree Structure

Only enabled heads appear as top-level branches. Inspect leaf paths in
canonical sorted order with `iter_leaves()` or `leaf_paths()`:

```{code-block} python
for path, leaf in tree.iter_leaves():
    print(path, leaf.value)

paths = tree.leaf_paths()
```

Every path ends in a `LossLeaf`, but the number and names of intermediate
branches depend on the head. Expand the complete reference when you need an
exact built-in path.

:::{dropdown} Complete Built-In Branch Structure

Names separated by `|` share the same structure.

```text
atac | dnase | procap | cage | rna_seq
├── 1bp
│   ├── positional
│   └── total_count
└── 128bp
    ├── positional
    └── total_count

chip_tf | chip_histone
└── 128bp
    ├── positional
    └── total_count

contact_maps
└── mse

splice_sites_classification
└── cross_entropy

splice_sites_usage
└── binary_cross_entropy

splice_sites_junction
├── ratios
│   ├── acceptor
│   └── donor
└── total_counts
    ├── acceptor
    └── donor

masked_language_modeling
└── cross_entropy
```

:::

## Parallelism

The package's DDP/FSDP wrappers, constructed with `distribute_alphagenome()`,
support both `model.loss(batch)` and `model(batch, mode="loss")`. Their built-in
loss statistics are already reduced over `model.parallel_context.loss_group`,
which matches the wrapper's gradient-averaging group.

With SP, predictions and embeddings remain local to each shard. Unequal
sequence lengths or valid-target counts across DP replicas are handled by
summing numerators and denominators before dividing.

```{code-block} python
output = parallel_model(batch, mode="loss")
output.total.backward()
```

See [Parallelism](parallelism.md) for setup, data distribution, output shapes,
BatchNorm behavior, and checkpoint saving.
