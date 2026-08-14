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

`model(batch, mode="loss")` and `model.loss(batch)` are equivalent for direct
model use. Both preserve the current model mode and gradient context.

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

## MetricTree

Loss mode returns a `LossOutput` whose `tree` has one top-level branch for each enabled head. The structure beneath each head depends on its loss, and every path ends in a `LossLeaf` containing one scalar loss term.

:::{note}
When an availability mask excludes every value from a loss term, its
`LossLeaf` remains in the tree as a zero so paths stay consistent across
batches.
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
| `add()` | Sums trees with matching leaf paths and detaches by default. Pass `detach=False` to retain both computation graphs |

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

`to_dict()` returns new nested dictionaries while preserving the original leaf tensors and their autograd history:

```{code-block} python
values = tree.to_dict()
positional = values["rna_seq"]["128bp"]["positional"]
```

### Detach and Accumulate

`add()` accumulates the complete loss hierarchy across batches without
manually traversing nested dictionaries. `detach()` removes autograd history
from trees retained for reporting. Together, they make it possible to
accumulate logging statistics over multiple batches:

```{code-block} python
accumulated = None
num_batches = 0

for batch in batches:
    tree = model(batch, mode="loss").tree
    accumulated = (
        tree.detach()
        if accumulated is None
        else accumulated.add(tree)
    )
    num_batches += 1

mean_head_losses = {
    head: total / num_batches
    for head, total in accumulated.head_loss_totals().items()
}
```

:::{admonition} Aggregation Semantics
:class: note

Divide accumulated values by `num_batches` for an unweighted mean across
batches. This does not account for sample or valid-element weighting and does
not reduce across distributed processes.

Future `MetricTree` leaves will store numerator/denominator values and support
distributed aggregation, replacing division by `num_batches`.
:::

### From Existing Predictions

If predictions are already available, compute their losses without another model pass:

```{code-block} python
batch = model.as_data_batch(batch)
predictions = model(batch, mode="predict")
tree = model.metric_tree_from_predictions(predictions, batch)
```

`metric_tree_from_predictions()` does not run the model or prepare the batch. It computes each enabled head's losses from the supplied predictions and the normalized batch's organism index, targets, and masks. It does not detach predictions, so gradients are retained when the predictions come from a gradient-tracked model call.

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

During `DistributedDataParallel` (DDP) training, compute loss through `ddp_model(batch, mode="loss")` so the forward pass runs through DDP's hooks:

```{code-block} python
output = ddp_model(batch, mode="loss")
output.total.backward()
```

:::{dropdown} DDP Wrapper Requirement
:color: warning
:icon: alert

Do not call `ddp_model.module.loss(batch)`. Bypassing the wrapper can cause
reducer errors when `find_unused_parameters=True`.
:::

Other forms of parallelism are not currently offered, but sequence parallelism
is in development.
