# Predictions and Embeddings

AlphaGenome can return prediction-head outputs, internal representations, or
both from the same input.

| Output | Calls | Contents |
| --- | --- | --- |
| [Predictions](#predictions) | `model.predict(data)` or `model(data)` | Nested dictionary of enabled-head outputs |
| [Embeddings](#embeddings) | `model.embed(data)` or `model(data, mode="embed")` | 1-bp, 128-bp, and pair representations |

## Return Behavior

| Call | Returns | Gradient Tracking |
| --- | --- | --- |
| `model(data)` | Prediction dictionary | Current context |
| `model.predict(data)` | Prediction dictionary | Disabled |
| `model.embed(data)` or `model(data, mode="embed")` | `Embeddings` | Current context |
| `model(data, return_embeddings=True)` | `(predictions, embeddings)` | Current context |
| `model.predict(data, return_embeddings=True)` | `(predictions, embeddings)` | Disabled |

::::{tab-set}
:::{tab-item} Direct Predictions
```python
data = "ACGT" * 2048
predictions = model(data, organism_index=0)
```
:::
:::{tab-item} Inference Predictions
```python
data = "ACGT" * 2048
predictions = model.predict(data, organism_index=0)
```
:::
:::{tab-item} Direct Embeddings
```python
data = "ACGT" * 2048
embeddings = model.embed(data, organism_index=0)
```
:::
:::{tab-item} Embedding Mode
```python
data = "ACGT" * 2048
embeddings = model(data, organism_index=0, mode="embed")
```
:::
:::{tab-item} Direct Predictions and Embeddings
```python
data = "ACGT" * 2048
predictions, embeddings = model(
    data,
    organism_index=0,
    return_embeddings=True,
)
```
:::
:::{tab-item} Inference Predictions and Embeddings
```python
data = "ACGT" * 2048
predictions, embeddings = model.predict(
    data,
    organism_index=0,
    return_embeddings=True,
)
```
:::
::::

:::{important}
None of these calls changes model mode. Call `model.eval()` explicitly for
inference. `model.predict()` disables gradient tracking, while direct and
embedding-only calls preserve the current gradient context.
:::

Embedding-only calls skip prediction-head execution. All calls accept the
[DNA and organism inputs](../background/data-and-metadata.md#model-inputs)
described in Data and Metadata.

Shape notation used throughout this page:

:::{container} long-table

| Symbol | Meaning | Definition | Published Value |
| :---: | --- | --- | --- |
| $\mathrm{B}$ | Batch Size | Model input | Input-dependent |
| $\mathrm{S}_1$ | 1-bp Sequence Length | Model input | Up to 1,048,576 |
| $\mathrm{S}_{128}$ | 128-bp Sequence Length | $\mathrm{S}_1 / 128$ | Up to 8,192 |
| $\mathrm{S}_{\mathrm{pair}}$ | Pair-Grid Side Length | $\mathrm{S}_{128} / 16 = \mathrm{S}_1 / 2048$ | Up to 512 |
| $\mathrm{C}$ | Base Num Channels | `num_channels` | 768 |
| $\mathrm{I}$ | Channel Increment | `channel_increment` | 128 |
| $\mathrm{R}$ | Output Embedder MLP Ratio | `embedder_mlp_ratio` | 2 |
| $\mathrm{C}_1$ | 1-bp Embedding Channels | $\mathrm{C} \times \mathrm{R}$ | 1,536 |
| $\mathrm{C}_{128}$ | 128-bp Embedding Channels | $(\mathrm{C} + 6\mathrm{I}) \times \mathrm{R}$ | 3,072 |
| $\mathrm{C}_{\mathrm{pair}}$ | Pair Embedding Channels | `pair_channels` | 128 |
| $\mathrm{T}$ | Head Output Size | Head metadata | Head-dependent |
| $\mathrm{U}$ | Num Splice-Junction Tissues | Head metadata | Head-dependent |
| $\mathrm{K}$ | Num Splice Candidates | Supplied positions width or `num_splice_sites` when generated | 512 when generated |

:::

See [Model Architecture](../background/model-architecture.md) for how these shapes are derived.

## Predictions

Access each output as `predictions[head][output]`. Heads absent from metadata or marked `enabled: false` are omitted (see [Head Configuration](../background/data-and-metadata.md#head-configuration)).

The principal outputs are:

| Heads | Output Keys | Shape |
| --- | --- | --- |
| `atac`, `dnase`, `procap`, `cage`, `rna_seq` | `scaled_predictions_1bp`, `predictions_1bp`<br>`scaled_predictions_128bp`, `predictions_128bp` | $[\mathrm{B}, \mathrm{S}_1, \mathrm{T}]$<br>$[\mathrm{B}, \mathrm{S}_{128}, \mathrm{T}]$ |
| `chip_tf`, `chip_histone` | `scaled_predictions_128bp`, `predictions_128bp` | $[\mathrm{B}, \mathrm{S}_{128}, \mathrm{T}]$ |
| `contact_maps` | `predictions` | $[\mathrm{B}, \mathrm{S}_{\mathrm{pair}}, \mathrm{S}_{\mathrm{pair}}, \mathrm{T}]$ |
| `splice_sites_classification` | `logits`, `predictions` | $[\mathrm{B}, \mathrm{S}_1, 5]$ |
| `splice_sites_usage` | `logits`, `predictions` | $[\mathrm{B}, \mathrm{S}_1, \mathrm{T}]$ |
| `splice_sites_junction` | `predictions`<br>`splice_site_positions`<br>`splice_junction_mask` | $[\mathrm{B}, \mathrm{K}, \mathrm{K}, 2\mathrm{U}]$<br>$[\mathrm{B}, 4, \mathrm{K}]$<br>$[\mathrm{B}, \mathrm{K}, \mathrm{K}, 2\mathrm{U}]$ |
| `masked_language_modeling` | `logits`, `predictions` | $[\mathrm{B}, \mathrm{S}_1, 5]$ |

`scaled_predictions_*`
: Normalized genome-track outputs.

`predictions_*`
: Genome-track outputs transformed to experimental scale. See
  [Predictions Scaling](../background/model-architecture.md#predictions-scaling).

`logits` and `predictions`
: For splice-site classification, splice-site usage, and masked language
  modeling, `predictions` contains probabilities computed from `logits`.

`splice_junction_mask`
: Effective validity mask combining candidate validity, metadata availability,
  and the optional batch mask. Junction predictions are exactly zero wherever
  this mask is zero.

:::{dropdown} Show the Prediction Tree

```python
for head, outputs in predictions.items():
    print(head)
    for output_name in outputs:
        print(f"  {output_name}")
```

With all published heads enabled, this prints:

```text
atac
  scaled_predictions_1bp
  predictions_1bp
  scaled_predictions_128bp
  predictions_128bp
dnase
  scaled_predictions_1bp
  predictions_1bp
  scaled_predictions_128bp
  predictions_128bp
procap
  scaled_predictions_1bp
  predictions_1bp
  scaled_predictions_128bp
  predictions_128bp
cage
  scaled_predictions_1bp
  predictions_1bp
  scaled_predictions_128bp
  predictions_128bp
rna_seq
  scaled_predictions_1bp
  predictions_1bp
  scaled_predictions_128bp
  predictions_128bp
chip_tf
  scaled_predictions_128bp
  predictions_128bp
chip_histone
  scaled_predictions_128bp
  predictions_128bp
contact_maps
  predictions
splice_sites_classification
  logits
  predictions
splice_sites_usage
  logits
  predictions
splice_sites_junction
  predictions
  splice_site_positions
  splice_junction_mask
```

:::

See [Predictions Meaning](../background/model-architecture.md#predictions-meaning)
for prediction ranges and final-axis meanings.

:::{note}
Prediction tensors use the maximum track or tissue width across organisms.
Metadata masks identify the valid channels for each organism without removing
masked channels from the returned tensors. See
[Metadata](../background/data-and-metadata.md#metadata) for channel ordering
and masks.
:::

:::{dropdown} Providing Splice-Site Candidates
:color: info
:icon: info

Splice-junction candidates may be supplied through
`DataBatch.splice_site_positions` or generated from an enabled
`splice_sites_classification` head. For generated candidates, sites with
classification probability below `splice_site_threshold` are stored as `-1`
padding, and every junction involving them is excluded by
`splice_junction_mask`. See
[Data and Metadata](../background/data-and-metadata.md#other-fields) for the
position format.

:::


## Embeddings

`Embeddings` contains the final 1-bp, 128-bp, and pair representations available to prediction heads:

| Attribute | Representation | Generic Shape | Published Shape |
| --- | --- | --- | --- |
| `embeddings_1bp` | Sequence &middot; 1-bp | $[\mathrm{B}, \mathrm{S}_1, \mathrm{C}_1]$ | $[\mathrm{B}, \mathrm{S}_1, 1{,}536]$ |
| `embeddings_128bp` | Sequence &middot; 128-bp | $[\mathrm{B}, \mathrm{S}_{128}, \mathrm{C}_{128}]$ | $[\mathrm{B}, \mathrm{S}_{128}, 3{,}072]$ |
| `embeddings_pair` | Pair &middot; (2,048-bp &times; 2,048-bp) | $[\mathrm{B}, \mathrm{S}_{\mathrm{pair}}, \mathrm{S}_{\mathrm{pair}}, \mathrm{C}_{\mathrm{pair}}]$ | $[\mathrm{B}, \mathrm{S}_{\mathrm{pair}}, \mathrm{S}_{\mathrm{pair}}, 128]$ |

See [Model Architecture](../background/model-architecture.md) for how these widths are produced.
