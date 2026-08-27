# Data and Metadata

This page defines the inputs passed to AlphaGenome, the metadata that configures
its organisms and prediction heads, and the targets and masks used for loss
calculation.

## Shape Notation

The following symbols are used throughout the data and model references:

:::{container} long-table

| Symbol | Meaning | Definition | Published Value |
| :---: | --- | --- | --- |
| $\mathrm{B}$ | Batch Size | Model input | Input-dependent |
| $\mathrm{S}_1$ | 1-bp Sequence Length | Model input | Up to 1,048,576 |
| $\mathrm{S}_{128}$ | 128-bp Sequence Length | $\mathrm{S}_1 / 128$ | Up to 8,192 |
| $\mathrm{S}_{\mathrm{pair}}$ | Pair-Grid Side Length | $\mathrm{S}_{128} / 16 = \mathrm{S}_1 / 2048$ | Up to 512 |
| $\mathrm{O}$ | Num Organisms | Model metadata | 2 |
| $\mathrm{T}$ | Head Output Size | Head metadata | Head-dependent |
| $\mathrm{U}$ | Num Splice-Junction Tissues | Head metadata | Head-dependent |
| $\mathrm{K}$ | Num Splice Candidates | Supplied positions width or `num_splice_sites` when generated | 512 when generated |

:::

See [Model Architecture](model-architecture.md) for how these shapes are derived.

## Model Inputs

Every model call requires a DNA representation for each batch item. Organism indices are optional and default to `0`.

### DNA Inputs

Raw DNA
: Pass a `str` or equal-length `Sequence[str]` directly, or provide it through
  `DataBatch.dna_sequence`.

One-hot DNA
: Pass a `torch.Tensor` with shape $[\mathrm{S}_1, 4]$ or
  $[\mathrm{B}, \mathrm{S}_1, 4]$ directly, or provide it through
  `DataBatch.dna_sequence_one_hot`. Channels are ordered A, C, G, T.

A `DataBatch` may provide either or both DNA forms through their corresponding fields.

:::{admonition} DNA Input Normalization
:class: note

The model adds a leading batch dimension when necessary. One-hot encoding
accepts lowercase bases in the same way and encodes other characters (e.g. N)
as `[0, 0, 0, 0]`.
:::

:::{dropdown} Sequence Length Requirements
:color: warning
:icon: alert

Every sequence must be at least 2,048 bp, divisible by 2,048, and no longer
than `model.max_seq_len`. All sequences within one batch must have equal
length.
:::

### Organism Indices

Scalar index
: Pass an `int` or scalar integer `torch.Tensor` to use the same organism for
  every batch item.

Per-item indices
: Pass a `Sequence[int]` or integer `torch.Tensor` with shape
  $[\mathrm{B}]$ or $[\mathrm{B}, 1]$ to select an organism separately for
  each batch item.

:::{admonition} Organism Index Normalization
:class: note

The model normalizes organism indices to an integer tensor with shape
$[\mathrm{B}]$. Each index must be an integer in $[0, \mathrm{O} - 1]$, with its
meaning determined by the order of `metadata["organisms"]`. An omitted index
defaults every batch item to organism `0`. If indices are supplied both inside
`DataBatch` and as a model argument, their normalized values must match.
:::

### Input Example

Prepare three equivalent DNA inputs:
```{code-block} python

from alphagenome_pt import DataBatch, DNAOneHotEncoder, small_alphagenome

sequence = "ACGT" * 2048
one_hot = DNAOneHotEncoder().encode(sequence)
batch = DataBatch(dna_sequence=sequence, organism_index=0)

model = small_alphagenome(max_seq_len=len(sequence))
```

Each equivalent DNA representation results in the same embeddings:

::::{tab-set}
:::{tab-item} Raw DNA
```python
embeddings = model(sequence, organism_index=0, mode="embed")
```
:::

:::{tab-item} One-hot DNA
```python
embeddings = model(one_hot, organism_index=0, mode="embed")
```
:::

:::{tab-item} DataBatch
```python
embeddings = model(batch, mode="embed")
```
:::
::::

## Metadata

`Metadata` defines the organisms and prediction tasks for the model.

### Example

```{code-block} python
:caption: Configure two organisms with different RNA-seq output widths

import torch

from alphagenome_pt import Metadata, small_alphagenome

metadata = Metadata({
    "organisms": ["human", "mouse"],
    "heads": {
        "rna_seq": {
            "num_tracks": [2, 1],
            "means": torch.tensor([
                [2.1, 0.8],
                [3.8, 1.0],  # neutral value for the padded track
            ]),
        },
    },
})
print(metadata.get_organism_index("mouse"))  # 1
print(metadata.get_track_mask("rna_seq"))
# tensor([[ True, True],
#         [ True, False]])

model = small_alphagenome(metadata)
model.eval()
predictions = model.predict("ACGT" * 2048, organism_index=0)  # 8,192 bp
print(predictions.keys())
# dict_keys(['rna_seq'])
print(predictions["rna_seq"]["predictions_1bp"].shape)
# torch.Size([1, 8192, 2])
print(predictions["rna_seq"]["predictions_128bp"].shape)
# torch.Size([1, 64, 2])
```

:::{admonition} Track Counts and Ordering
:class: note

For each head, `num_tracks` contains one value per organism and therefore has
length $\mathrm{O}$. Tracks at the same numeric index generally do not describe
the same context across organisms or heads.
:::

:::{admonition} Padded Prediction Width
:class: important

Predictions are padded to $\mathrm{T}$, the maximum value in `num_tracks`.
:::

### Head Configuration

The supported metadata head keys are exposed through the public `HeadName` enum:

```python
from alphagenome_pt import HeadName

head_keys = [head.value for head in HeadName]
```

| Heads | Metadata Fields |
| --- | --- |
| `atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone` | `num_tracks`, `means`, optional `track_mask` |
| `contact_maps`, `splice_sites_usage` | `num_tracks`, optional `track_mask` |
| `splice_sites_classification` | `num_tracks` (`5` for every organism), optional `track_mask` |
| `splice_sites_junction` | `num_tissues`, optional `tissue_mask` |
| `masked_language_modeling` | No head-specific fields (`{}` enables the head) |

:::{dropdown} Genome-Track Mean Requirements
:color: warning
:icon: alert

Genome-track `means` have shape $[\mathrm{O}, \mathrm{T}]$. Every entry,
including masked and padded entries, must be finite and positive because targets
are scaled before masking. Compute means from the training split only and use
the neutral value `1.0` for padding. In the published metadata, each mean is
computed over that track's nonzero values.
:::

:::{dropdown} Metadata with All Built-In Heads
:color: info
:icon: info

This one-organism example includes every `HeadName`. It uses two outputs for
each configurable-width head and omits optional masks, which are derived from
the track and tissue counts.

```python
from alphagenome_pt import Metadata

# Replace these illustrative values with positive training-split means in
# target-channel order.
metadata = Metadata({
    "organisms": ["human"],
    "heads": {
        "atac": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "dnase": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "procap": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "cage": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "rna_seq": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "chip_tf": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "chip_histone": {"num_tracks": [2], "means": [[2.1, 0.8]]},
        "contact_maps": {"num_tracks": [2]},
        "splice_sites_classification": {"num_tracks": [5]},
        "splice_sites_usage": {"num_tracks": [2]},
        "splice_sites_junction": {"num_tissues": [2]},
        "masked_language_modeling": {},
    },
})
```

`masked_language_modeling` is built in but is not part of the published
checkpoint. Use the resulting object with a construction path described in
[Configuration](../model/configuration.md).
:::

Configured heads can be disabled or re-enabled without rebuilding the model:

```python
head_metadata = model.metadata.metadata["heads"]["rna_seq"]
head_metadata["enabled"] = False
head_metadata["enabled"] = True
```

Every configured head is enabled unless its metadata sets `enabled: false`. Disabled heads are omitted from predictions and loss calculation. Changing this flag cannot add a head that was absent when the model was constructed.

### Masks

:::{dropdown} How Metadata Masks Are Derived
:color: info
:icon: info

Unless provided, `Metadata` derives `track_mask` with shape
$[\mathrm{O}, \mathrm{T}]$ from `num_tracks`, where valid channels are `True`
and padded channels are `False`. The splice-junction head similarly derives
`tissue_mask` with shape $[\mathrm{O}, \mathrm{U}]$ from `num_tissues`.
:::

Metadata masks represent dataset-wide channel padding, whereas
[`DataBatch` masks](#databatch) represent per-sample target availability. Loss
calculation combines the two with a logical `AND`:

```text
effective loss mask = metadata channel mask AND DataBatch availability mask
```

### Lookup Methods

:::{dropdown} Common Metadata Lookups

```python
metadata.get_organisms()
metadata.get_heads()
metadata.get_num_tracks("rna_seq")
metadata.get_num_tracks_organism("mouse", "rna_seq")
organism_index = torch.tensor([0, 1])
metadata.get_multiorg_track_mask("rna_seq", organism_index)

# Tissue methods apply only to splice_sites_junction
metadata.get_num_tissues("splice_sites_junction")
metadata.get_num_tissues_organism("mouse", "splice_sites_junction")
metadata.get_multiorg_tissue_mask("splice_sites_junction", organism_index)
```
:::


## DataBatch

`DataBatch` groups model inputs, targets, and masks. It is required for the model to
[calculate its built-in losses](../model/losses-and-metric-tree.md), whereas prediction
and embedding calls also accept raw DNA strings or one-hot DNA tensors directly.

### Input Fields

| Field | Shape | Notes |
| --- | --- | --- |
| `dna_sequence` | $[\mathrm{B}]$ after normalization | Raw DNA described in [DNA Inputs](#dna-inputs) |
| `dna_sequence_one_hot` | $[\mathrm{B}, \mathrm{S}_1, 4]$ after normalization | One-hot DNA described in [DNA Inputs](#dna-inputs) |
| `organism_index` | $[\mathrm{B}]$ after normalization | Indices described in [Organism Indices](#organism-indices) (defaults to `0`) |

At least one DNA field is required. Both may be supplied when their batch and sequence dimensions agree.

`DataBatch.to(device)` moves all tensor fields to the requested device, updates the batch in place, and returns that same batch. Non-tensor fields (e.g. raw DNA strings and organism index lists) are unchanged.

### Target and Mask Fields

:::{admonition} Required Targets
:class: important

Calculating built-in losses requires at least one enabled head and a
corresponding `DataBatch` target for every enabled head. Targets supplied for
disabled heads are ignored.
:::

:::{container} long-table

| Head(s) | Target Field | Target Shape | Mask Field | Mask Shape |
| --- | --- | --- | --- | --- |
| `atac`, `dnase`, `procap`, `cage`, `rna_seq` | Same as head | $[\mathrm{B}, \mathrm{S}_1, \mathrm{T}]$ | Same name with `_mask` | $[\mathrm{B}, 1, \mathrm{T}]$ |
| `chip_tf`, `chip_histone` | Same as head | $[\mathrm{B}, \mathrm{S}_{128}, \mathrm{T}]$ | Same name with `_mask` | $[\mathrm{B}, 1, \mathrm{T}]$ |
| `contact_maps` | Same as head | $[\mathrm{B}, \mathrm{S}_{\mathrm{pair}}, \mathrm{S}_{\mathrm{pair}}, \mathrm{T}]$ | Same name with `_mask` | $[\mathrm{B}, \#\mathrm{S}_{\mathrm{pair}}, \#\mathrm{S}_{\mathrm{pair}}, \mathrm{T}]$ |
| `splice_sites_classification` | `splice_sites` | $[\mathrm{B}, \mathrm{S}_1, 5]$ | N/A, but use an all-zero target row to mask a position | N/A |
| `splice_sites_usage` | `splice_site_usage` | $[\mathrm{B}, \mathrm{S}_1, \mathrm{T}]$ | `splice_site_usage_mask` | $[\mathrm{B}, \#\mathrm{S}_1, \mathrm{T}]$ |
| `splice_sites_junction` | `splice_junctions` | $[\mathrm{B}, \mathrm{K}, \mathrm{K}, 2\mathrm{U}]$ | `splice_junctions_mask` | $[\mathrm{B}, \#\mathrm{K}, \#\mathrm{K}, \mathrm{U}]$ or $[\mathrm{B}, \#\mathrm{K}, \#\mathrm{K}, 2\mathrm{U}]$ |
| `masked_language_modeling` | `mlm` | $[\mathrm{B}, \mathrm{S}_1]$ | N/A, but set unselected targets to `-100` | N/A |

:::

In mask shapes, $\#\mathrm{X}$ means that the axis may have size $1$ and broadcast or have the full size $\mathrm{X}$ and vary along that axis.

:::{admonition} How Batch Masks Are Applied
:class: note

Batch masks exclude targets unavailable for individual samples. Loss
calculation combines them with organism-specific metadata masks using a logical
`AND`. Genome-track masks broadcast across sequence positions, while
contact-map, splice-site-usage, and splice-junction masks may vary across their
spatial axes.
:::

:::{dropdown} Head-Specific Target and Mask Requirements
:color: warning
:icon: alert

Genome-track heads
: Targets use experimental-data scale and must be finite and nonnegative.
  Each head scales its targets using the prediction resolution and metadata
  means. When a batch mask is omitted, every metadata-valid track is treated
  as observed.

Contact maps
: `NaN` target values are treated as unavailable. Finite padding with an
  explicit mask is the clearer general convention.

Splice-site classification
: Targets are five-channel class distributions, typically one-hot, ordered
  donor `+`, acceptor `+`, donor `-`, acceptor `-`, other. An all-zero row
  excludes that position.

Splice-site usage
: Targets are independent values in $[0, 1]$.

Splice junctions
: Targets are nonnegative counts. Channels contain all positive-strand tissues
  followed by all negative-strand tissues. A mask with $\mathrm{U}$ channels is
  shared across both strand blocks, while a mask with $2\mathrm{U}$ channels
  may differ by strand.

Masked language modeling
: `mlm` must contain `torch.long` labels: `0` (A), `1` (C), `2` (G),
  `3` (T), `4` (ambiguous), or `-100` (ignored). Each loss batch must contain
  at least one non-ignored label.


Masking preserves tensor widths.
:::

### Other Fields

| Field | Shape | Values | Notes |
| --- | --- | --- | --- |
| `rna_seq_strand` | $[\mathrm{B}, 1, \mathrm{T}]$ | $-1$ (reverse), $0$ (unknown or paired), and $1$ (forward) | Currently unused and reserved for future reverse-complement batch creation. |
| `splice_site_positions` | $[\mathrm{B}, 4, \mathrm{K}]$ | Integers in $[0, \mathrm{S}_1 - 1]$ (sequence positions) or $-1$ (padding) | Donor and acceptor candidates for the junction head. When omitted, they are generated from `splice_sites_classification` predictions. |

The four `splice_site_positions` rows contain positive-strand donors, positive-strand acceptors, negative-strand donors, and negative-strand acceptors, in that order.
