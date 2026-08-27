# Quickstart

Load the published AlphaGenome checkpoint, predict from an 8,192-bp DNA
sequence, and inspect the model's learned representations.

If the package is not installed yet, start with [Installation](installation.md).

## 1. Load the Published Model

```{code-block} python
import torch

from alphagenome_pt import deepmind_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepmind_model(load_state=True, device=device)
```

:::{dropdown} Checkpoint Download Size
:color: warning
:icon: alert

When the checkpoint is not already available locally, this call downloads the
roughly 1.8 GB checkpoint from Hugging Face.
:::

See [DeepMind Checkpoints](model/deepmind-checkpoints.md) for checkpoint
arguments and loading behavior.

## 2. Run the Model

Pass DNA strings and optional organism indices to the model.

```{code-block} python
sequence = "ACGT" * 2048    # 8,192 bp
organism_index = 0          # human in published checkpoint

model.eval()
predictions, embeddings = model.predict(
    sequence,
    organism_index=organism_index,
    return_embeddings=True,
)
```

:::{admonition} Gradient Tracking
:class: note

`model.predict(...)` disables gradient tracking. For differentiable
predictions, use `model(..., mode="predict")`, which preserves the current
gradient context. See [Training](model/training.md) for training with model
outputs.
:::

:::{dropdown} Sequence Length Requirements
:color: warning
:icon: alert

Every sequence must be at least 2,048 bp, divisible by 2,048, and no longer
than `model.max_seq_len` (1,048,576 bp in the published checkpoint). Sequences
within one batch must have equal length.
:::

### 2.1 Accepted Input Forms

The same prediction call accepts raw DNA, one-hot tensors, or a `DataBatch`:

::::{tab-set}
:::{tab-item} Raw DNA
```python
predictions = model.predict(sequence, organism_index=0)
```
:::

:::{tab-item} One-hot DNA
```python
from alphagenome_pt import DNAOneHotEncoder

one_hot = DNAOneHotEncoder().encode(sequence)
predictions = model.predict(one_hot, organism_index=0)
```
:::

:::{tab-item} DataBatch
```python
from alphagenome_pt import DataBatch, DNAOneHotEncoder

batch = DataBatch(
    dna_sequence_one_hot=DNAOneHotEncoder().encode(sequence),
    organism_index=0,
)
predictions = model.predict(batch)
```
:::
::::

Model calls normalize these inputs and move tensor fields to the model device.
See [Model Inputs](background/data-and-metadata.md#model-inputs) for batching,
organism indices, and input normalization.

## 3. Inspect Outputs

A `model.predict(...)` call with `return_embeddings=True` will
return a nested prediction dictionary and an `Embeddings` object.

### Predictions

Predictions are keyed by head and output.

:::{dropdown} Show Prediction Heads
```python
for head_name in predictions:
    print(head_name)
```

```text
atac
dnase
procap
cage
rna_seq
chip_tf
chip_histone
contact_maps
splice_sites_classification
splice_sites_usage
splice_sites_junction
```
:::

::::{dropdown} Show Prediction Shapes
Print the returned tensors by head:

```python
for head, outputs in predictions.items():
    print(head)
    for name, value in outputs.items():
        print(f"  {name}: {tuple(value.shape)}")
```

```text
atac
  scaled_predictions_1bp: (1, 8192, 256)
  predictions_1bp: (1, 8192, 256)
  scaled_predictions_128bp: (1, 64, 256)
  predictions_128bp: (1, 64, 256)
dnase
  scaled_predictions_1bp: (1, 8192, 384)
  predictions_1bp: (1, 8192, 384)
  scaled_predictions_128bp: (1, 64, 384)
  predictions_128bp: (1, 64, 384)
procap
  scaled_predictions_1bp: (1, 8192, 128)
  predictions_1bp: (1, 8192, 128)
  scaled_predictions_128bp: (1, 64, 128)
  predictions_128bp: (1, 64, 128)
cage
  scaled_predictions_1bp: (1, 8192, 640)
  predictions_1bp: (1, 8192, 640)
  scaled_predictions_128bp: (1, 64, 640)
  predictions_128bp: (1, 64, 640)
rna_seq
  scaled_predictions_1bp: (1, 8192, 768)
  predictions_1bp: (1, 8192, 768)
  scaled_predictions_128bp: (1, 64, 768)
  predictions_128bp: (1, 64, 768)
chip_tf
  scaled_predictions_128bp: (1, 64, 1664)
  predictions_128bp: (1, 64, 1664)
chip_histone
  scaled_predictions_128bp: (1, 64, 1152)
  predictions_128bp: (1, 64, 1152)
contact_maps
  predictions: (1, 4, 4, 28)
splice_sites_classification
  logits: (1, 8192, 5)
  predictions: (1, 8192, 5)
splice_sites_usage
  logits: (1, 8192, 734)
  predictions: (1, 8192, 734)
splice_sites_junction
  predictions: (1, 512, 512, 734)
  splice_site_positions: (1, 4, 512)
  splice_junction_mask: (1, 512, 512, 734)
```

The dimensions reflect one 8,192-bp sequence and the published model's padded
track widths. See [Predictions and Embeddings](model/predictions-and-embeddings.md)
for output meanings, generic shapes, and metadata masks.
::::

### Embeddings

:::{dropdown} Show Embedding Shapes
```python
print(embeddings.embeddings_1bp.shape)
print(embeddings.embeddings_128bp.shape)
print(embeddings.embeddings_pair.shape)
```

```text
torch.Size([1, 8192, 1536])
torch.Size([1, 64, 3072])
torch.Size([1, 4, 4, 128])
```
:::

:::{dropdown} Embedding-Only Calls
:color: info
:icon: info

`model.embed(...)` returns embeddings without running prediction heads.
Call it inside `torch.inference_mode()` for inference because `embed()`
preserves gradient tracking by default. See
[Predictions and Embeddings](model/predictions-and-embeddings.md) for
output keys, shapes, and gradient behavior.
:::

## Example with FASTA

Install `pysam`:

```bash
python -m pip install pysam
```

Fetch an 8,192-bp interval and pass it directly to the model:

```python
import pysam

with pysam.FastaFile("file.fa") as fasta:
    sequence = fasta.fetch("chr1", 0, 8192)

predictions = model.predict(sequence, organism_index=0)
```

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Run Inference
:link: model/inference
:link-type: doc
:class-card: sd-card-hover

Select prediction heads and run predictions on single sequences or batches.
:::

:::{grid-item-card} Inspect Model Outputs
:link: model/predictions-and-embeddings
:link-type: doc
:class-card: sd-card-hover

Understand keys and tensor shapes for predictions and embeddings.
:::

:::{grid-item-card} Construct a Model
:link: model/construction
:link-type: doc
:class-card: sd-card-hover

Use custom organisms, tracks, or architecture hyperparameters and optionally load checkpoints.
:::

:::{grid-item-card} Train or Fine-Tune
:link: model/training
:link-type: doc
:class-card: sd-card-hover

Fine-tune the published model or train a custom architecture with built-in or custom losses.
:::

::::
