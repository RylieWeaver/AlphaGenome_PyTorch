# Inference

Inference accepts the
[DNA and organism inputs](../background/data-and-metadata.md#model-inputs)
described in Data and Metadata.


## Predict from DNA

Load the published model and run every enabled prediction head:

```{code-block} python
import torch

from alphagenome_pt import deepmind_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepmind_model(load_state=True, device=device)
model.eval()

sequence = "ACGT" * 2048  # 8,192 bp
predictions = model.predict(sequence, organism_index=0)  # human
```

:::{admonition} Example Behavior
:class: note

`load_state=True`
: Loads pretrained state. See [DeepMind Checkpoints](deepmind-checkpoints.md)
  for `deepmind_model()` arguments and loading behavior.

`model.eval()`
: Selects evaluation behavior, including stored BatchNorm statistics.

`model.predict()`
: Disables gradient tracking.

See [Example Inference Output Shapes](#example-inference-output-shapes) for the
exact output from this example and
[Predictions and Embeddings](predictions-and-embeddings.md) for generic output
keys and shapes.

:::


### Select Prediction Heads

Enable only the RNA-seq prediction head:

```{code-block} python
for head in model.metadata.metadata["heads"].values():
    head["enabled"] = False

model.metadata.metadata["heads"]["rna_seq"]["enabled"] = True
```

Changing `enabled` controls execution and returned outputs without rebuilding the model. See [Head Configuration](../background/data-and-metadata.md#head-configuration).

:::{dropdown} Splice-Junction Memory Use
:color: warning
:icon: alert

Splice-junction prediction is particularly memory-intensive. Disable heads you
do not need before running inference.
:::


## Batch Inputs and Organisms

Pass equal-length sequences and organism indices directly, or collect them in
a `DataBatch`:

```{code-block} python
import torch

from alphagenome_pt import DataBatch, DNAOneHotEncoder

sequences = ["ACGT" * 2048, "TGCA" * 2048]
one_hot = DNAOneHotEncoder().encode(sequences)
organism_indices = torch.tensor([0, 1])  # human, mouse

predictions = model.predict(sequences, organism_index=organism_indices)

batch = DataBatch(
    dna_sequence_one_hot=one_hot,
    organism_index=organism_indices,
).to(device)
predictions_from_batch = model.predict(batch)
```

See [Model Inputs](../background/data-and-metadata.md#model-inputs) for accepted
DNA forms and organism indices.

:::{dropdown} Sequence Length Requirements
:color: warning
:icon: alert

Every sequence must be at least 2,048 bp, divisible by 2,048, and no longer
than `model.max_seq_len`. Sequences within one batch must have equal length.

:::


## Example Inference Output Shapes

For the 8,192-bp single-sequence example in
[Predict from DNA](#predict-from-dna), the principal returned shapes are shown
below.

:::{container} long-table

| Head | 1-bp Shape | 128-bp Shape | Other Shapes |
| --- | --- | --- | --- |
| `atac` | $[1, 8192, 256]$ | $[1, 64, 256]$ | — |
| `dnase` | $[1, 8192, 384]$ | $[1, 64, 384]$ | — |
| `procap` | $[1, 8192, 128]$ | $[1, 64, 128]$ | — |
| `cage` | $[1, 8192, 640]$ | $[1, 64, 640]$ | — |
| `rna_seq` | $[1, 8192, 768]$ | $[1, 64, 768]$ | — |
| `chip_tf` | — | $[1, 64, 1664]$ | — |
| `chip_histone` | — | $[1, 64, 1152]$ | — |
| `contact_maps` | — | — | Predictions: $[1, 4, 4, 28]$ |
| `splice_sites_classification` | $[1, 8192, 5]$ | — | — |
| `splice_sites_usage` | $[1, 8192, 734]$ | — | — |
| `splice_sites_junction` | — | — | Predictions and mask: $[1, 512, 512, 734]$<br>Positions: $[1, 4, 512]$ |

:::

:::{dropdown} Full Printed Output
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

:::

:::{admonition} Output Shape Context
:class: note

These shapes use a batch size of `1`, an 8,192-bp sequence, all published heads
enabled, and `num_splice_sites=512`. Genome-track heads return both
`scaled_predictions_*` and `predictions_*` at each listed resolution.
Splice-site classification and usage return both `logits` and `predictions`
with the listed shape. Different inputs, enabled heads, metadata, or supplied
splice-site positions can change the output shapes.

:::
