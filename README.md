# AlphaGenome PyTorch

<p align="center">
  <strong>Inference, fine-tuning, and research training with AlphaGenome in PyTorch.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/alphagenome-pt/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/alphagenome-pt"></a>
  <a href="https://pypi.org/project/alphagenome-pt/"><img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/alphagenome-pt"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.0 or newer" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/RylieWeaver/alphagenome-pytorch"><img alt="Converted checkpoints on Hugging Face" src="https://img.shields.io/badge/converted%20checkpoints-Hugging%20Face-FFD21E"></a>
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

<p align="center">
  <a href="docs/installation.md">Installation</a> &middot;
  <a href="docs/quickstart.md">Quickstart</a> &middot;
  <a href="docs/index.md">Documentation</a> &middot;
  <a href="https://github.com/RylieWeaver/AlphaGenome_PyTorch/issues">Issues</a>
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/RylieWeaver/AlphaGenome_PyTorch/main/images/AG_PyTorch.png"
    width="680"
    alt="AlphaGenome PyTorch logo over an illustration of DNA and a neural network"
  >
</p>

`alphagenome-pt` implements AlphaGenome in PyTorch, including its prediction
heads, losses, and PyTorch-converted checkpoint loading. It supports the
published model and custom configurations for inference and training.

## Get Started

Install the package:

```bash
python -m pip install alphagenome-pt
```

Load the published checkpoint and generate predictions:

```python
import torch

from alphagenome_pt import deepmind_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepmind_model(load_state=True, device=device)
model.eval()

sequence = "ACGT" * 2048  # 8,192 bp
predictions, embeddings = model.predict(
    sequence,
    organism_index=0,  # human in the published metadata
    return_embeddings=True,
)
```

<details>
<summary><strong>Prediction Shapes</strong></summary>

```python
for head, outputs in predictions.items():
    print(head)
    for name, value in outputs.items():
        print(f"  {name}: {tuple(value.shape)}")
```

Output:

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

</details>

<details>
<summary><strong>Prediction Resolutions</strong></summary>

| Head | Type | Resolution(s) |
| --- | --- | --- |
| `atac`, `dnase`, `procap`, `cage`, `rna_seq` | Sequence | 1 bp and 128 bp |
| `chip_tf`, `chip_histone` | Sequence | 128 bp |
| `contact_maps` | Pair | 2,048 bp &times; 2,048 bp |
| `splice_sites_classification`, `splice_sites_usage` | Sequence | 1 bp |
| `splice_sites_junction` | Donor &times; acceptor pairs | Selected 1-bp candidates |

**Prediction Shapes** includes the batch and output-channel dimensions omitted from this table. Published prediction tensors retain metadata-padded output widths. Organism-specific [metadata masks](docs/background/data-and-metadata.md#masks) identify valid track and tissue channels. Splice-junction outputs are indexed by selected candidates rather than over all base pairs.

</details>

<details>
<summary><strong>Embedding Shapes</strong></summary>

```python
print(embeddings.embeddings_1bp.shape)
print(embeddings.embeddings_128bp.shape)
print(embeddings.embeddings_pair.shape)
```

Output:

```text
torch.Size([1, 8192, 1536])
torch.Size([1, 64, 3072])
torch.Size([1, 4, 4, 128])
```

</details>

<br>

> **Note**
>
> When the artifacts are not already cached,
> `deepmind_model(load_state=True)` downloads the published metadata and a
> roughly 1.8 GB checkpoint.
> See [DeepMind Checkpoints](docs/model/deepmind-checkpoints.md) for
> loading arguments and behavior.

> **Warning**
>
> Sequences must be at least 2,048 bp, divisible by 2,048, and no longer than
> `model.max_seq_len`. See
> [Data and Metadata](docs/background/data-and-metadata.md) for the
> complete input contract.

## Select Prediction Heads

Run only the heads needed for a prediction:

```python
selected_heads = {"atac", "rna_seq"}

for head_name, head_metadata in model.metadata.metadata["heads"].items():
    head_metadata["enabled"] = head_name in selected_heads

predictions = model.predict(sequence, organism_index=0)
```

Disabled heads are skipped and omitted from `predictions`. This is especially
useful for the memory-intensive `splice_sites_junction` head. Changing
`enabled` does not rebuild the model or add heads that were not configured.

## Model Construction

Choose a construction path:

### Published Model

Load the published metadata, architecture, and checkpoint:

```python
from alphagenome_pt import deepmind_model

published_model = deepmind_model(load_state=True)
```

### Published Architecture with Custom Metadata

Define custom metadata, then flexibly load compatible organism and prediction-head state from the published checkpoint:

```python
from alphagenome_pt import Metadata, deepmind_model

custom_metadata = Metadata({
    "organisms": ["human"],
    "heads": {
        "rna_seq": {
            "num_tracks": [2],
            "means": [[2.1, 0.8]],  # Per-track divisor in target scaling (1.0 has no effect)
        },
    },
})

adapted_model = deepmind_model(
    metadata=custom_metadata,
    load_state=True,
    organisms=True,
    heads=True,
)
```

### Custom Architecture

Define custom metadata and initialize the model from scratch:

```python
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, Metadata

custom_metadata = Metadata({
    "organisms": ["dragon"],
    "heads": {
        "rna_seq": {
            "num_tracks": [2],
            "means": [[2.1, 0.8]],  # Per-track divisor in target scaling (1.0 has no effect)
        },
    },
})

custom_model = AlphaGenome(
    AlphaGenomeConfig(
        max_seq_len=8_192,
        num_channels=96,
        transformer_layers=3,
        metadata=custom_metadata,
    )
)
```

### Save and Load a Model

Save and load the model configuration, metadata, parameters, and persistent
buffers in a directory:

```python
from alphagenome_pt import AlphaGenome

model.save("checkpoints/model")
loaded_model = AlphaGenome.load("checkpoints/model", device=device)
```

See [Model Construction](docs/model/construction.md),
[DeepMind Checkpoints](docs/model/deepmind-checkpoints.md), and
[Configuration](docs/model/configuration.md) for complete options.

## Model Inputs and Forward Calls

Prediction and embedding calls accept several input forms. Built-in loss calls
require a `DataBatch` with targets.

### DNA Inputs

Pass raw strings or one-hot tensors directly, or provide either representation
through `DataBatch`:

| Representation | Pass Directly | Provide Through `DataBatch` |
| --- | --- | --- |
| Raw DNA | One `str` or an equal-length `Sequence[str]` | `dna_sequence` |
| One-hot DNA | `torch.Tensor` with shape `[S, 4]` or `[B, S, 4]` | `dna_sequence_one_hot` |

```python
from alphagenome_pt import DataBatch, DNAOneHotEncoder

sequences = ["ACGT" * 2048, "TGCA" * 2048]
one_hot = DNAOneHotEncoder().encode(sequences)  # [2, 8192, 4]
batch = DataBatch(dna_sequence=sequences, organism_index=[0, 1])

# Raw DNA and organism indices passed separately
string_predictions = model.predict(sequences, organism_index=[0, 1])

# One-hot DNA and organism indices passed separately
one_hot_predictions = model.predict(one_hot, organism_index=[0, 1])

# DNA and organism indices read from DataBatch
batch_predictions = model.predict(batch)
```

A single raw sequence or `[S, 4]` tensor receives a leading batch dimension.
One-hot channels are ordered A, C, G, T.

### Organism Indices

Omit the index, share one index across the batch, or provide one per sequence:

| Selection | Accepted Value | Normalized Shape |
| --- | --- | --- |
| Default | Omit `organism_index` | `[B]`, filled with `0` |
| Shared across the batch | `int` or scalar integer `torch.Tensor` | `[B]` |
| Per sequence | `Sequence[int]` or integer tensor with shape `[B]` or `[B, 1]` | `[B]` |

```python
import torch

sequences = ["ACGT" * 2048, "TGCA" * 2048]

# Omitted indices default every sequence to organism 0
default_predictions = model.predict(sequences)

# A scalar index is shared across the batch
shared_predictions = model.predict(sequences, organism_index=1)

# A length-B tensor selects one organism per sequence
per_sequence_predictions = model.predict(
    sequences,
    organism_index=torch.tensor([0, 1]),
)
```

The index may instead be stored in `DataBatch.organism_index`. Index
meanings follow the order of `metadata["organisms"]`.

### Forward Calls

Choose the call based on its return value and gradient behavior:

| Call | Returns | Gradient Tracking |
| --- | --- | --- |
| `model(data)` | Prediction dictionary | Current context |
| `model.predict(data)` | Prediction dictionary | Disabled |
| `model.embed(data)` or `model(data, mode="embed")` | `Embeddings` | Current context |
| `model(data, return_embeddings=True)` | `(predictions, embeddings)` | Current context |
| `model.predict(data, return_embeddings=True)` | `(predictions, embeddings)` | Disabled |
| `model(batch, mode="loss")` | `LossOutput` | Current context |

> **Warning**
>
> During DDP training, run parameter-using forwards through `ddp_model(...)`,
> including `mode="embed"` and `mode="loss"`. Calling the underlying module
> bypasses DDP's hooks and can cause reducer errors when
> `find_unused_parameters=True`.

Use direct model calls to define custom objectives from differentiable
predictions or embeddings.

For a target-bearing
[`DataBatch`](docs/background/data-and-metadata.md#databatch),
backpropagate the built-in loss total with:

```python
from alphagenome_pt import LossOutput

# Contains a target for every enabled head
output: LossOutput = model(batch, mode="loss")
loss, tree = output.total, output.tree
loss.backward()
```

> **Note**
>
> The built-in loss method remains under active development and testing.

See [Training](docs/model/training.md),
[Losses and the Metric Tree](docs/model/losses-and-metric-tree.md), and
[Predictions and Embeddings](docs/model/predictions-and-embeddings.md)
for complete behavior.

## Command-Line Interface

Inspect the install:

```bash
alphagenome-pt --version
alphagenome-pt --help
```

Download the converted all-folds checkpoint and metadata to a local directory:

```bash
alphagenome-pt download --local-dir checkpoints --fold all_folds
```

Load the downloaded artifacts into a model:

```python
model = deepmind_model(
    load_state=True,
    local_dir="checkpoints",
    device=device,
)
```

See the [Command-Line Interface](docs/cli.md) for all download options.

## Learn More

- **Understand:** [Data and Metadata](docs/background/data-and-metadata.md)
  &middot; [Architecture](docs/background/model-architecture.md)
- **Construct:** [Model Construction](docs/model/construction.md) &middot;
  [Checkpoints](docs/model/deepmind-checkpoints.md)
- **Run and Train:** [Running the Model](docs/model/running.md)

## Development and Support

<details>
<summary><strong>Install an editable checkout and run tests</strong></summary>

```bash
git clone https://github.com/RylieWeaver/AlphaGenome_PyTorch.git
cd AlphaGenome_PyTorch
python -m pip install -e ".[dev]"
python -m pytest tests
```

</details>

See [Development Setup](docs/development/setup.md) for contributor
instructions. To report a bug or request a feature, [open an
issue](https://github.com/RylieWeaver/AlphaGenome_PyTorch/issues).

## Citation

If AlphaGenome PyTorch supports your work, cite the published AlphaGenome
paper and link this repository:

> Avsec, Ž., Latysheva, N., Cheng, J. *et al.* Advancing regulatory variant
> effect prediction with AlphaGenome. *Nature* **649**, 1206–1218 (2026).
> [https://doi.org/10.1038/s41586-025-10014-0](https://doi.org/10.1038/s41586-025-10014-0)

Use the Nature article when citing AlphaGenome. The original bioRxiv entry is
included below for historical reference.

<details>
<summary><strong>Nature BibTeX</strong></summary>

```bibtex
@article{avsec_advancing_2026,
	title = {Advancing regulatory variant effect prediction with {AlphaGenome}},
	volume = {649},
	issn = {1476-4687},
	url = {https://www.nature.com/articles/s41586-025-10014-0},
	doi = {10.1038/s41586-025-10014-0},
	number = {8099},
	journal = {Nature},
	publisher = {Nature Publishing Group},
	author = {Avsec, Žiga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drotár, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
	month = jan,
	year = {2026},
	pages = {1206--1218},
}
```

</details>


<details>
<summary><strong>bioRxiv Preprint BibTeX</strong></summary>

```bibtex
@article{avsec_alphagenome_2025,
	title = {{AlphaGenome}: advancing regulatory variant effect prediction with a unified {DNA} sequence model},
	url = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.25.661532},
	doi = {10.1101/2025.06.25.661532},
	journal = {bioRxiv},
	publisher = {Cold Spring Harbor Laboratory},
	author = {Avsec, Žiga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drotár, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
	year = {2025},
}
```

</details>

## Acknowledgements

This implementation bases its model components off of
[AlphaGenome package](https://github.com/google-deepmind/alphagenome) and
[AlphaGenome research](https://github.com/google-deepmind/alphagenome_research) code repositories.
Some components are direct ports of the code, others are paper-based reimplementations, and some are original additions (e.g. the Masked Language Modeling Head). Source files identify applicable provenance.

We acknowledge the independent
[`genomicsxai/alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch)
implementation from the Kundaje Lab at Stanford and the authors named in its
copyright notice:
Danila Bredikhin, Martin Kjellberg, Christopher Zou, Alejandro Buendia,
Xinming Tu, and Anshul Kundaje.
<!-- (names listed at the end of their README in the license section) -->

We also acknowledge earlier independent AlphaGenome PyTorch work by
[Phil Wang](https://gitlab.com/lucidrains),
[Miquel Anglada-Girotto](https://github.com/MiqG), and
[Xinming Tu](https://github.com/XinmingTu).

This package was developed with assistance from LLM coding agents, including
OpenAI Codex. The repository banner was generated with Nano Banana 2.

Developing this project used resources of the Oak Ridge Leadership Computing
Facility, which is a DOE Office of Science User Facility supported under
Contract DE-AC05-00OR22725.

## License and Model Terms

> **Important**
>
> AlphaGenome PyTorch is an independent research implementation, not an
> official Google DeepMind package. It has not been designed or validated for
> direct clinical use.

Portions of this project are derived from Google DeepMind's
[AlphaGenome research code](https://github.com/google-deepmind/alphagenome_research),
which is licensed under the Apache License 2.0:

> Copyright 2026 Google LLC

Google DeepMind's published AlphaGenome model parameters, outputs produced from
them, and related derivatives remain subject to the
[AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms),
including restrictions on commercial use.

The [AlphaGenome PyTorch repository](https://github.com/RylieWeaver/AlphaGenome_PyTorch)
and its `alphagenome-pt` Python package are available under the
[Apache License 2.0](LICENSE):

> Copyright 2026 Rylie Weaver, Gomathi Lakshmanan, and John Lagergren
