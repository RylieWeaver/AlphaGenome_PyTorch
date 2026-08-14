# AlphaGenome PyTorch

`alphagenome-pt` implements AlphaGenome in PyTorch, including its prediction heads, losses, and PyTorch-converted checkpoint loading. It supports the published model and custom configurations for inference and training.

## Start Here

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install AlphaGenome PyTorch
:link: installation
:link-type: doc
:class-card: sd-card-hover

Install the published package and the optional tools needed for documentation or development.
:::

:::{grid-item-card} Follow the Quickstart
:link: quickstart
:link-type: doc
:class-card: sd-card-hover

Construct a model, prepare DNA input, and generate predictions or embeddings.
:::

::::

## Common Tasks

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Construct a Model
:link: model/construction
:link-type: doc
:class-card: sd-card-hover

Load PyTorch-converted checkpoints or configure an AlphaGenome model for a custom
organism, track set, or architecture.
:::

:::{grid-item-card} Run Inference
:link: model/inference
:link-type: doc
:class-card: sd-card-hover

Select prediction heads and generate predictions/embeddings from DNA.
:::

:::{grid-item-card} Train a Model
:link: model/training
:link-type: doc
:class-card: sd-card-hover

Fine-tune published models or train custom architectures with built-in or custom losses.
:::

::::

## Reference

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Data and Metadata
:link: background/data-and-metadata
:link-type: doc
:class-card: sd-card-hover

Understand model inputs, organisms, prediction heads, targets, and masks.
:::

:::{grid-item-card} Model Architecture
:link: background/model-architecture
:link-type: doc
:class-card: sd-card-hover

Understand how AlphaGenome processes DNA to produce embeddings and predictions.
:::

:::{grid-item-card} Command-Line Interface
:link: cli
:link-type: doc
:class-card: sd-card-hover

Check your installation and download PyTorch-converted checkpoints.
:::

::::

## Development

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Contribute and Test
:link: development/index
:link-type: doc
:class-card: sd-card-hover

Set up a checkout, run tests, build the docs, and generate compatible test data.
:::

:::{grid-item-card} Convert Checkpoints
:link: development/checkpoint-conversion/index
:link-type: doc
:class-card: sd-card-hover

Reproduce and publish PyTorch-converted checkpoints from the official JAX release.
:::

::::

```{toctree}
:caption: Contents
:maxdepth: 3
:hidden:

installation
quickstart
background/index
model/construction
model/running
cli
development/index
```
