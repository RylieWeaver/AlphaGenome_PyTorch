# Training

AlphaGenome PyTorch provides the model, `DataBatch` contract, and task losses. The package expects user-supplied tensors and training loops rather than providing a data pipeline or trainer.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Use a Custom Objective
:link: custom-objectives
:link-type: ref
:class-card: sd-card-hover

Train from differentiable predictions, embeddings, or both when you need
complete control over the objective.
:::

:::{grid-item-card} Use Built-In Losses
:link: built-in-losses
:link-type: ref
:class-card: sd-card-hover

Let AlphaGenome apply task-specific target processing, masking, and loss
components.
:::

::::

(custom-objectives)=
## Training from Embeddings or Predictions

Use a custom objective from differentiable model outputs when you need direct control over target handling, masking, weighting, or reduction:

| Train From | Differentiable Call |
| --- | --- |
| Embeddings | `model(batch, mode="embed")` |
| Predictions | `model(batch)` |
| Both | `model(batch, return_embeddings=True)` |

```{code-block} python
model.train()
batch.to(device)
optimizer.zero_grad(set_to_none=True)

predictions, embeddings = model(batch, return_embeddings=True)
loss = custom_loss(predictions, embeddings, batch)
loss.backward()
optimizer.step()
```

`model(batch, mode="embed")` and `model.embed(batch)` are equivalent for direct
model use. Both preserve the current model mode and gradient context. Do not
use `model.predict()` for training because it disables gradients. See
[Predictions and Embeddings](predictions-and-embeddings.md) for the returned
structures and shapes.

:::{dropdown} Distributed Training with Embeddings
:color: warning
:icon: alert

During `DistributedDataParallel` (DDP) training, request embeddings through
`ddp_model(batch, mode="embed")` so the forward pass runs through DDP's hooks.
Do not call `ddp_model.module.embed(batch)`. Bypassing the wrapper can cause
reducer errors when `find_unused_parameters=True`.
:::


(built-in-losses)=
## Training with Built-In Losses

:::{warning}

The built-in loss API remains under active development and testing.
:::

`model(batch, mode="loss")` computes AlphaGenome's built-in task losses with
task-specific target processing and masking. It returns a `LossOutput`
containing the differentiable total loss and a
[`MetricTree`](losses-and-metric-tree.md#metrictree) of component losses:

```{code-block} python
model.train()
optimizer.zero_grad(set_to_none=True)

output = model(batch, mode="loss")
output.total.backward()
optimizer.step()
```

`model(batch, mode="loss")` and `model.loss(batch)` are equivalent for direct
model use. Both preserve the current model mode and gradient context.

:::{admonition} Loss Requirements
:class: important

At least one head must be enabled, with a corresponding target for every
enabled head. See
[Target and Mask Fields](../background/data-and-metadata.md#target-and-mask-fields)
for the batch contract.
:::

:::{dropdown} Distributed Training with Built-In Losses
:color: warning
:icon: alert

During DDP training, compute loss through `ddp_model(batch, mode="loss")` so
the forward pass runs through DDP's hooks. Do not call
`ddp_model.module.loss(batch)`. Bypassing the wrapper can cause reducer errors
when `find_unused_parameters=True`.
:::


## Save and Load a Model

Save and load the model configuration, metadata, parameters, and buffers through one directory:

```{code-block} python
from alphagenome_pt import AlphaGenome

model.save("checkpoint/model")
loaded_model = AlphaGenome.load("checkpoint/model")
```

The directory contains `config.json`, `metadata.pt`, and `model.pt`. Pass the
`device` argument when loading to select the model device (default `"cpu"`).
Optimizer state and training progress are the user's responsibility.


## Training Guides

| Guide | Covers |
| --- | --- |
| [Data and Metadata](../background/data-and-metadata.md#databatch) | `DataBatch` inputs, targets, and masks |
| [DeepMind Checkpoints](deepmind-checkpoints.md) | Initializing training from converted DeepMind state |
| [Predictions and Embeddings](predictions-and-embeddings.md) | Differentiable model outputs for custom objectives |
| [Losses and the Metric Tree](losses-and-metric-tree.md) | Computing, inspecting, and accumulating built-in losses |
| [Synthetic Helpers](../development/synthetic-utilities.md) | Generating compatible metadata, models, and batches for testing |

```{toctree}
:maxdepth: 1
:hidden:

losses-and-metric-tree
```
