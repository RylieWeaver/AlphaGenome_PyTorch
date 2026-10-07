# Parallelism

{bdg-warning}`Evolving API`

Once the model and distributed environment are initialized, 
`distribute_alphagenome()` wraps the model for parallel execution
across multiple GPUs:

```{code-block} python
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from alphagenome_pt import distribute_alphagenome

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device)

mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "sp"))
model = distribute_alphagenome(
    base_model.to(device),
    strategy="sp_ddp",
    mesh=mesh,
    device_ids=[device.index],
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

:::{dropdown} Using the Parallel Model
:color: info
:icon: info

AlphaGenome supports data parallelism (DP) and sequence parallelism (SP).

The wrapped model provides the same `.loss()`, `.embed()`, `.predict()`,
and `.save()` methods as the base model.

Create the optimizer **after wrapping the model**, using the wrapped model's
parameters.
:::


## Arguments

Specify `strategy` for the overall parallelism type, `mesh` for the DP/SP layout, and
(optionally) `device_mesh` when using FSDP to specify parameter sharding (defaults to WORLD).
The `sync_bn` argument controls [BatchNorm synchronization](#batchnorm).

### Parallel Strategies

| `strategy` | Parameters (default) | Sequences | Notes |
| --- | --- | --- | --- |
| `"sp"` | Full copy | Split | Routes through `"sp_ddp"` |
| `"ddp"` | Full copy | Whole | Optional `mesh` and must not contain an `"sp"` axis |
| `"fsdp"` | Split | Whole |  Optional `mesh` and must not contain an `"sp"` axis |
| `"sp_ddp"` | Full copy | Split | Requires `mesh` with an `"sp"` axis (`"dp"` is optional) |
| `"sp_fsdp"` | Split | Split | Requires `mesh` with an `"sp"` axis (`"dp"` is optional)|
| `"none"` | Unchanged | Whole | Returns the original model |

See [Mesh Layouts](#mesh-layouts) to configure how parameters are split across GPUs.

### Mesh Layouts

The DP/SP `mesh` and FSDP `device_mesh` control different parts of parallel
execution and can be configured independently:

| Argument | Layout | Controls |
| --- | --- | --- |
| `mesh` | `(dp, sp)` | Which ranks process different batches or split the same sequences |
| `device_mesh` | `(replicate, shard)` | Which ranks replicate parameter shards or share a sharded copy of the model |

For example, this 16-GPU setup uses DP=4, SP=4, and two FSDP sharding groups
of eight ranks:

```{code-block} python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import ShardingStrategy

sequence_mesh = init_device_mesh(
    "cuda", (4, 4), mesh_dim_names=("dp", "sp"),
)
fsdp_mesh = init_device_mesh(
    "cuda", (2, 8), mesh_dim_names=("replicate", "shard"),
)

model = distribute_alphagenome(
    base_model,
    strategy="sp_fsdp",
    mesh=sequence_mesh,
    device_mesh=fsdp_mesh,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```

:::{dropdown} Sharding Types and Defaults
:color: info
:icon: info

FSDP defaults to `FULL_SHARD` over WORLD. Use `HYBRID_SHARD` or
`_HYBRID_SHARD_ZERO2` with `device_mesh` to shard within smaller groups.

The hybrid mesh must be 2D, contain every WORLD rank exactly once, and
list ranks in ascending order within each sharding row.

If a hybrid strategy is requested without `device_mesh`, or with only one
rank per sharding group, the wrapper falls back to `FULL_SHARD` over WORLD.
:::

### BatchNorm

`sync_bn=True` (the default) combines training BatchNorm statistics across
WORLD while preserving the computation graph. Set `sync_bn=False` to compute
statistics separately on each rank.

Model mode controls which statistics are used:

| Mode | Statistics used | Updates running statistics? |
| --- | --- | --- |
| `.train()` | Current input statistics | Yes |
| `.eval()` | Stored running statistics | No |

:::{dropdown} How Sequence Parallel BatchNorm Breaks Exact Equivalence
:color: warning
:icon: alert

SP and non-parallel training are not guaranteed to produce identical
outputs or gradients, even with `sync_bn=True`.

As described in [DeepMind's paper](https://storage.googleapis.com/deepmind-media/papers/alphagenome.pdf#page=35),
SP uses 1,024 bp of overlap per neighboring shard, which BatchNorm
double-counts during training. Evaluation uses stored EMA statistics
rather than batch statistics, but those statistics may also reflect
overlaps seen in training.
:::

### Other Arguments

Additional keyword arguments, passed directly or via `**kwargs`, are forwarded to
[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
or [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html). The table below lists common
options:

| Argument | Applies to | Usage |
| --- | --- | --- |
| `device_ids` | DDP | Pass `[device.index]` for the local GPU |
| `device_id` | FSDP | Pass `device` for the local GPU |
| `find_unused_parameters` | DDP | Defaults to `True` to handle parameters unused by the loss |
| `use_orig_params` | FSDP | Defaults to `True`, exposing the original parameters to the optimizer |
| `process_group` | Both | Defaults to WORLD. Leave unset for hybrid FSDP |

Other arguments supported by the selected wrapper can also be passed as keyword arguments.

### Additional Restrictions

Parallel execution adds the following restrictions:

| Setting | Applies to | Requirement |
| --- | --- | --- |
| Input length | SP | A positive multiple of `2048 * sp_size` |
| Convolution widths | SP | `first_conv_width=15` and `block_width=5` |
| Splice candidates | SP with junction predictions | `num_splice_sites` (or `splice_site_positions.shape[-1]` when supplied) must be divisible by `sp_size` |

:::{dropdown} Why These Restrictions Apply
:color: info
:icon: info

Contact maps and pair embeddings use 2,048-bp bins. Each SP rank needs an
equal number of whole bins, so the sequence length must be a positive
multiple of `2048 * sp_size`.

SP uses 1,024 bp of overlap per neighboring shard so convolutions near shard
boundaries can access the sequence on both sides. Changing kernel widths may
change the required overlap, so SP currently restricts them to the defaults of
`first_conv_width=15` and `block_width=5`.

Junction predictions are split evenly along the donor candidate rows, so the
candidate count `K` must be divisible by `sp_size` to split evenly.
:::


## Running the Model

### Parallel Context

The wrapper creates a `ParallelContext` which can be accessed through
`model.parallel_context` to inspect the DP/SP layout and the current
rank’s place in that layout:

| Attribute | Significance |
| --- | --- |
| `mesh` | Defines the SP/DP layout |
| `dp_mesh_index` | Position along the mesh's DP axis |
| `sp_mesh_index` | Position along the mesh's SP axis |
| `dp_rank` | Rank within the DP communication group |
| `dp_size` | How many different local batches are processed in parallel |
| `sp_rank` | Rank within the SP communication group; determines the sequence shard |
| `sp_size` | How many parts each sequence is split into |
| `dp_group` | Processes at the same SP mesh position across DP batches |
| `sp_group` | Processes handling the same local batch |
| `loss_group` | Processes whose loss sums and counts are combined |

Mesh indices follow the supplied rank arrangement whereas communication-group ranks
use the sorted order. Without a named mesh axis, its mesh index falls back to the corresponding rank.

With SP or hybrid FSDP, `loss_group` includes every process (`WORLD`).

### Broadcasting Data

Give each DP replica a different full batch, then broadcast it to that
replica's SP ranks. The model handles sequence splitting. For example:

```{code-block} python
from alphagenome_pt import sp_broadcast_data_batch, synthetic_batch

context = model.parallel_context
batch = None
if context.sp_mesh_index == 0:
    torch.manual_seed(42 + context.dp_mesh_index)
    batch = synthetic_batch(
        base_model.metadata, batch_size=1, seq_len=8192,
    )

# Every rank in the SP group calls the broadcast.
batch = sp_broadcast_data_batch(batch, parallel_context=context)
```

For real data, replace `synthetic_batch()` with a loader partitioned by
`context.dp_mesh_index` and `context.dp_size`. The source is always the process
with `context.sp_mesh_index == 0` in each SP group.

### Returned Predictions and Embeddings

Use `.predict()` to return predictions and, optionally, embeddings:

```{code-block} python
model.eval()
predictions, embeddings = model.predict(batch, return_embeddings=True)
```

Using the notation from
[Predictions and Embeddings](predictions-and-embeddings.md),
we denote the expected output shapes under SP below:

:::{container} long-table sp-output-shapes

| Embedding / Heads | Attribute / Output Keys | Full Shape | Local SP Shape |
| --- | --- | --- | --- |
| 1-bp embeddings | `embeddings_1bp` | `[B, S_1, C_1]` | `[B, S_1 / SP, C_1]` |
| 128-bp embeddings | `embeddings_128bp` | `[B, S_128, C_128]` | `[B, S_128 / SP, C_128]` |
| Pair embeddings | `embeddings_pair` | `[B, S_pair, S_pair, C_pair]` | `[B, S_pair / SP, S_pair, C_pair]` |
| `atac`, `dnase`<br>`procap`, `cage`<br>`rna_seq` | `scaled_predictions_1bp`, `predictions_1bp`<br>`scaled_predictions_128bp`, `predictions_128bp` | `[B, S_1, T]`<br>`[B, S_128, T]` | `[B, S_1 / SP, T]`<br>`[B, S_128 / SP, T]` |
| `chip_tf`<br>`chip_histone` | `scaled_predictions_128bp`, `predictions_128bp` | `[B, S_128, T]` | `[B, S_128 / SP, T]` |
| `contact_maps` | `predictions` | `[B, S_pair, S_pair, T]` | `[B, S_pair / SP, S_pair, T]` |
| `splice_sites_classification` | `logits`, `predictions` | `[B, S_1, 5]` | `[B, S_1 / SP, 5]` |
| `splice_sites_usage` | `logits`, `predictions` | `[B, S_1, T]` | `[B, S_1 / SP, T]` |
| `splice_sites_junction` | `predictions`<br>`splice_site_positions`<br>`splice_junction_mask` | `[B, K, K, 2U]`<br>`[B, 4, K]`<br>`[B, K, K, 2U]` | `[B, K / SP, K, 2U]`<br>`[B, 4, K]`<br>`[B, K / SP, K, 2U]` |
| `masked_language_modeling` | `logits`, `predictions` | `[B, S_1, 5]` | `[B, S_1 / SP, 5]` |

:::

:::{dropdown} Splice-Junction Shards
:color: info
:icon: info

Each rank receives `K / SP` donor rows, independently of the candidates' DNA
positions. As a result, `K` must be divisible by `SP`. Pad splice sites with `-1`
if necessary, along with corresponding padding of junction targets and masks.
:::

:::{dropdown} Gather Full Outputs
:color: info
:icon: info

Call `all_gather()` on every SP rank to concatenate sharded outputs
in SP-rank order along a dimension:

```{code-block} python
from alphagenome_pt import all_gather

full_embeddings = all_gather(
    embeddings.embeddings_1bp,
    dim=1,
    group=model.parallel_context.sp_group,
)
```
:::

### Training

Calling `.loss()` and backpropagating are done in the same way as non-parallel
execution, with gradient communication and synchronization handled internally:

```{code-block} python
model.train()
optimizer.zero_grad(set_to_none=True)

output = model.loss(batch)
output.total.backward()
optimizer.step()
```

:::{dropdown} Loss and Gradient Reduction
:color: info
:icon: info

`output.total` is already the global loss so should be used directly.

`model.loss()` sums each loss term's numerator and count across
`model.parallel_context.loss_group` before dividing. This gives every rank
the same global loss, even when DP replicas have different sequence lengths
or numbers of valid targets.

Each rank's backward pass follows its local computation, so DDP/FSDP should
still combine parameter gradients across WORLD.
:::

:::{dropdown} Custom Objectives
:color: info
:icon: info

Use `.embed()` or a direct model call to define a custom loss.
`MetricTree.distributed_reduce()` can be used to combine loss sums and
counts across ranks to compute the global mean. The following example
uses the mean squared embedding as a loss objective:

```{code-block} python
from alphagenome_pt import LossLeaf, MetricTree

optimizer.zero_grad(set_to_none=True)
embeddings = model.embed(batch)
values = embeddings.embeddings_1bp.square()
local_tree = MetricTree({
    "embedding_penalty": LossLeaf(values.sum(), values.numel()),
})
global_tree = local_tree.distributed_reduce(model.parallel_context.loss_group)
global_tree.total_loss().backward()
optimizer.step()
```
:::

### Checkpointing

Parallel models use the [ordinary checkpoint format](training.md#save-and-load-a-model):

```{code-block} python
from alphagenome_pt import AlphaGenome

# Save on every rank before destroying the process group.
# Only WORLD rank zero writes the checkpoint.
model.save("checkpoint/model")
dist.barrier()

# Load on every rank, then apply the desired strategy and mesh.
base_model = AlphaGenome.load("checkpoint/model", device=device)
model = distribute_alphagenome(
    base_model,
    strategy="sp_ddp",
    mesh=mesh,
    device_ids=[device.index],
)
```

:::{dropdown} Helpful Notes
:color: info
:icon: info

- You can choose a different parallel layout when loading.
- Create the optimizer after wrapping, using `model.parameters()`, so it uses
  the parameters managed by FSDP.
- Saving and restoring the optimizer state is the responsibility of the user since the
  model checkpoints currently do not include it.
:::

:::{dropdown} Why Every Rank Calls Save
:color: info
:icon: info

FSDP gathers full parameters during saving and can hang if only rank zero calls `.save()`,
whereas DDP permits rank-zero-only saving. We advise calling `.save()` on all ranks because
it works with either wrapper.
:::


## Example Parallel Training Script

The script below runs three training steps on four GPUs with DP=2,
SP=2, a small model, and synthetic ATAC targets:

:::{dropdown} Complete Training Script
:color: info
:icon: info

```{code-block} python
:caption: train_parallel.py

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from alphagenome_pt import (
    HeadName,
    distribute_alphagenome,
    small_alphagenome,
    sp_broadcast_data_batch,
    synthetic_batch,
    synthetic_metadata,
)


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device)

torch.manual_seed(42)  # Identical metadata and initial parameters on all ranks.
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "sp"))
sequence_length = 16_384
metadata = synthetic_metadata(heads=(HeadName.ATAC,), num_organisms=1)
base_model = small_alphagenome(
    metadata, max_seq_len=sequence_length, dtype_policy="float32",
).to(device)
model = distribute_alphagenome(
    base_model,
    strategy="sp_ddp",
    mesh=mesh,
    device_ids=[device.index],
).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create a different batch per DP replica and share it with its SP ranks.
context = model.parallel_context
batch = None
if context.sp_mesh_index == 0:
    torch.manual_seed(42 + context.dp_mesh_index)
    batch = synthetic_batch(metadata, batch_size=1, seq_len=sequence_length)
batch = sp_broadcast_data_batch(batch, parallel_context=context)

# Reuse the batch to demonstrate fitting (memorizing) the same data.
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    output = model.loss(batch)
    output.total.backward()
    optimizer.step()
    if dist.get_rank() == 0:
        print(f"step={step + 1} loss={output.total.item():.6f}", flush=True)

dist.destroy_process_group()
```
:::

Save the script as `train_parallel.py` and launch with:

```{code-block} bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc-per-node=4 train_parallel.py
```

To use FSDP, change `strategy` to `"sp_fsdp"` and replace
`device_ids=[device.index]` with `device_id=device`.

See [Synthetic Utilities](../development/synthetic-utilities.md) for the 
synthetic model and batch constructors used here.
