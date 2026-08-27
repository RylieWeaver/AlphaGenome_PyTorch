# Configuration

Choose the construction pattern closest to your use case: [published
inference](#inference), [fine-tuning with compatible checkpoint
state](#fine-tuning), or [random initialization from scratch](#from-scratch).
These are common starting points rather than restrictions on how each model can
be used.


## Inference

Construct the published model and load its converted PyTorch checkpoint:

```{code-block} python
from alphagenome_pt import deepmind_model

model = deepmind_model(load_state=True)
model.eval()
```

See [DeepMind Checkpoints](deepmind-checkpoints.md) for `deepmind_model()` arguments and loading behavior, and [Inference](inference.md) for model inputs and prediction calls.


## Fine-Tuning

Fine-tuning can start from the published model above or customize its organisms and prediction tracks through metadata:

```{code-block} python
from alphagenome_pt import deepmind_model

model = deepmind_model(
    metadata=metadata,
    load_state=True,
)
```

See [Data and Metadata](../background/data-and-metadata.md#metadata) for metadata configuration, [DeepMind Checkpoints](deepmind-checkpoints.md) for `deepmind_model()` arguments and loading behavior, [Flexible Loading](deepmind-checkpoints.md#flexible-loading) for organism and head mapping, and [Training](training.md) for the training workflow.


## From Scratch

Construct `AlphaGenomeConfig` directly to configure the model architecture and runtime behavior.

:::{important}
A [`Metadata`](../background/data-and-metadata.md#metadata) object or dictionary is required because it defines the model's organisms and prediction heads.
:::

```{code-block} python
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig

config = AlphaGenomeConfig(
    max_seq_len=8_192,
    num_channels=96,
    metadata=metadata,
)
model = AlphaGenome(config)
```

This path initializes the model without loading checkpoint state.

### Hyperparameters

The table compares `AlphaGenomeConfig` defaults with the values used by `deepmind_config()` to construct a model compatible with the published checkpoints. It does not reproduce the original training recipe. Arrows indicate defaults derived when an argument is `None`.

:::{div} long-table
| Argument | Default | `deepmind_config()` | Behavior |
| --- | --- | --- | --- |
| `max_seq_len` | 8,192 | 1,048,576 | Maximum 1-bp input length |
| `num_channels` | 768 | 768 | Base number of channels |
| `channel_increment` | `None` &rarr; `num_channels // 6` | 128 | Encoder/decoder channel-width increment/decrement per stage |
| `transformer_layers` | 9 | 9 | Number of transformer blocks |
| `first_conv_width` | 15 | 15 | First encoder convolution kernel width |
| `block_width` | 5 | 5 | Encoder/decoder convolution kernel width |
| `num_q_heads` | 8 | 8 | Query head count for grouped-query attention |
| `num_kv_heads` | 1 | 1 | Key/value head count for grouped-query attention |
| `qk_head_dim` | `None` &rarr; `2 * (num_channels // 12)` | 128 | Per-head query/key width |
| `v_head_dim` | `None` &rarr; `2 * (num_channels // 8)` | 192 | Per-head value width |
| `pair_channels` | `None` &rarr; `num_channels // 6` | 128 | Pair-state and embedding width |
| `pair_heads` | 32 | 32 | Sequence-to-pair head count |
| `pos_channels` | 64 | 64 | Sequence-to-pair position width (odd values warn and use the next even value) |
| `transformer_mlp_ratio` | 2 | 2 | Transformer/pair MLP expansion ratio |
| `init_scale` | 0.9 | 0.9 | Decoder upsampling-scale initialization. Loaded checkpoint state replaces it |
| `embedder_mlp_ratio` | 2 | 2 | 1-bp/128-bp embedding width multiplier |
| `num_splice_sites` | `None` &rarr; `max_seq_len // 2048` | 512 | Maximum generated candidates per donor/acceptor and strand group |
| `splice_site_channels` | `None` &rarr; `num_channels` | 768 | Splice-junction latent width |
| `splice_site_threshold` | 0.1 | 0.1 | Minimum probability for generated splice candidates |
| `metadata` | `None` | Supplied metadata or published metadata when omitted | Organisms, heads, output widths, means, and masks |
:::

<br>

:::{dropdown} Configuration Requirements
:color: warning
:icon: alert

- `max_seq_len` and every input sequence length must be at least 2,048 and
  divisible by 2,048. Inputs cannot exceed `max_seq_len`.
- `num_q_heads` and `num_kv_heads` must be positive, and `num_q_heads` must be
  divisible by `num_kv_heads`.
- `qk_head_dim` must be positive and even for RoPE. `v_head_dim` must be
  positive.
- `splice_site_channels` must be positive and even for RoPE when the
  splice-junction head is used.
- `transformer_layers` must be at least `1` to create a pair representation.
- `first_conv_width` and `block_width` must be positive and odd so convolution
  padding preserves sequence length.

:::

:::{admonition} Derived Defaults
:class: note

Derived defaults let you scale the model by changing only `num_channels` and `max_seq_len`. Related dimensions update automatically, but each can still be set explicitly.
:::

The numbers of encoder/decoder stages and their pooling/upsampling factors are fixed. See [Model Architecture](../background/model-architecture.md) for the complete component layout and how configurable settings affect model shapes and computation.

#### Runtime and Loss Settings

| Argument | Default | Behavior |
| --- | --- | --- |
| `dropout` | 0.0 | Transformer and pair dropout during train mode |
| `sync_bn` | `True` | Synchronizes BatchNorm statistics across distributed processes during train mode |
| `min_zero_multinomial_loss` | `True` | Zero-shifts positional genome-track loss when `True` |

These settings do not affect checkpoint compatibility. `dropout` and `sync_bn` are inactive in evaluation mode, while `min_zero_multinomial_loss` affects only loss computation. Setting it to `False` matches the released JAX loss definition.


## Precision and Dtype Policies

A dtype policy controls floating-point operations throughout the model
and can be chosen during any model construction:
`AlphaGenomeConfig`:

```{code-block} python
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, deepmind_model

published_model = deepmind_model(
    load_state=True,
    dtype_policy="deepmind",
)

custom_model = AlphaGenome(
    AlphaGenomeConfig(
        metadata=metadata,
        dtype_policy="deepmind",
    )
)
```

| Policy | Parameters | Inputs | General compute | Selected upcasts | Outputs |
| --- | --- | --- | --- | --- | --- |
| `deepmind` | FP32 | FP32 | BF16 | FP32 | BF16 |
| `fp32_params_bf16_compute_fp32_compute_uptype_bf16_output` | FP32 | FP32 | BF16 | FP32 | BF16 |
| `bfloat16` | BF16 | BF16 | BF16 | FP32 | BF16 |
| `float32` | FP32 | FP32 | FP32 | FP32 | FP32 |
| `float64` | FP64 | FP64 | FP64 | FP64 | FP64 |

:::{dropdown} Dtype policies
:class: note
The dtype policies balance computational speed/memory and numerical precision.
`deepmind` is the default and matches the published JAX mixed-precision policy.
`float64` offers the highest precision but is intended mainly for numerical
validation because it requires the most time and memory.

`Outputs` refers to values returned by the public `embed()`, `predict()`, and
`loss()` methods. The complete `LossOutput`, including its metric tree and
total, is cast to `output_dtype`. Internal embeddings and predictions remain in
their compute dtypes as they pass between model stages.
:::

:::{dropdown} What Is Affected by Compute Uptype?
:class: note
The compute uptype is generally used to upcast numerically sensitive operations
(e.g., normalization statistics, attention and splice-junction contractions,
softmax, and loss reductions).
:::


## Save and Load a Configuration

`AlphaGenomeConfig.save()` and `AlphaGenomeConfig.load()` round-trip the
configuration and metadata through separate files:

```{code-block} python
from alphagenome_pt import AlphaGenomeConfig

config.save(
    cfg_path="config.json",
    metadata_path="metadata.pt",
)
loaded_config = AlphaGenomeConfig.load(
    cfg_path="config.json",
    metadata_path="metadata.pt",
)
```

:::{admonition} Configuration Files Do Not Include Model State
:class: note

This saves configuration and metadata, not model state. See [Save and Load a Model](training.md#save-and-load-a-model) to round-trip a complete model.
:::
