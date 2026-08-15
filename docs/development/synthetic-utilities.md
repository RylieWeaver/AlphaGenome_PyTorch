# Synthetic Helpers

`synthetic_metadata()`, `small_alphagenome()`, and `synthetic_batch()` create compatible `Metadata`, `AlphaGenome`, and `DataBatch` objects for testing. See [Data and Metadata](../background/data-and-metadata.md) for the real input contracts and [Configuration](../model/configuration.md) for full model settings.


## Test a Model

Select the heads once, then pass the same metadata to the model and batch:

```{code-block} python
:caption: Build a compatible model and batch, then check gradients

from alphagenome_pt import (
    HeadName,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)

metadata = synthetic_metadata(
    heads=(HeadName.RNA_SEQ, HeadName.CONTACT_MAPS),
    organisms=("human", "mouse"),
    num_tracks=4,
)
model = small_alphagenome(metadata, max_seq_len=8_192)
batch = synthetic_batch(
    metadata,
    batch_size=2,
    seq_len=model.max_seq_len,
)

output = model.loss(batch)
output.total.backward()
```


## Synthetic Function Arguments

### `synthetic_metadata()`

| Argument | Default | Behavior |
| --- | --- | --- |
| `heads` | `None` | Heads to include. `None` creates no heads. Selecting `HeadName.SPLICE_SITES_JUNCTION` also adds `HeadName.SPLICE_SITES_CLASSIFICATION` |
| `num_organisms` | `2` | Number of generated names (`organism_0`, `organism_1`, and so on). Used only when `organisms` is omitted |
| `organisms` | `None` | Explicit organism names. When supplied, their count replaces `num_organisms` |
| `num_tracks` | `None` | Uses four tracks or junction tissues per organism. Splice-site classification always uses five classes, while masked language modeling ignores this setting |

The helper returns `Metadata` with derived masks and random positive means for genome-track heads.

### `small_alphagenome()`

| Argument | Default | Behavior |
| --- | --- | --- |
| `metadata` | `None` | Metadata used to construct the model. `None` calls `synthetic_metadata()` and therefore creates no prediction heads |
| `**cfg_overrides` | Omitted | `AlphaGenomeConfig` overrides. The reduced defaults are `max_seq_len=16_384`, `num_channels=64`, and `transformer_layers=3` |

The returned model is randomly initialized. Pass the same metadata to `synthetic_batch()` to generate compatible targets.

### `synthetic_batch()`

| Argument | Default | Behavior |
| --- | --- | --- |
| `metadata` | `None` | Determines the generated targets and masks. When omitted, the helper creates metadata containing every built-in head |
| `batch_size` | `2` | Number of examples |
| `seq_len` | `8192` | 1-bp sequence length. This is not inferred from a model. Model inputs must be at least 2,048 bp, divisible by 2,048, and no longer than `model.max_seq_len` |
| `num_splice_sites` | `2` | Candidate count per donor/acceptor and strand group. Used only for splice junctions |

:::{admonition} Generated DataBatch
:class: note

The returned `DataBatch` contains:

- Random A/C/G/T/N DNA encoded as $[\mathrm{B}, \mathrm{S}_1, 4]$. Encoded
  `N` bases contain four zeros.
- Organism indices that cycle through the metadata organisms.
- Targets for every head present in the metadata.

:::

## Notes

:::{dropdown} Generated Organisms and Masks
:color: info
:icon: info

Generated organisms have equal per-head track or tissue counts, and generated
batch masks are all `True`. Use custom metadata to test organism padding and
modify the batch masks to test sample-specific missing data.
:::

Helpers use PyTorch's global random-number generator and create models and
tensors on CPU. Use `torch.manual_seed()` for repeatability and move the
returned objects when testing another device.
