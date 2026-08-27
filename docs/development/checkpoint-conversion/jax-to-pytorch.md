# JAX-to-PyTorch Conversion

{bdg-warning}`Maintainer workflow`

This workflow reproduces the PyTorch state dictionaries and metadata from the
official JAX release. Most users should load the preconverted artifacts
described in [DeepMind Checkpoints](../../model/deepmind-checkpoints.md).


## Install Conversion Dependencies

Conversion requires Python 3.11 or newer because of the public
`alphagenome_research` loader. Run the following command from the AlphaGenome
PyTorch repository root.

:::{dropdown} Conversion Environment Size
:color: warning
:icon: alert

The optional conversion environment can be large because it includes JAX and
the public AlphaGenome loader.
:::

```{code-block} bash
:caption: Install AlphaGenome PyTorch conversion dependencies

python -m pip install -e ".[jax2pt]"
```

```{code-block} bash
:caption: Install the official public loader

git clone https://github.com/google-deepmind/alphagenome_research.git \
  ../alphagenome_research
python -m pip install -e ../alphagenome_research
```

Loading official checkpoints through Hugging Face requires accepting the [AlphaGenome model terms](https://deepmind.google.com/science/alphagenome/model-terms) and authenticating through saved credentials or `HF_TOKEN`.

## Convert Checkpoint State

:::{dropdown} Conversion Resource Requirements
:color: warning
:icon: alert

Converting all five checkpoint variants can require substantial memory,
download time, and storage. Use `--fold` when only one variant is needed.
:::

```{code-block} bash
:caption: Convert every checkpoint variant

python -m alphagenome_pt.jax2pt.convert_state \
  --torch-output-dir checkpoints
```

:::{container} long-table

| Argument | Default | Behavior |
| --- | --- | --- |
| `--fold` | `None` | Supplied: convert `all_folds` or one of `fold_0` through `fold_3`. Omitted: convert all five |
| `--all-checkpoints` | `False` | Convert all five explicitly. Cannot be combined with `--fold` |
| `--jax-input-dir` | `None` | `None`: load through the official Hugging Face loader. Supplied: load each fold from a same-named JAX checkpoint directory under this path |
| `--torch-output-dir` | `.` | Write each converted state dictionary as `alphagenome_<fold>.pt` |
| `--device` | `cpu` | JAX backend used to load the checkpoint. The first matching device is selected |

:::

:::{dropdown} Convert an Already Downloaded Orbax Checkpoint

Supply the parent of the selected fold's complete Orbax checkpoint directory:

```text
jax-checkpoints/
└── fold_0/
    ├── _CHECKPOINT_METADATA
    ├── _METADATA
    ├── d/
    │   └── ...
    ├── manifest.ocdbt
    └── ocdbt.process_0/
        ├── d/
        │   └── ...
        └── manifest.ocdbt
```

Each fold directory must contain the complete Orbax checkpoint. `--jax-input-dir` specifies the parent directory, and the converter appends the selected fold name.

```{code-block} bash
:caption: Convert one local Orbax checkpoint

python -m alphagenome_pt.jax2pt.convert_state \
  --fold fold_0 \
  --jax-input-dir jax-checkpoints \
  --torch-output-dir checkpoints
```

:::

:::{dropdown} Output Format
:color: info
:icon: info

Each output is a bare PyTorch state dictionary containing model parameters and
persistent state. Metadata-derived means and masks are rebuilt as
nonpersistent buffers during model construction.
:::

## Convert Metadata

```bash
python -m alphagenome_pt.jax2pt.convert_metadata \
  --output checkpoints/alphagenome_metadata.json \
  --raw-output checkpoints/alphagenome_metadata_raw.json \
  --summary-output checkpoints/alphagenome_metadata_summary.json
```

| File | Contents |
| --- | --- |
| `alphagenome_metadata.json` | Model-ready organisms, head sizes, means, and padding masks |
| `alphagenome_metadata_raw.json` | Public human and mouse metadata serialized before package conversion |
| `alphagenome_metadata_summary.json` | Track, tissue, and padding counts for inspection |

The same converted metadata applies to every checkpoint fold.

## Verify Local Artifacts

:::{dropdown} Verify Local Files
:color: warning
:icon: alert

The file assertions ensure this check uses the local conversion. Without them,
the loading helpers can resolve a missing artifact through Hugging Face.
:::

```{code-block} python
:caption: Verify one converted checkpoint against its metadata

from pathlib import Path

from alphagenome_pt import deepmind_metadata, deepmind_model, load_deepmind_state

artifact_dir = Path("checkpoints")
fold = "all_folds"

assert (artifact_dir / "alphagenome_metadata.json").is_file()
assert (artifact_dir / f"alphagenome_{fold}.pt").is_file()

metadata = deepmind_metadata(artifact_dir)
model = deepmind_model(metadata=metadata)
result = load_deepmind_state(
    model,
    local_dir=artifact_dir,
    fold=fold,
)

assert not result.missing_keys
assert not result.ignored_missing_keys
assert not result.unexpected_keys
```

## Run Conversion Tests

```{code-block} bash
:caption: Install test dependencies

python -m pip install -e ".[dev]"
```

The fixture-based metadata tests check conversion and summary behavior without external downloads:

```{code-block} bash
:caption: Run fixture-based metadata conversion tests

python -m pytest tests/test_mapping_metadata.py
```

:::{dropdown} Run the Official State-Mapping Test
:color: warning
:icon: alert

This optional test loads the official `all_folds` checkpoint, checks mapping
coverage and tensor shapes, and loads the converted state into the published
PyTorch model. It requires official checkpoint access and substantial memory,
download time, and storage.

```{code-block} bash
:caption: Run the opt-in JAX state-mapping test

ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST=1 \
  python -m pytest -s tests/test_mapping_state.py
```

:::

After verifying a complete artifact set, continue to
[Publish Converted Artifacts](publishing.md).
