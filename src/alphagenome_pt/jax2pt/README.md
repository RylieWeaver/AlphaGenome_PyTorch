# JAX to PyTorch Conversion
This README contains information related to mapping the official AlphaGenome JAX params/state into this package's PyTorch state dict.

Normally, you should be able to get converted checkpoints from HuggingFace and not need to run these steps. However, if the Hugging Face checkpoint is broken, or for some other reason you want to convert the JAX checkpoint, you can use this tooling.


## Install
From this repository root, install the package in editable mode with the JAX
conversion extra.

```bash
pip install -e ".[jax2pt]"
cd /path/to/workspace
git clone https://github.com/google-deepmind/alphagenome_research.git
pip install -e alphagenome_research
```

If you only want to run the module form from a source checkout without
installing the package, set `PYTHONPATH` from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m alphagenome_pt.jax2pt.convert_state
```


## Convert Checkpoint
Module form:

```bash
python -m alphagenome_pt.jax2pt.convert_state
```

Arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--fold` | `str` | none | Convert one official checkpoint fold: `all_folds`, `fold_0`, `fold_1`, `fold_2`, or `fold_3`. If omitted, every supported checkpoint is converted. |
| `--all-checkpoints` | `flag` | off | Explicitly convert every supported checkpoint. This is also the default when `--fold` is omitted. |
| `--jax-input-dir` | `path` | none | Local official JAX checkpoint directory. If set, it must contain checkpoint files named `alphagenome_all_folds.pt`, `alphagenome_fold_0.pt`, `alphagenome_fold_1.pt`, `alphagenome_fold_2.pt`, and `alphagenome_fold_3.pt`. |
| `--torch-output-dir` | `path` | `.` | Directory where the converted checkpoint is written as `alphagenome_{fold_name}.pt`. |
| `--device` | `str` | `cpu` | Device string for JAX checkpoint loading. Examples: `cpu`, `gpu`, `cuda`. |

The converted state dict contains only persistent checkpoint tensors. Metadata-derived track means and masks are non-persistent model buffers built from metadata during model initialization.


## Testing

```bash
pytest -s tests/test_mappings.py
```


## Inspect Keys
Module form:

```bash
python -m alphagenome_pt.jax2pt.inspect_state_jax
python -m alphagenome_pt.jax2pt.inspect_metadata_public
python -m alphagenome_pt.jax2pt.inspect_metadata_table
python -m alphagenome_pt.jax2pt.inspect_state_torch
```

JAX inspect arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--fold` | `str` | `all_folds` | Official checkpoint fold to inspect: `all_folds`, `fold_0`, `fold_1`, `fold_2`, or `fold_3`. |
| `--model-path` | `path` | none | Local official JAX checkpoint path. If set, inspection uses this path instead of loading `--fold` from Hugging Face. |
| `--device` | `str` | `cpu` | Device string. Defaults to CPU. Examples: `cpu`, `gpu`, `cuda`, `mps`. |
| `--output`, `-o` | `path` | `jax_params.txt` | Writes JAX parameter/state key and shape summary to this path. |

Metadata inspect arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--output`, `-o` | `path` | `alphagenome_metadata_raw.json` | Writes raw public AlphaGenome JAX metadata JSON to this path. |

Metadata table inspect arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--organism` | `str` | `human` | Organism table to inspect: `human`, `mouse`, or `all`. |
| `--head` | `str` | `procap` | Output metadata table to inspect, or `all`. |
| `--output`, `-o` | `path` | `alphagenome_metadata_table.csv` | Writes the selected public metadata dataframe. Use `.csv`, `.json`, `.pkl`, `.pickle`, or `.parquet`. |

Torch inspect arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--device` | `str` | `cpu` | Device string. Defaults to CPU. Examples: `cpu`, `gpu`, `cuda`, `mps`. |
| `--output`, `-o` | `path` | `torch_params.txt` | Writes PyTorch parameter/state key and shape summary to this path. |

## Export Public Metadata
Export public JAX metadata into three files: raw metadata, converted loadable metadata, and a compact summary for inspection.

Module form:

```bash
python -m alphagenome_pt.jax2pt.convert_metadata
```

Arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--output`, `-o` | `path` | `alphagenome_metadata.json` | Writes the converted loadable metadata JSON to this path. |
| `--raw-output` | `path` | `alphagenome_metadata_raw.json` | Writes the raw public AlphaGenome metadata JSON to this path. |
| `--summary-output` | `path` | `alphagenome_metadata_summary.json` | Writes a compact derived summary for human inspection. |


## Upload Checkpoints to Hugging Face (For Maintainer)
Helper for uploading converted checkpoints to Hugging Face:

```bash
python -m alphagenome_pt.jax2pt.hf_upload --local-dir /path/to/checkpoints
```

By default, this uploads `alphagenome_all_folds.pt`, `alphagenome_fold_0.pt`, `alphagenome_fold_1.pt`, `alphagenome_fold_2.pt`, `alphagenome_fold_3.pt`, `alphagenome_metadata.json`, `alphagenome_metadata_raw.json`, and `alphagenome_metadata_summary.json` into `v{package-version}/` in the Hub repo. The metadata files live beside the checkpoints so runtime metadata loading can use the same local/Hugging Face fallback pattern as checkpoint loading.

Arguments:

| Argument | Type | Default | Values / behavior |
| --- | --- | --- | --- |
| `--local-dir` | `path` | `.` | Local directory containing converted checkpoint files plus converted and raw metadata JSON. |
| `--repo-id` | `str` | `RylieWeaver/alphagenome-pytorch` | Hugging Face model repo to upload to. |
| `--repo-dir` | `str` | `v{package-version}` | Directory inside the Hub repo. |
| `--dry-run` | `flag` | off | Prints upload targets without creating the repo or uploading files. |


## Notes
If using UV, consider running `export UV_CACHE_DIR=<large_disk_path>` before install. This avoids UV cache OOM/disk-full errors, especially on computational systems where home directories have strict storage quotas.
