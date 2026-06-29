# Tests

Run the normal test suite from the repo root. This skips slow/networked tests:

```bash
pytest tests
```

For local development, install test dependencies first:

```bash
pip install -e ".[dev]"
```

## Run Everything

To run the full pytest suite, set the env vars:

```bash
ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST=1 pytest tests
```

The checkpoint tests will download large checkpoint/metadata files, so will be slower. 
The conversion parity tests require the optional JAX conversion dependencies:

```bash
pip install -e ".[jax2pt]"
```

The distributed batch-norm gradient check is not part of normal single-process
pytest and is run separately with `torchrun`:

```bash
torchrun --nproc_per_node=2 tests/test_bn_grad.py
```

## Downloading Tests

Download tests are skipped by default. Enable the Hugging Face checkpoint and
metadata download tests explicitly:

```bash
ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 pytest tests/test_checkpoint_download.py
```

These tests use `huggingface_hub`, which is part of the base package
dependencies, but they require network access and may download large files.

## Conversion Tests

Metadata conversion tests use small dummy metadata and run in the normal test
suite:

```bash
pytest tests/test_mapping_metadata.py
```

The state conversion parity test downloads/loads the official JAX checkpoint,
so it is skipped by default. Enable it explicitly:

```bash
export ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST=1
pytest tests/test_mapping_state.py
```

This requires the optional JAX conversion dependencies and access to the
official DeepMind checkpoint data.

It also requires the public `alphagenome_research` checkout used by the
official loader:

```bash
git clone https://github.com/google-deepmind/alphagenome_research.git
pip install -e alphagenome_research
```

## Distributed Tests

The batch-norm gradient test is intended for explicit `torchrun` use:

```bash
torchrun --nproc_per_node=2 tests/test_bn_grad.py
```

This requires a PyTorch installation with distributed support. On GPU systems,
make sure the installed PyTorch build matches the available CUDA runtime.

Optional tolerances and debug output:

```bash
torchrun --nproc_per_node=2 tests/test_bn_grad.py --atol 1e-4 --rtol 1e-4 --print
```

The same options can be passed through env vars when invoking pytest directly:

```bash
export ALPHAGENOME_PT_BN_GRAD_ATOL=1e-4
export ALPHAGENOME_PT_BN_GRAD_RTOL=1e-4
export ALPHAGENOME_PT_BN_GRAD_PRINT=1
pytest tests/test_bn_grad.py
```

## Developer Notes

Pytest injects arguments with fixture names automatically. For example,
`tmp_path` is a fresh temporary `pathlib.Path` for that test, useful for
checkpoint files or other throwaway outputs.

`monkeypatch` temporarily changes Python objects or environment variables for
one test. This is useful for replacing slow or networked functions with a fake
implementation, then asserting the outer code called that fake with the
expected arguments.
