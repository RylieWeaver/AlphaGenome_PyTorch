# Environment Setup

Most users should install the Python package directly:

```bash
pip install alphagenome-pt
```

When working from a local checkout, install the package in editable mode from
the repo root:

```bash
pip install -e .
```

The instructions below are primarily for development, source installs, or
systems where you want a fully controlled local environment. They may change as
the package and optional dependencies evolve.

## Developer Setup With Script

The helper script creates a `uv` environment and installs the core runtime
dependencies. You can pass a project name, environment name, target directory,
and Python version:

```bash
cd */AlphaGenome_PyTorch/docs/environment
chmod +x make_env.sh
./make_env.sh <project_name> <env_name> <directory> [python_version]
```

Example:

```bash
./make_env.sh AlphaGenome ag-env /path/to/workspace 3.12
source /path/to/workspace/AlphaGenome/ag-env/bin/activate
cd /path/to/AlphaGenome_PyTorch
pip install -e ".[dev]"
```

## Developer Setup Manually

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init AlphaGenome --bare --python 3.12
cd AlphaGenome
uv venv ag-env --python 3.12 --native-tls
source ag-env/bin/activate
uv pip install torch numpy einops einx huggingface_hub
cd /path/to/AlphaGenome_PyTorch
pip install -e ".[dev]"
```

For JAX-to-PyTorch checkpoint conversion tooling, install the optional extra:

```bash
pip install -e ".[jax2pt]"
```

## Notes

On systems with small home directories, consider setting `UV_CACHE_DIR` before
installing dependencies:

```bash
export UV_CACHE_DIR=<large_disk_path>
```

This avoids UV cache disk-full errors on shared compute systems.
