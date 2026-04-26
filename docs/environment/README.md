# Environment Setup
This README shows how to set up an environment to run AlphaGenome. This tutorial only covers setting up a UV environment as of now, although there should be a large amount of transferability to other dependency managers (e.g. Anaconda, pip).

Choose one of the options below...

## Run a Script
You can also include arguments for the project and environment name (e.g. ./make_env.sh <project_name> <env_name> <directory> [python_version])
```bash
cd */AlphaGenome_PyTorch/docs/environment
chmod +x make_env.sh
./make_env.sh
```

## Do it Manually
```bash
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init AlphaGenome --bare --python 3.12
cd AlphaGenome
uv venv ag-env --python 3.12 --native-tls
source ag-env/bin/activate
uv pip install torch numpy einops huggingface_hub
```


## Notes
Consider running `export UV_CACHE_DIR=<large_disk_path>` before install. This avoids UV cache OOM/disk-full errors (especially helpful on computational systems where home directories have strict storage quotas).
