# Environment Setup
This README shows how to set up an environment to map the AlphaGenome params/state. This tutorial only covers setting up a UV environment as of now, although there should be a large amount of transferability to other dependency managers (e.g. Anaconda, pip).

You should NOT need to run these steps unless you want to re-convert the pytorch keys, but it is MUCH easier
to just retrieve the converted keys which are posted on huggingface (see `src/alphagenome_pt/checkpoint.py`).


## Instructions
```bash
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init JAX2PT --bare --python 3.11 && cd JAX2PT
uv venv jax2pt --python 3.11 --native-tls && source jax2pt/bin/activate
uv pip install alphagenome
git clone https://github.com/google-deepmind/alphagenome_research.git && uv pip install -e alphagenome_research
uv pip install torch einops einx aiohttp requests huggingface_hub pytest 
```

## Convert Checkpoint
From this directory:
```bash
pytest -s test_mappings.py
python convert_checkpoint.py
```

The conversion script writes `alphagenome_converted_state_dict.pt` in this directory by default.

## Inspect Keys
From this directory:
```bash
python inspect_jax.py --cpu
python inspect_torch.py
```


## Notes
Consider running `export UV_CACHE_DIR=<large_disk_path>` before install. This avoids UV cache OOM/disk-full errors (especially helpful on computational systems where home directories have strict storage quotas).
