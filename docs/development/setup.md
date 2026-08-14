# Development Setup

## Install from a Checkout

Clone the repository and install the package with its test dependencies:

```bash
git clone https://github.com/RylieWeaver/AlphaGenome_PyTorch.git
cd AlphaGenome_PyTorch
python -m pip install -e ".[dev]"
```

The editable installation makes changes under `src/` available without reinstalling the package. Run the remaining commands on this page from the repository root.

## Run Tests

```bash
python -m pytest tests
```

The normal suite skips tests that require network access, official JAX checkpoint state, or `torchrun`. Run these test-suite checks explicitly when relevant:

### Optional Checks

| Check | Command | Requirements |
| --- | --- | --- |
| Hugging Face downloads | `ALPHAGENOME_PT_RUN_HF_DOWNLOAD_TEST=1 python -m pytest tests/test_checkpoint_download.py` | Network access and storage for converted artifacts |
| JAX state mapping | `ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST=1 python -m pytest -s tests/test_mapping_state.py` | [Checkpoint Conversion](checkpoint-conversion/index.md) setup and official checkpoint access |
| DDP loss gradients | `torchrun --standalone --nproc_per_node=1 --module tests.test_model_ddp_loss` | PyTorch distributed support |
| Distributed batch normalization | `torchrun --nproc_per_node=2 tests/test_bn_grad.py` | Two workers and PyTorch distributed support |

:::{dropdown} Optional Check Resource Requirements
:color: warning
:icon: alert

The Hugging Face and JAX mapping checks can download large checkpoint files.
The distributed checks require launching separate worker processes.
:::
