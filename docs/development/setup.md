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

The normal suite skips tests that require network access, official JAX checkpoint state, or `torchrun`. Run these additional checks explicitly when relevant:

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

## DeepMind JAX Equivalence Tests

Verify equivalence between the JAX and PyTorch implementations. These tests are
kept separate because JAX adds substantial dependencies.

```{code-block} bash
:caption: Install equivalence-test dependencies

python -m pip install -e ".[dev,equivalence]"
```

:::{note}
The equivalence dependency pins `alphagenome_research` to commit
`1e55dcffb98ba26b31e74edc5e9f038f54c0e89d`, keeping the reference
implementation stable as its repository changes.
:::

```{code-block} bash
:caption: Run checkpoint-free tests

python -m pytest \
  tests/deepmind_equivalence/test_without_checkpoint.py \
  --equivalence-report tests/deepmind_equivalence/report.csv
```

:::{dropdown} What do checkpoint-free tests compare?
:color: info
:icon: info

These tests compare core operations and losses with deterministic inputs and
explicitly shared parameters. They do not use a checkpoint, so they can
localize mathematical differences independently of parameter and state
mapping.
:::

```{code-block} bash
:caption: Run checkpoint-backed tests

python -m pytest \
  tests/deepmind_equivalence/test_checkpoint.py \
  --run-equivalence \
  --checkpoint-equivalence-device cuda \
  --equivalence-sequence-length 4096 \
  --equivalence-report tests/deepmind_equivalence/report.csv
```

:::{dropdown} What do checkpoint-backed tests compare?
:color: info
:icon: info

These tests load the official JAX and converted PyTorch `all_folds`
checkpoints, then compare corresponding encoder, decoder, transformer,
pair-update, attention, and output-embedder modules with small deterministic
representations. The final test composes those parts into the complete model 
and compares its embeddings and prediction heads. At sequence lengths of at 
least 131,072 bp, it also compares losses. These tests provide the strongest 
equivalence evidence because they use the published checkpoint and execute 
the complete JAX model forward pass.
:::

```{code-block} bash
:caption: Run all equivalence tests

python -m pytest \
  tests/deepmind_equivalence/ \
  --run-equivalence \
  --checkpoint-equivalence-device cuda \
  --equivalence-sequence-length 4096 \
  --equivalence-report tests/deepmind_equivalence/report.csv
```

All checkpoint runs write one CSV row per compared representation for the
published DeepMind, full-FP32, and FP64 policies.

:::{dropdown} Inspect the generated equivalence report
:color: info
:icon: info

Each row describes one representation compared by one parametrized test under
one precision policy. View the
[formatted report](https://github.com/RylieWeaver/AlphaGenome_PyTorch/blob/main/tests/deepmind_equivalence/report.md)
or download the
[raw CSV](https://raw.githubusercontent.com/RylieWeaver/AlphaGenome_PyTorch/main/tests/deepmind_equivalence/report.csv).

| Column | Description |
| --- | --- |
| `test_name` | Pytest node ID that produced the comparison. |
| `dtype_policy` | Precision policy: `deepmind`, `float32`, or `float64`. |
| `representation` | Compared tensor, output, or loss within the test. |
| `relative_L2` | L2 norm of the difference divided by the JAX-reference L2 norm. |
| `relative_Linf` | Maximum absolute difference divided by the maximum absolute JAX-reference value. |
| `max_abs` | Maximum absolute elementwise difference. |
| `mean_abs` | Mean absolute elementwise difference. |
| `reference_max_abs` | Maximum absolute JAX-reference value. |
| `reference_mean_abs` | Mean absolute JAX-reference value. |
| `exact_fraction` | Fraction of finite values that match exactly. |
| `num_values` | Number of finite values included in the metrics. |
| `pytorch_dtype` | Observed dtype of the PyTorch representation. |
| `jax_dtype` | Observed dtype of the JAX representation. |
| `dtype_match` | Whether the observed framework dtypes match. A mismatch fails the test. |
:::

To use chromosome-1 reference sequence instead of synthetic DNA, first download
1,048,576 bases each for human and mouse:

```{code-block} bash
python tests/deepmind_equivalence/download_dna.py
```

Then add the two files to the test invocation:

```{code-block} bash
:caption: Run all equivalence tests with reference DNA

python -m pytest \
  tests/deepmind_equivalence/ \
  --run-equivalence \
  --checkpoint-equivalence-device cuda \
  --equivalence-sequence-length 131072 \
  --equivalence-report tests/deepmind_equivalence/report.csv \
  --equivalence-dna \
    tests/deepmind_equivalence/dna/human_hg38_chr1.fa \
    tests/deepmind_equivalence/dna/mouse_mm39_chr1.fa
```

The checked-in
[full equivalence report](https://github.com/RylieWeaver/AlphaGenome_PyTorch/blob/main/tests/deepmind_equivalence/report.md)
and its
[downloadable CSV](https://raw.githubusercontent.com/RylieWeaver/AlphaGenome_PyTorch/main/tests/deepmind_equivalence/report.csv)
were generated from this 131,072-bp human and mouse reference-sequence
comparison.

:::{dropdown} Full equivalence report
:color: info
:icon: info

```{include} ../../tests/deepmind_equivalence/report.md
:start-line: 2
```
:::

:::{dropdown} Custom DNA input
:class: note
The option accepts FASTA or plain-text DNA containing at least the requested
sequence length. One or two paths may be provided and are assigned to organism
indices `[0, 1]` in that order. If only one path is provided, it is used for
both organisms.
:::

:::{caution}
Checkpoint-backed comparisons require substantial memory and computation
because they keep both framework checkpoints on the selected device. Lengths
below 131,072 bp still compare embeddings and predictions but skip the
checkpoint-backed loss. The default is 4,096 bp and 131,072 bp is the minimum
length for a complete comparison that includes losses.

The splice-junction comparison defaults to one splice site per 2,048 bp, with
a minimum of 16 and a maximum of 512.

Set `--equivalence-sequence-length 1048576` to check the published maximum if
memory and computation allow.
:::

See [DeepMind JAX Equivalence](../deepmind-equivalence.md)
for the comparison design and limits.
