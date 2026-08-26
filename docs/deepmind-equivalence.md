# DeepMind JAX Equivalence

AlphaGenome PyTorch is verified against the public DeepMind JAX implementation
at three levels, each satisfying explicit numerical criteria:

- **Single Module:** Core repeated operations, losses, and isolated model modules.
- **Architecture:** Higher-level modules (encoder, transformer, decoder, output embedder).
- **Full Model:** End-to-end outputs (embeddings, predictions, losses).

The full-model comparison is the strongest evidence of equivalence. We include the smaller
comparisons here as well because they helped localize differences during debugging and show
the full scope of what has been verified.

## Models and Checkpoints

| Artifact | Selection |
| --- | --- |
| JAX implementation | [`alphagenome_research` commit `1e55dcf`](https://github.com/google-deepmind/alphagenome_research/commit/1e55dcffb98ba26b31e74edc5e9f038f54c0e89d) |
| Source model state | Official JAX `all_folds` checkpoint from [`google/alphagenome-all-folds`](https://huggingface.co/google/alphagenome-all-folds) |
| Converted model state | Corresponding `alphagenome_all_folds.pt`, derived with this repository's converter and loaded from [`RylieWeaver/alphagenome-pytorch`](https://huggingface.co/RylieWeaver/alphagenome-pytorch) |

## Test Inputs

| | Full model | Checkpoint modules | Without checkpoint |
| --- | --- | --- | --- |
| Parameters | Full checkpoint | Checkpoint slices | Synthetic |
| Inputs | DNA and Organism Indices `[0, 1]` | Synthetic | Synthetic |
| Targets (as needed) | Synthetic | Synthetic | Synthetic |

The full model uses 4,096-bp synthetic DNA by default. `--equivalence-dna`
accepts one or two FASTA files to use for the full model DNA input, truncated to
`--equivalence-sequence-length`. By default, the test selects one splice site
per 2,048 bp, with a minimum of 16 and a maximum of 512, using a 0.1 prediction
threshold.

:::{dropdown} How are synthetic values constructed?
:color: info
:icon: info

Synthetic generation is deterministic. Activations generally use fixed-seed
draws from $\mathcal{N}(0,1)$. Synthetic parameters generally use evenly spaced
values within fan-in-scaled bounds similar to default PyTorch initialization,
while targets and other domain-constrained quantities use test-specific intervals.

Some operations deliberately use inputs outside these general patterns. For
example, GELU uses an evenly spaced grid over $[-4, 4]$ to cover its nonlinear
transition and both tails.

Magnitude matters because value scale can change the observed error. We don't claim
our synthetic values to be perfect, but they are chosen to avoid artificially
inflating or suppressing differences because of synthetic data scale.
:::


## Numerical Comparison

Equivalence tests run under three policies (described in [Precision and Dtype
Policies](model/configuration.md#precision-and-dtype-policies)):

- `deepmind` matches the published JAX precision policy: FP32 inputs and parameters,
  BF16 compute and outputs, and FP32 for specific upcasted operations.
- `float32` uses FP32 throughout the model.
- `float64` uses FP64 throughout the model.

The comparison thresholds for the test suite are:

| Comparison level | Representation | Metric | DeepMind | FP32 | FP64 |
| --- | --- | --- | ---: | ---: | ---: |
| Single module | Default | Relative L2 | `1e-2` | `1e-5` | `1e-7` |
|  |  | Relative L∞ | `2e-2` | `2e-5` | `2e-7` |
| Architecture | Default | Relative L2 | `2e-2` | `2e-5` | `2e-7` |
|  |  | Relative L∞ | `4e-2` | `4e-5` | `4e-7` |
| Full model | Default | Relative L2 | `5e-2` | `5e-5` | `5e-7` |
|  |  | Relative L∞ | `1e-1` | `1e-4` | `1e-6` |
| Full model | Splice-junction predictions | Relative L2 | `1.25e-1` | `1.25e-4` | `1.25e-6` |
|  |  | Relative L∞ | `2.5e-1` | `2.5e-4` | `2.5e-6` |
| Full model | Descaled predictions | Relative L2 | `1.25e-1` | `1.25e-4` | `1.25e-6` |
|  |  | Relative L∞ | `2.5e-1` | `2.5e-4` | `2.5e-6` |

:::{dropdown} How are thresholds constructed?
:color: info
:icon: info

Each threshold multiplies precision, composition, metric, and representation
coefficients. The representation coefficient equals 1 by default. Descaled
genome-track and splice-junction predictions use an empirical coefficient of
2.5 because they showed higher errors, particularly RNA-seq among the
genome-track predictions. We believe these larger errors come from
floating-point differences accumulating through their respective operations.
:::

:::{dropdown} How are relative {math}`L_2` and {math}`L_\infty` calculated?
:color: info
:icon: info

For the finite PyTorch values $p$ and corresponding JAX reference values $j$,
the metrics are

$$
\operatorname{relative\ L2} = \frac{\lVert p-j \rVert_2}{\lVert j \rVert_2},
\qquad
\operatorname{relative\ L\infty} = \frac{\max |p-j|}{\max |j|}
$$

A zero-scale reference has relative error zero for an exact match and infinity
otherwise. An infinite relative error fails the equivalence check, ensuring that
a nonzero PyTorch result cannot pass against a zero JAX reference. Values are converted
to FP64 for these calculations. NaN and $\pm\infty$ locations must match
exactly between frameworks and are excluded from the norm calculations.
:::

:::{dropdown} How is the precision policy enforced in the JAX model?
:color: info
:icon: info

- `use_jax_compute_uptype_policy` temporarily overwrites the functionality
  of specific operations in specific modules of the JAX model to obey the
  explicit precision choices for the selected DtypePolicy.
:::

:::{dropdown} How are organism-padded tracks handled?
:color: info
:icon: info

The published heads use shared tensor shapes across organisms, so some channels
are padding rather than biological prediction tracks.

JAX produces `NaN` in padded descaled-prediction channels because no track
means are available, whereas PyTorch uses finite filler values. All scaled and
descaled genome-track prediction comparisons therefore use checkpoint metadata
to include only tracks that are valid for each organism.
:::

:::{dropdown} How reproducible are the tests?
:color: info
:icon: info

Overall, we have deterministic construction of synthetic inputs, parameters,
and targets. However, repeated GPU runs have not produced identical reported
metrics, which may reflect nondeterministic execution in either backend.
We use the following settings to reduce variability:

- Synthetic generation uses either deterministic construction or fixed seeds
  for random construction.
- PyTorch sets `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`,
  `torch.backends.cudnn.deterministic = True`, and
  `torch.backends.cudnn.benchmark = False`. It also sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- JAX runs under `jax.default_matmul_precision("float32")`.
:::

## Equivalence Trends

Our equivalence tests show two trends:

1. Full-model equivalence improves as floating-point precision increases.
2. Differences increase as values propagate through successive layers.

Together, these trends indicate that the remaining differences are consistent with accumulated floating-point error, despite the forgiving thresholds in the default `deepmind` DtypePolicy.


We show full-model output equivalence errors across precision policies:

:::::{tab-set}
::::{tab-item} Relative {math}`L_2`
![Relative L2 differences for full-model embeddings, model-space predictions, descaled genome-track predictions, and per-head losses across DeepMind BF16, FP32, and FP64 policies.](./_static/deepmind-equivalence/full-model-relative_l2.svg)
::::
::::{tab-item} Relative {math}`L_\infty`
![Relative Linf differences for full-model embeddings, model-space predictions, descaled genome-track predictions, and per-head losses across DeepMind BF16, FP32, and FP64 policies.](./_static/deepmind-equivalence/full-model-relative_linf.svg)
::::
:::::


We show encoder representation equivalence errors across stages 0–7:

:::::{tab-set}
::::{tab-item} Relative {math}`L_2`
![Relative L2 differences through the encoder stages, shown on aligned logarithmic axes shifted by powers of ten for DeepMind BF16, FP32, and FP64 policies.](./_static/deepmind-equivalence/encoder-relative_l2.svg)
::::
::::{tab-item} Relative {math}`L_\infty`
![Relative Linf differences through the encoder stages, shown on aligned logarithmic axes shifted by powers of ten for DeepMind BF16, FP32, and FP64 policies.](./_static/deepmind-equivalence/encoder-relative_linf.svg)
::::
:::::


:::{dropdown} Regenerate the figures
The figures use the CSV report produced by a complete 131,072-bp equivalence
run. First download the deterministic human and mouse chromosome-1 inputs:

```bash
python tests/deepmind_equivalence/download_dna.py
```

Run the equivalence suite with both reference sequences and save its report:

```bash
python -m pytest tests/deepmind_equivalence/ \
  --run-equivalence \
  --checkpoint-equivalence-device cuda \
  --equivalence-sequence-length 131072 \
  --equivalence-report tests/deepmind_equivalence/report.csv \
  --equivalence-dna \
    tests/deepmind_equivalence/dna/human_hg38_chr1.fa \
    tests/deepmind_equivalence/dna/mouse_mm39_chr1.fa \
  -v
```

Generate the figures from that report:

```bash
python tests/deepmind_equivalence/plot_report.py \
  tests/deepmind_equivalence/report.csv \
  --output-dir docs/_static/deepmind-equivalence
```
:::

## Terms and Resources

:::{caution}
Checkpoint-backed tests require acceptance of the
[AlphaGenome model terms](https://deepmind.google.com/science/alphagenome/model-terms),
authenticated checkpoint access, substantial storage, and enough memory for
both framework checkpoints. Complete loss comparison requires at least 131,072
bp and a multiple of 131,072 bp.
:::

## Reproduce the Checks

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install Equivalence Dependencies
:link: installation
:link-type: doc
:class-card: sd-card-hover

Install the JAX reference implementation and optional dependencies for
equivalence testing.
:::

:::{grid-item-card} Run Equivalence Tests
:link: development/setup
:link-type: doc
:class-card: sd-card-hover

Run checkpoint-free and checkpoint-loaded comparison tests.
:::

::::
