# tests/set1 — correctness tests

## Purpose

The tests already in `tests/` are baseline tests: they help verify that a forward pass runs, that a loss is finite, and that output shapes are correct. They do not verify that any computed value is right. A sign error in a loss function, a scaling transform that is no longer invertible, or a parameter that never receives a gradient would all pass the existing suite.

This package adds 81 tests that check correctness rather than liveness. It runs in roughly 10–25 seconds on CPU and requires no GPU, no JAX, no network access, and no checkpoint download, so it is suitable for running on every commit.

```bash
pytest tests/set1 -q
```

## Sources

Most of these tests are adapted from the test suite of [genomicsxai/alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) (Apache-2.0), maintained by Anshul Kundaje's lab and the successor to `lucidrains/alphagenome`. Both that project and this one are Apache-2.0, so the adaptation is permitted; each file carries a header naming the source file it came from and listing every deviation.

That suite contains 54 test files. Seven were adapted here. The remaining 47 exercise features this project does not implement — LoRA and other adapters, variant scoring, in-silico mutagenesis and attribution, the gRPC/REST serving stack, the BigWig/FASTA/GTF data layer, named outputs, a dtype policy abstraction, and sequence parallelism — and so cannot be adapted without first building the corresponding feature.

| File in this package | Adapted from |
| --- | --- |
| `test_losses.py` | `tests/unit/test_losses.py` |
| `test_scaling.py` | `tests/unit/test_heads.py` (`TestScalingFunctions`, `TestPredictionsScaling`) |
| `test_attention_rope.py` | `tests/unit/test_attention.py` (`TestApplyRope`, `TestApplyRopeGradients`) |
| `test_head_selection.py` | `tests/unit/test_model_heads.py` |
| `test_backward.py` | `tests/integration/test_backward.py` |
| `test_checkpoint_roundtrip.py` | `tests/integration/test_checkpoint_roundtrip.py` |
| `test_determinism.py` | `tests/integration/test_determinism.py` |
| `conftest.py` | no upstream source (a shim; the real configuration is in `_config.py`) |
| `_config.py` | no upstream source |
| `_helpers.py` | no upstream source |

## Golden values

`test_losses.py` is the only file containing hardcoded numbers. Every other file verifies a property — an identity, an invariance, a shape, or a raised exception — so there was nothing to carry over.

The constants below did not originate in `genomicsxai/alphagenome-pytorch`. They were produced by running the reference JAX implementation ([google-deepmind/alphagenome](https://github.com/google-deepmind/alphagenome)) on fixed inputs; that suite recorded them. Each was re-verified against `alphagenome_pt.losses` before being pinned here.

| Function | Value | Tolerance |
| --- | --- | --- |
| `poisson_loss` | `0.0079382462` | exact, `atol=1e-7` |
| `mse` | `0.0199999977` | exact, `atol=1e-7` |
| `cross_entropy_loss_from_logits` | `0.5554059148` | exact, `atol=1e-7` |
| `binary_crossentropy_from_logits` | `0.3047555685` | exact, `atol=1e-7` |
| `cross_entropy_loss` | `1.4022779465` | `atol=1e-6`, not `1e-7` |
| `multinomial_loss` `loss_total` | `17.7364959717` | exact |
| `multinomial_loss` `loss_positional` | `8.7234439850` | exact |
| `multinomial_loss` `loss` | `26.4599399567` | with `min_zero=False` only |

This gives parity with the JAX reference without requiring JAX to be installed: the value was computed once against the reference and frozen. If one of these tests fails, the numerics changed.

The multinomial case requires a note. This implementation's `multinomial_loss` carries a `min_zero` switch that the reference does not have:

- `min_zero=False`: `26.4599399567`, reproducing the JAX reference exactly
- `min_zero=True`: `18.0260715485`, this implementation's default

Both branches are pinned so that neither can drift.

To remove the intermediate dependency entirely, install `alphagenome_research` in a disposable environment, run the reference losses on the same inputs, and regenerate the constants directly. The values are identical; the provenance is shorter.

## Behaviour discovered while writing these tests

The following were surfaced by the tests on their first run. Each is currently undocumented, and each is now pinned so that a change in behaviour is noticed rather than silently absorbed.

**1. Some buffers are absent from `state_dict()`.**

`_track_means` and `_track_mask` are registered non-persistent. The model therefore cannot be restored from a checkpoint alone; it also requires the exact metadata it was constructed with. This appears intentional — metadata is the source of truth, and `checkpoint.py` retrieves state and metadata separately — but it is an undocumented coupling. Pinned in `test_checkpoint_roundtrip.py::test_track_means_are_not_in_the_checkpoint`. If these buffers are ever made persistent, that test should be removed.

**2. `StandardizedConv1d.weight` is zero-initialised.**

Weight standardization computes `scale * (w - mean) / std`. With `w == 0` the resulting kernel is zero regardless of `scale`, so `d(loss)/d(scale) == 0` on the first backward pass: 87 of 363 parameters receive no gradient at step 0. The weights themselves do receive gradients, so this resolves after a single optimiser step (87 falls to 6). This is an initialisation artefact rather than a defect, but a direct port of the upstream gradient sweep fails without accounting for it, so the sweep here performs one warm-up step first. Pinned in `test_backward.py::TestZeroInitialisation`.

**3. The six parameters that remain without gradients belong to the splice junction head.**

Specifically the positive and negative donor and acceptor logit embeddings, and `multiorg_linear.weight` and `multiorg_linear.bias`. This is expected. The junction head only predicts at positions the classification head identifies as splice sites, and an untrained classifier identifies none, so every position is masked out. The upstream suite excludes this head from its own gradient sweep for the same reason, which its docstring states explicitly. Pinned in `test_backward.py::TestJunctionHeadGradients`.

## File summary

### `test_losses.py` — 30 tests, no model

Golden values pin the numerics against the JAX reference. The property tests cover behaviour that can be verified without a reference: masking a position must be indistinguishable from removing it; a perfect prediction must cost zero; a fully masked input must yield `0.0` rather than a 0/0 NaN.

### `test_scaling.py` — 11 tests, no model

`targets_scaling` and `predictions_scaling` must be exact inverses: in both directions, with squashing enabled and disabled, and with uniform and non-uniform track means. This is the highest-leverage test in the package. If the two are not inverses, every training target is silently wrong — the loss remains finite, training still converges, and it converges to the wrong solution. No shape check, NaN check, or gradient check detects this.

### `test_attention_rope.py` — 11 tests, no model

Rotary embeddings must be translation-invariant: shifting all positions by a constant changes the raw outputs, but must leave the inner products between equally-offset pairs unchanged. This is the defining property of RoPE. An implementation with sin and cos transposed passes every shape check and fails this immediately.

### `test_backward.py` — 13 tests, small model

Every parameter must receive a gradient. A parameter without one never trains: it remains at its initialisation for the entire run and nothing reports it. Also includes per-component gradient health with a `(1e-12, 1e8)` norm band, so a layer whose gradient is `1e-30` is caught even though it technically has one.

### `test_checkpoint_roundtrip.py` — 7 tests, small model

`checkpoint.py` is the largest module in the project at 1244 lines and had no fast tests. Save, load into a differently seeded model, and require bitwise equality. The differing seed is essential: had the second model started identical, a key that never loaded would still appear correct.

### `test_determinism.py` — 4 tests, small model

Two pairs. The same input twice must give identical output; the same seed must give the same model. Then the pair that is usually omitted: a different seed must give a different output, and a different input must give a different output. Without those, a model returning zeros passes the first pair perfectly.

### `test_head_selection.py` — 5 tests, small model

Requesting a head must construct it, and requesting fewer must construct fewer, or the memory saving is illusory.