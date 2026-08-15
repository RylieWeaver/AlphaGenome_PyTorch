# Installation

AlphaGenome PyTorch has two core requirements:

Python
: Version 3.10 or newer

PyTorch
: Version 2.0 or newer

## Install

```{code-block} bash
:caption: Install the package

python -m pip install alphagenome-pt
```

```{code-block} bash
:caption: Check the installed version

alphagenome-pt --version
```

The base installation includes the dependencies for model construction,
training, inference, and loading converted checkpoints.

Continue to the [Quickstart](quickstart.md) to construct and run a model.

## Optional Extras

:::{important}
The optional extras below require a repository checkout. Run their commands
from the repository root and see [Development Setup](development/setup.md) for
cloning instructions.
:::

### Documentation

```{code-block} bash
:caption: Install and build the documentation

python -m pip install -e ".[docs]"
python -m sphinx -E -W -b html docs docs/_build/html
```

The generated docs are in `docs/_build/html`.

### Development

```{code-block} bash
:caption: Install an editable development checkout

python -m pip install -e ".[dev]"
```

See [Development Setup](development/setup.md) for test commands and contributor guidance.

### Checkpoint Conversion

```{code-block} bash
:caption: Install checkpoint-conversion dependencies

python -m pip install -e ".[jax2pt]"
```

:::{important}
The public `alphagenome_research` loader used during conversion requires Python
3.11 or newer.
:::

See [Checkpoint Conversion](development/checkpoint-conversion/index.md) for the
complete workflow.
