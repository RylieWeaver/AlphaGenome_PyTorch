# Building Docs

Run these commands from the repository root after completing [Development Setup](setup.md).

## Build the Documentation

Install the documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

```{code-block} bash
:caption: Build the documentation strictly

python -m sphinx -M clean docs docs/_build
python -m sphinx -E -W -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the generated docs.

## Generated Figures

Documentation figures come from Python generators.

```{code-block} bash
:caption: Regenerate documentation figures

python docs/_scripts/generate_checkpoint_loading_diagrams.py
python docs/_scripts/generate_model_architecture_diagrams.py
```

```{code-block} bash
:caption: Check for stale generated figures without rewriting them

python docs/_scripts/generate_checkpoint_loading_diagrams.py --check
python docs/_scripts/generate_model_architecture_diagrams.py --check
```

Edit the generator scripts rather than their SVG outputs under
`docs/_static/checkpoint-loading/` and `docs/_static/model-architecture/`
directly.
