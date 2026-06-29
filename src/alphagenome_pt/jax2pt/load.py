"""Load official AlphaGenome JAX checkpoints."""

from __future__ import annotations

# External
from pathlib import Path
from typing import Any

# Internal
from alphagenome_pt import DEFAULT_FOLD, FOLD_NAMES



def load_jax_state(
    fold: str = DEFAULT_FOLD,
    *,
    model_path: Path | str | None = None,
    device: str = "cpu",
    verbose: bool = True,
):
    # Lazy import to avoid requiring JAX deps at module load time.
    try:
        if verbose:
            print("Importing JAX...", flush=True)
        import jax
        if verbose:
            print("Importing alphagenome_research loader...", flush=True)
        from alphagenome_research.model.dna_model import create, create_from_huggingface
        if verbose:
            print("JAX loader imports complete.", flush=True)
    except ImportError as e:
        raise ImportError(
            'Missing JAX conversion dependencies. Install with: pip install "alphagenome-pt[jax2pt]". '
            "Also install the public loader repo: "
            "git clone https://github.com/google-deepmind/alphagenome_research.git "
            "&& pip install -e alphagenome_research"
        ) from e

    device = jax.devices(device)[0]
    if verbose:
        print(f"Using JAX device: {device}")

    if fold not in FOLD_NAMES:
        allowed = ", ".join(FOLD_NAMES)
        raise ValueError(f"Unknown AlphaGenome fold {fold!r}. Expected one of: {allowed}.")

    if model_path is not None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"JAX checkpoint path does not exist: {model_path}")

        if verbose:
            print(f"Loading checkpoint from local path: {model_path}")
            print("Calling alphagenome_research create(...)", flush=True)
        model = create(
            checkpoint_path=str(model_path),
            device=device,
        )
        if verbose:
            print("alphagenome_research create(...) returned.", flush=True)
    else:
        if verbose:
            print(f"Loading checkpoint fold: {fold}")
            print("Calling alphagenome_research create_from_huggingface(...)", flush=True)
        model = create_from_huggingface(
            model_version=fold,
            device=device,
        )
        if verbose:
            print("alphagenome_research create_from_huggingface(...) returned.", flush=True)

    return model._params, model._state


def load_public_metadata() -> list[tuple[str, Any]]:
    try:
        from alphagenome_research.model import dna_model
        from alphagenome_research.model.metadata import metadata as metadata_lib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Could not import public AlphaGenome metadata dependencies. "
            "Run this in an environment with alphagenome and alphagenome_research "
            "installed/importable. "
            f"Missing module: {exc.name!r}"
        ) from exc

    organisms = [
        ("human", dna_model.Organism.HOMO_SAPIENS),
        ("mouse", dna_model.Organism.MUS_MUSCULUS),
    ]
    return [(name, metadata_lib.load(organism)) for name, organism in organisms]


def load_jax_checkpoint(
    fold: str = DEFAULT_FOLD,
    *,
    model_path: Path | str | None = None,
    device: str = "cpu",
    verbose: bool = True,
):
    params, state = load_jax_state(
        fold=fold,
        model_path=model_path,
        device=device,
        verbose=verbose,
    )
    metadata = load_public_metadata()
    return params, state, metadata
