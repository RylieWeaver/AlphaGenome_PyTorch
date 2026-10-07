"""General package utilities."""

from __future__ import annotations

# External
from importlib import metadata as importlib_metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PACKAGE_DISTRIBUTION_NAME = "alphagenome-pt"


def _normalize_distribution_name(name: str) -> str:
    return name.replace("_", "-").lower()


def project_root() -> Path | None:
    # NOTE: This is hardcoded to the package layout:
    #   repo_root/src/alphagenome_pt/utils.py
    # We don't dynamically search because installed packages often live under
    # another uv/project directory that can have an unrelated pyproject.toml.
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None

    with pyproject.open("rb") as f:
        project = tomllib.load(f).get("project", {})
    if (
        _normalize_distribution_name(project.get("name", ""))
        != PACKAGE_DISTRIBUTION_NAME
    ):
        return None
    return root


def project_metadata() -> dict | None:
    root = project_root()
    if root is None:
        return None

    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def package_name() -> str:
    project = project_metadata()
    if project is not None:
        return project["name"]
    return PACKAGE_DISTRIBUTION_NAME


def package_version() -> str:
    project = project_metadata()
    if project is not None:
        return project["version"]
    try:
        return importlib_metadata.version(PACKAGE_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        raise RuntimeError(
            "Could not find installed package metadata or pyproject.toml."
        )


def _register_pytree_dataclass(cls):
    """
    NOTE: Put simply, all this function does is mark a class for pytree to know
    to traverse through it. As explained below, this allows DDP to find unused
    parameters correctly inside our special return classes (e.g. Embeddings).

    With find_unused_parameters=True, the relevant steps are:

    (1) DDP finds every returned tensor.
        - This traversal understands dicts, lists, tuples, and dataclasses
          such as Embeddings. All good so far.

    (2) DDP follows autograd graphs from returned tensors back to parameters.
        - Parameters unreachable from every returned tensor are marked unused.
        - Parameters feeding a returned output remain potentially used, even
          if we will later ignore that output when computing the loss.

    (3) DDP traverses the returned container again to wrap tensors in _DDPSink.
        - This traversal uses pytree, unlike the traversal in step (1).
        - For dicts, lists, and tuples, pytree enters the container, finds its
          tensors, allowing DDP to wrap them in the shared _DDPSink autograd node.
        - For an unregistered dataclass, pytree stops at the object. It does
          not inspect its fields, so the tensors inside are NOT wrapped.
        - This is the mismatch: step (1) finds tensors inside Embeddings, but
          step (3) misses those same tensors unless the dataclass is registered.

    (4) The caller runs backward on the loss.
        - DDP must now account for the "potentially used" parameters from step (2).
          Parameters used by the loss receive gradients. For parameters used only
          by ignored outputs, _DDPSink passes None gradients through those branches,
          notifying DDP that no gradient will arrive on this rank.
        - Without this registration, tensors inside our dataclasses (e.g. Embeddings)
          are NOT wrapped by _DDPSink in step (3). Parameters used only by ignored
          outputs therefore provide no backward notification. Those parameters stay
          considered as "maybe used", so reduction can remain unfinished and the next
          forward raises an error.

    Example failure mode:
    - model.loss(batch, return_embeddings=True) with a DDP-wrapped, MLM-only model,
      followed by output.total.backward(). The returned pair embeddings are ignored
      by the loss, so their output_pair parameters need the None notifications above.

    Registering LossOutput, Embeddings, MetricTree, and LossLeaf lets step (3)
    reach their tensors and reconstruct the dataclasses around DDP's wrapped outputs.
    Making the loss containers dataclasses also lets FSDP's own traversal attach
    pre-backward hooks to leaf numerators, so output.tree.total_loss().backward()
    reaches those hooks even though it does not use output.total.
    """
    from dataclasses import fields
    from torch.utils import _pytree

    names = tuple(field.name for field in fields(cls))
    register = getattr(_pytree, "register_pytree_node", None)
    if register is None:  # PyTorch 2.0/2.1 used the private spelling.
        register = _pytree._register_pytree_node

    register(
        cls,
        lambda value: ([getattr(value, name) for name in names], None),
        lambda values, context: cls(**dict(zip(names, values, strict=True))),
    )
    return cls
