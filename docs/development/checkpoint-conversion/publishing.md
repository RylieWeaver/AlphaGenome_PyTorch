# Publish Converted Artifacts

{bdg-danger}`External write`

This maintainer helper publishes a complete converted artifact set to a Hugging Face model repository. The local directory must contain all five `alphagenome_<fold>.pt` files and all three metadata files described in [JAX-to-PyTorch Conversion](jax-to-pytorch.md).

## Preview Uploads

```{code-block} bash
:caption: Validate the local release set without uploading

python -m alphagenome_pt.jax2pt.hf_upload \
  --local-dir checkpoints \
  --dry-run
```

:::{note}
`--dry-run` verifies that every expected path exists and prints each
destination. It does not inspect file contents, authenticate, verify repository
access, or write to Hugging Face.
:::

## Upload

Authenticate through saved Hugging Face credentials or `HF_TOKEN`.

:::{dropdown} Public Repository and Partial Upload Risk
:color: warning
:icon: alert

The command creates a **public** model repository if needed and uploads the
eight files one at a time. A failed run can therefore leave a partial release.
Confirm the repository ID and local artifacts before continuing.
:::

```{code-block} bash
:caption: Publish the complete converted release

python -m alphagenome_pt.jax2pt.hf_upload \
  --local-dir checkpoints \
  --repo-id OWNER/REPOSITORY
```

| Argument | Default | Behavior |
| --- | --- | --- |
| `--local-dir` | `.` | Directory containing all eight converted artifacts |
| `--repo-id` | `RylieWeaver/alphagenome-pytorch` | Destination Hugging Face model repository |
| `--repo-dir` | `v{package-version}` | Directory receiving the files inside the repository |
| `--dry-run` | `False` | Verify filenames and print destinations without writing to Hugging Face |
