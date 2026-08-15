# Command-Line Interface

The `alphagenome-pt` CLI provides package information and artifact downloads.

## Show Help

```bash
alphagenome-pt --help
```

## Check the Installed Version

```bash
alphagenome-pt --version
```

## Download Checkpoints and Metadata

:::{dropdown} Full Download Size
:color: warning
:icon: alert

Omitting `--fold` downloads all five checkpoints, which total approximately
9 GB. Pass `--fold` when you need only one checkpoint.
:::

```{code-block} bash
:caption: Download metadata and every converted checkpoint

alphagenome-pt download --local-dir checkpoints
```

The command copies these files into `checkpoints/`:

| File | Contents |
| --- | --- |
| `alphagenome_metadata.json` | Metadata used for model construction |
| `alphagenome_metadata_raw.json` | Public metadata before conversion |
| `alphagenome_metadata_summary.json` | Human-readable metadata inventory |
| `alphagenome_all_folds.pt` | Converted all-folds checkpoint state |
| `alphagenome_fold_0.pt`<br>`alphagenome_fold_1.pt`<br>`alphagenome_fold_2.pt`<br>`alphagenome_fold_3.pt` | Converted fold-specific checkpoint states |

```{code-block} bash
:caption: Download metadata and one fold-specific checkpoint

alphagenome-pt download --local-dir checkpoints --fold fold_0
```

`--fold` selects one checkpoint. Use `all_folds` for the all-folds checkpoint or `fold_0` through `fold_3` for fold-specific checkpoints.

### Arguments

| Argument | Default | Behavior |
| --- | --- | --- |
| `--local-dir PATH` | `.` | Destination for copied files. Created if needed |
| `--fold FOLD` | Omitted | Download all five checkpoints when omitted. Supply `all_folds` or one of `fold_0` through `fold_3` to download one |
| `--repo-id ID` | `RylieWeaver/alphagenome-pytorch` | Hugging Face model repository |
| `--repo-dir DIR` | `v{package-version}` | Directory containing the files inside the repository |
| `--force-download` | `False` | `False`: reuse available Hugging Face cache entries. `True`: download fresh files before copying |

:::{dropdown} Existing Files Are Replaced
:color: warning
:icon: alert

Requested files already present under `--local-dir` are replaced by the
resolved Hugging Face copies.
:::

:::{dropdown} How Download Paths Are Resolved
:color: info
:icon: info

For this command, `--local-dir` is a destination rather than a local-first
lookup source. Each requested file resolves from
`<repo-id>/<repo-dir>/<filename>` through Hugging Face and is copied to
`<local-dir>/<filename>`. An existing destination file does not bypass this
resolution.
:::

:::{dropdown} Authentication
:color: info
:icon: info

Public repositories support anonymous access. Private or gated repositories use
saved Hugging Face credentials or `HF_TOKEN`. The CLI does not accept tokens as
arguments because shell history and process listings can expose them.
:::
