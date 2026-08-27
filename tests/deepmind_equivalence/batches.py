from __future__ import annotations

# External
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
import warnings

import torch

# Internal
from alphagenome_pt import (
    DNAOneHotEncoder,
    DataBatch,
    Metadata,
    synthetic_batch,
)
from alphagenome_pt.synthetic import GENOME_TRACK_HEADS
from .utils import _numpy


##### CONSTANTS #####
BATCH_SIZE = 2
MIN_NUM_SPLICE_SITES = 16
BASE_PAIRS_PER_SPLICE_SITE = 2048


### HELPERS ###
def equivalence_num_splice_sites(sequence_length: int) -> int:
    """Choose a bounded splice-site count for a full-model comparison."""
    return max(
        MIN_NUM_SPLICE_SITES,
        sequence_length // BASE_PAIRS_PER_SPLICE_SITE,
    )


@contextmanager
def _seeded(seed: int):
    """Make generation repeatable without changing RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def load_dna_chunks(
    paths: Sequence[str | Path],
    sequence_length: int,
) -> torch.Tensor:
    """Load one shared or two organism-specific DNA sequence files."""
    if len(paths) not in (1, 2):
        raise ValueError("equivalence DNA input requires one or two files")

    sequences = []
    for path in paths:
        lines = Path(path).read_text(encoding="ascii").splitlines()
        sequence = "".join(
            line.strip()
            for line in lines
            if line.strip() and not line.startswith(">")
        ).upper()
        invalid_bases = set(sequence) - set("ACGTN")
        if invalid_bases:
            raise ValueError(
                f"{path} contains unsupported DNA bases: "
                f"{sorted(invalid_bases)}"
            )
        if len(sequence) < sequence_length:
            raise ValueError(
                f"{path} contains {len(sequence)} bases, fewer than the "
                f"requested {sequence_length}"
            )
        sequences.append(sequence[:sequence_length])

    if len(sequences) == 1:
        sequences *= BATCH_SIZE
    return DNAOneHotEncoder().encode(sequences)


### FRAMEWORK BATCHES ###
def make_pytorch_batch(
    metadata: Metadata | dict,
    sequence_length: int,
    seed: int,
) -> DataBatch:
    if isinstance(metadata, dict):
        metadata = Metadata(metadata)
    with _seeded(seed):
        batch = synthetic_batch(
            metadata,
            batch_size=BATCH_SIZE,
            seq_len=sequence_length,
            num_splice_sites=equivalence_num_splice_sites(sequence_length),
        )

    organism_index = batch.get_organism_index()
    if organism_index.tolist() != [0, 1]:
        warnings.warn(
            "Equivalence testing should use all organism indices in the JAX "
            "checkpoint ([0, 1] for human and mouse), but synthetic_batch "
            f"produced {organism_index.tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Let JAX derive junction positions; the full test gives those same
    # positions to PyTorch so the downstream junction head sees equal inputs.
    batch.splice_site_positions = None

    # The official JAX DataBatch has no batch-specific junction-mask field.
    batch.splice_junctions_mask = None

    # JAX expects each batch mask to identify the organism's valid,
    # non-padded tracks before loss computation.
    for head_name in GENOME_TRACK_HEADS:
        if getattr(batch, head_name, None) is not None:
            metadata_mask = metadata.get_multiorg_track_mask(
                head_name,
                organism_index,
            )
            setattr(batch, f"{head_name}_mask", metadata_mask[:, None, :])
    return batch


def make_jax_target_batch(batch: DataBatch):
    import jax.numpy as jnp
    from alphagenome_research.model import schemas

    def jax_array(name):
        value = _numpy(getattr(batch, name))
        return None if value is None else jnp.asarray(value)

    return schemas.DataBatch(
        dna_sequence=jax_array("dna_sequence_one_hot"),
        organism_index=jax_array("organism_index"),
        atac=jax_array("atac"),
        atac_mask=jax_array("atac_mask"),
        dnase=jax_array("dnase"),
        dnase_mask=jax_array("dnase_mask"),
        procap=jax_array("procap"),
        procap_mask=jax_array("procap_mask"),
        cage=jax_array("cage"),
        cage_mask=jax_array("cage_mask"),
        rna_seq=jax_array("rna_seq"),
        rna_seq_mask=jax_array("rna_seq_mask"),
        rna_seq_strand=jax_array("rna_seq_strand"),
        chip_tf=jax_array("chip_tf"),
        chip_tf_mask=jax_array("chip_tf_mask"),
        chip_histone=jax_array("chip_histone"),
        chip_histone_mask=jax_array("chip_histone_mask"),
        contact_maps=jax_array("contact_maps"),
        splice_sites=jax_array("splice_sites"),
        splice_site_usage=jax_array("splice_site_usage"),
        splice_junctions=jax_array("splice_junctions"),
    )
