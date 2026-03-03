import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from alphagenome_pt import AlphaGenome, AlphaGenomeConfig
from alphagenome_pt.metadata import Metadata
from alphagenome_pt.sequence_encoder import SequenceEncoder


ORGANISMS = ("human", "mouse")


def make_means(num_tracks: Sequence[int], low: float = 3.0, high: float = 10.0) -> torch.Tensor:
    all_means = []
    max_tracks = max(num_tracks)
    for num in num_tracks:
        means = torch.empty(num).uniform_(low, high)
        means = F.pad(means, (0, max_tracks - num), value=1.0)
        all_means.append(means)
    return torch.stack(all_means, dim=0)


def build_metadata(heads: dict, organisms: Sequence[str] = ORGANISMS) -> Metadata:
    metadata = Metadata(
        metadata={
            "organisms": list(organisms),
            "heads": heads,
        }
    )
    metadata.make_all_masks()
    return metadata


def build_small_model(metadata: Metadata, **cfg_overrides) -> AlphaGenome:
    cfg_kwargs = {
        "input_seq_len": 2048,
        "num_channels": 64,
        "transformer_layers": 1,
        "metadata": metadata,
    }
    cfg_kwargs.update(cfg_overrides)
    return AlphaGenome(AlphaGenomeConfig(**cfg_kwargs))


def make_organism_index(batch_size: int, num_organisms: int = 2) -> torch.Tensor:
    return torch.tensor([i % num_organisms for i in range(batch_size)], dtype=torch.long)


def random_dna_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    encoder = SequenceEncoder()
    seqs = ["".join(random.choices("ACGT", k=seq_len)) for _ in range(batch_size)]
    return encoder.encode(seqs).to(torch.float32)


def make_poisson_tracks(means: torch.Tensor, organism_index: torch.Tensor, length: int) -> torch.Tensor:
    rates = means[organism_index].unsqueeze(1).repeat(1, length, 1)
    return torch.poisson(rates).to(torch.float32)


def make_splice_sites(batch_size: int, seq_len: int, num_classes: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch_size, seq_len), dtype=torch.long)
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def assert_finite_scalars(scalars: dict[str, torch.Tensor]) -> None:
    for key, value in scalars.items():
        assert torch.isfinite(value), key

