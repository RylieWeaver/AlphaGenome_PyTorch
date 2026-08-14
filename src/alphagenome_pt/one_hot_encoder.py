# Provenance: Derived from AlphaGenome (Google LLC) Apache-2.0 code and translated to PyTorch. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This is NOT the usual one-hot -> embedding table process used in language models.
# Instead, ACGT are converted to one-hot and 'N' is converted to all zeros. This is fed
# directly to the convolutions rather than an embedding table. I believe this is used so
# that we more explicitly ignore ambiguous base pairs, which are represented by 'N'.

# NOTE: Input strings must contain only Latin-1 characters.


from collections.abc import Sequence

import numpy as np
import numpy.typing as np_typing
import torch


class DNAOneHotEncoder:
    """A one-hot encoder for DNA sequences.

    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0, 0, 1, 0]
    T -> [0, 0, 0, 1]

    Input sequences must contain only Latin-1 characters. All Latin-1
    characters other than A, C, G, and T are encoded as zeros [0, 0, 0, 0].
    """

    def __init__(self, dtype: np_typing.DTypeLike = np.float32):
        self._lookup_table = np.zeros((256, 4), dtype=dtype)
        for index, base in enumerate("ACGT"):
            self._lookup_table[ord(base), index] = 1
            self._lookup_table[ord(base.lower()), index] = 1

    def encode(
        self,
        sequences: str | Sequence[str],
    ) -> torch.Tensor:
        """One-hot encode one DNA sequence or a batch of sequences.

        Args:
            sequences: A DNA sequence or equal-length sequence batch.

        Returns:
            A tensor with shape ``[S, 4]`` or ``[B, S, 4]``.
        """
        if isinstance(sequences, str):
            byte_values = np.frombuffer(sequences.encode("latin1"), dtype=np.uint8)
            return torch.from_numpy(self._lookup_table[byte_values])
        return torch.stack([self.encode(sequence) for sequence in sequences])

    def get_dna_one_hot(
        self,
        sequences: Sequence[str] | None,
        one_hot: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return supplied one-hot DNA or encode raw sequences."""
        if one_hot is None:
            if sequences is None:
                raise ValueError("DNA input is required.")
            one_hot = self.encode(sequences).to(dtype=dtype)
        if sequences is not None and one_hot.shape[:2] != (
            len(sequences),
            len(sequences[0]),
        ):
            raise ValueError("dna_sequence and dna_sequence_one_hot disagree.")
        return one_hot
