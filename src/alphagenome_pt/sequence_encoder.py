# Provenance: Derived from AlphaGenome (Google LLC) Apache-2.0 code; translated to PyTorch and added labels encoding for MLM head. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
import numpy as np
import numpy.typing as np_typing
from typing import Union

# Torch
import torch

# AlphaGenome



class SequenceEncoder:
    """A one-hot encoder for DNA sequences.

    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0, 0, 1, 0]
    T -> [0, 0, 0, 1]

    All other characters are encoded as zeros [0, 0, 0, 0].

    NOTE: This implies that all tokens are single-character strings.
    """

    def __init__(self, dtype: np_typing.DTypeLike = np.float32):
        self._lookup_table = np.zeros((256, 4), dtype=dtype)
        self._lookup_table[ord('A')] = [1, 0, 0, 0]
        self._lookup_table[ord('C')] = [0, 1, 0, 0]
        self._lookup_table[ord('G')] = [0, 0, 1, 0]
        self._lookup_table[ord('T')] = [0, 0, 0, 1]
        self._lookup_table[ord('a')] = self._lookup_table[ord('A')]
        self._lookup_table[ord('c')] = self._lookup_table[ord('C')]
        self._lookup_table[ord('g')] = self._lookup_table[ord('G')]
        self._lookup_table[ord('t')] = self._lookup_table[ord('T')]

        self._label_lookup_table = np.full((256,), fill_value=4, dtype=np.int64)  # Default to 4 for unknown chars
        self._label_lookup_table[ord('A')] = 0
        self._label_lookup_table[ord('C')] = 1
        self._label_lookup_table[ord('G')] = 2
        self._label_lookup_table[ord('T')] = 3
        self._label_lookup_table[ord('a')] = self._label_lookup_table[ord('A')]
        self._label_lookup_table[ord('c')] = self._label_lookup_table[ord('C')]
        self._label_lookup_table[ord('g')] = self._label_lookup_table[ord('G')]
        self._label_lookup_table[ord('t')] = self._label_lookup_table[ord('T')]

    def encode(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """One-hot encodes a DNA sequence string (or list of them).

        Args:
            sequences: A DNA sequence string or a list of DNA sequence strings 
            (e.g., "AGCTNacgt").

        Returns:
            A 2D torch.Tensor of shape (sequence_length, 4) containing the
            one-hot encoded representation of the sequence.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        xs = []
        for i, seq in enumerate(sequences):
            seq = sequences[i]
            if not isinstance(seq, str):
                raise ValueError(f"Each element in the input list must be a string, but got {type(seq)}")
            byte_values = np.frombuffer(seq.encode('latin1'), dtype=np.uint8)
            x = self._lookup_table[byte_values]
            xs.append(torch.from_numpy(x))
        xs = torch.stack(xs, dim=0)  # [B, S, 4]
        return xs
    
    def labels_encode(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """Encodes a DNA sequence string (or list of them) as class indices.

        A -> 0
        C -> 1
        G -> 2
        T -> 3
        All other characters -> 4 (e.g., N for masked token)

        Args:
            sequences: A DNA sequence string or a list of DNA sequence strings 
            (e.g., "AGCTNacgt").

        Returns:
            A 1D torch.Tensor of shape (sequence_length,) containing the
            class index representation of the sequence.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        xs = []
        for i, seq in enumerate(sequences):
            seq = sequences[i]
            if not isinstance(seq, str):
                raise ValueError(f"Each element in the input list must be a string, but got {type(seq)}")
            byte_values = np.frombuffer(seq.encode('latin1'), dtype=np.uint8)
            x = self._label_lookup_table[byte_values]
            xs.append(torch.from_numpy(x))
        xs = torch.stack(xs, dim=0)  # [B, S]
        return xs
