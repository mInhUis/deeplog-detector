"""Sliding-window dataset for DeepLog sequence prediction.

**Why a sliding window?**
DeepLog frames anomaly detection as a *next-token prediction* task: given a
window of ``w`` consecutive log keys, predict the ``(w+1)``-th key.  During
training on normal-only data, the LSTM learns the regular temporal patterns
of log key sequences.  At inference time, if the true next key falls outside
the model's top-k predictions, the event is flagged as anomalous.

A fixed-size sliding window (with stride 1) is the standard approach from
the original DeepLog paper (Du et al., 2017) because:
    1. It captures local temporal context without requiring variable-length
       inputs.
    2. It naturally augments the training set — a session of length L yields
       (L - w) training samples.
    3. It keeps memory bounded regardless of session length.

Data Flow:
    Drain log_keys (list[int]) per session
        → LogKeyDataset(sequences, window_size)
        → (X, y) pairs ready for DataLoader batching
        → DeepLog LSTM

Input Shape:
    sequences : list[list[int]]  — S sessions, each a variable-length
                                    integer sequence from Drain.

Output Shape (per __getitem__):
    X : torch.Tensor of shape (window_size,)   — int64, the input window.
    y : torch.Tensor of shape ()               — int64 scalar, the target
                                                  (next log key to predict).

Data-Leakage Prevention (Advisor Skill 3):
    Each sample's target ``y`` is the key at position ``[i + window_size]``,
    which is strictly *after* all keys in the input window ``X``.  The window
    slides with stride 1 and never wraps across session boundaries — sessions
    are processed independently, so future sessions cannot leak into past
    windows.  Train and test splits are performed *upstream* in preprocess.py
    (by user identity), so there is zero risk of test-set events appearing
    in training windows.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.utils.data import Dataset


class LogKeyDataset(Dataset):
    """Fixed-width sliding window dataset over log-key sessions.

    Each session (a chronological integer sequence from one user/IP combo)
    is sliced into overlapping ``(X, y)`` pairs where:

        X = keys[i : i + window_size]      — shape ``(window_size,)``
        y = keys[i + window_size]           — scalar target

    Sessions shorter than ``window_size + 1`` are silently skipped (they
    cannot produce even one valid sample).

    Args:
        sequences:   List of per-session log-key sequences.  Each inner
                     list is an ordered ``list[int]`` of Drain cluster IDs.
        window_size: Number of preceding keys used as input context.
                     Must be >= 1.

    Raises:
        ValueError: If *window_size* < 1.
    """

    __slots__ = ("_samples", "_window_size")

    def __init__(
        self,
        sequences: list[list[int]],
        window_size: int = 10,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self._window_size: Final[int] = window_size

        # ------------------------------------------------------------------
        # Build the flat sample list.
        # Each entry is a (window, target) tuple stored as plain Python lists
        # to minimise per-sample memory; tensors are created on-the-fly in
        # __getitem__ so the DataLoader can collate efficiently.
        #
        # DATA-LEAKAGE CHECK:
        #   • The inner loop index ``i`` ranges from 0 to len(seq) - w - 1.
        #   • target = seq[i + w]  →  always one step *ahead* of the window.
        #   • Sessions are independent — no cross-session slicing.
        # ------------------------------------------------------------------
        self._samples: list[tuple[list[int], int]] = []

        for seq in sequences:
            seq_len: int = len(seq)
            if seq_len < window_size + 1:
                continue  # too short to form even one (X, y) pair
            for i in range(seq_len - window_size):
                window: list[int] = seq[i : i + window_size]
                target: int = seq[i + window_size]
                self._samples.append((window, target))

    # ------------------------------------------------------------------ #
    #  Dataset protocol                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the total number of (window, target) samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one ``(X, y)`` pair as tensors.

        Returns:
            X: ``torch.int64`` tensor of shape ``(window_size,)``.
            y: ``torch.int64`` scalar tensor of shape ``()``.
        """
        window, target = self._samples[idx]
        # X shape: (window_size,)  — 1-D integer tensor
        X: torch.Tensor = torch.tensor(window, dtype=torch.long)
        # y shape: ()              — 0-D integer scalar
        y: torch.Tensor = torch.tensor(target, dtype=torch.long)
        return X, y

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def window_size(self) -> int:
        """The window width used to construct this dataset."""
        return self._window_size
