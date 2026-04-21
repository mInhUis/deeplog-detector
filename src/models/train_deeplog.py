"""Train the DeepLog LSTM on normal-only log-key sequences.

**Why train on normal data only (semi-supervised)?**
In real cloud environments, labelled attack data is scarce and constantly
evolving.  By training exclusively on *normal* sequences (Root user
baseline from preprocess.py), the LSTM learns the legitimate transition
grammar.  Any deviation at inference — regardless of attack type — is
flagged automatically.  This makes DeepLog robust to *novel* attacks that
were never seen during training.

Pipeline position (CLAUDE.md):
    Drain log_keys (train_sessions.csv)
        → LogKeyDataset sliding windows
        → DeepLog LSTM training (this script)
        → saved model checkpoint  (models/deeplog.pt)

Usage:
    python src/models/train_deeplog.py [--epochs N] [--mode mock]

Environment Variables:
    TRAIN_KEYS_PATH – Path to the training log-key sequence file
                      (default: data/processed/train_log_keys.json)
    MODEL_DIR       – Directory for saved checkpoints
                      (default: models/)
    DEEPLOG_MODE    – "mock" to use a tiny synthetic dataset (safe for
                      interactive sessions per CLAUDE.md), or "full" to
                      load real data.  Default: "mock".
    CUDA_MEMORY_MAX – Max VRAM fraction PyTorch is allowed to allocate
                      (0.0–1.0).  Default: 0.5 (cap at 50% to prevent
                      display driver crashes on consumer laptops).

Heavy Compute Warning (CLAUDE.md):
    This script defaults to ``--mode mock`` with 5–10 synthetic sessions
    to verify logic without straining hardware.  Switch to ``--mode full``
    only when running the complete pipeline on suitable hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

# Ensure project root is on sys.path so `src.*` imports resolve when the
# script is invoked directly (e.g. `python src/models/train_deeplog.py`).
_PROJECT_ROOT_EARLY = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT_EARLY not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_EARLY)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.dataset import LogKeyDataset
from src.detector.deeplog import DeepLogModel

# ---------------------------------------------------------------------------
# Configurable paths (Docker / env-var friendly per CLAUDE.md)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

TRAIN_KEYS_PATH: Final[Path] = Path(
    os.environ.get(
        "TRAIN_KEYS_PATH",
        str(_PROJECT_ROOT / "data" / "processed" / "train_log_keys.json"),
    )
)
MODEL_DIR: Final[Path] = Path(
    os.environ.get("MODEL_DIR", str(_PROJECT_ROOT / "models"))
)

# ---------------------------------------------------------------------------
# VRAM safety cap (per CLAUDE.md heavy-compute warning)
# ---------------------------------------------------------------------------
_CUDA_MEMORY_MAX: Final[float] = float(
    os.environ.get("CUDA_MEMORY_MAX", "0.5")
)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE: Final[int] = 10
DEFAULT_EMBEDDING_DIM: Final[int] = 64
DEFAULT_HIDDEN_SIZE: Final[int] = 64
DEFAULT_NUM_LAYERS: Final[int] = 2
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_LR: Final[float] = 1e-3
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_EPOCHS: Final[int] = 50


# ======================================================================== #
#  Mock data generator                                                      #
# ======================================================================== #

def generate_mock_sequences(
    num_keys: int = 5,
    num_sessions: int = 8,
    session_length: int = 30,
) -> list[list[int]]:
    """Create tiny synthetic sessions for safe logic verification.

    Generates deterministic repeating patterns so the LSTM can learn
    them quickly (verifying the training loop converges).

    Args:
        num_keys:       Vocabulary size (number of distinct log keys).
        num_sessions:   Number of sessions to generate.
        session_length: Events per session.

    Returns:
        List of sessions, each a ``list[int]`` of log-key IDs.
    """
    sessions: list[list[int]] = []
    for s in range(num_sessions):
        # Deterministic repeating pattern: 0,1,2,...,num_keys-1,0,1,...
        session: list[int] = [
            (s + i) % num_keys for i in range(session_length)
        ]
        sessions.append(session)
    return sessions


# ======================================================================== #
#  Training loop                                                            #
# ======================================================================== #

def train(
    model: DeepLogModel,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    start_epoch: int = 0,
    optimiser_state: dict | None = None,
    save_every: int = 10,
    on_checkpoint: Callable[[nn.Module, torch.optim.Adam, int, float], None] | None = None,
) -> tuple[list[float], torch.optim.Adam]:
    """Train the DeepLog model with cross-entropy loss.

    **Why cross-entropy?**
    The task is multi-class classification — predicting which of K
    possible log keys comes next.  Cross-entropy is the standard loss
    for this setting: it directly optimises the model's predicted
    probability of the correct next key, which is exactly what the
    top-k anomaly check relies on at inference time.

    Args:
        model:           DeepLogModel instance (moved to *device*).
        dataloader:      Training DataLoader yielding ``(X, y)`` batches.
        epochs:          Total number of epochs to reach (absolute, not
                         additional).
        lr:              Learning rate for Adam optimiser.
        device:          Torch device (cpu / cuda).
        start_epoch:     Epoch index to resume from (0-based).  Epochs
                         ``[start_epoch, epochs)`` will be executed.
        optimiser_state: If resuming, the saved
                         ``optimiser.state_dict()``.  Passed to
                         ``optimiser.load_state_dict()`` so that
                         momentum buffers and learning-rate schedule
                         are restored exactly.
        save_every:      Save a mid-training checkpoint every N epochs.
                         Default ``10``.  Requires *on_checkpoint*.
        on_checkpoint:   Callback invoked every *save_every* epochs
                         (but **not** on the final epoch — the caller
                         handles that).  Signature:
                         ``(model, optimiser, epoch, avg_loss) -> None``.

    Returns:
        A ``(epoch_losses, optimiser)`` tuple:
        - ``epoch_losses``: per-epoch average losses for the epochs
          actually run (length = ``epochs - start_epoch``).
        - ``optimiser``: the Adam optimiser instance (its
          ``state_dict()`` is needed for checkpoint saving).

    Tensor shapes per batch:
        X      : (batch_size, window_size) int64
        y      : (batch_size,)             int64
        logits : (batch_size, num_keys)    float32
        loss   : ()                        float32 scalar
    """
    model.to(device)
    model.train()

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimiser: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

    if optimiser_state is not None:
        optimiser.load_state_dict(optimiser_state)

    epoch_losses: list[float] = []

    for epoch in range(start_epoch, epochs):
        running_loss: float = 0.0
        num_batches: int = 0

        for X_batch, y_batch in dataloader:
            # X_batch: (batch_size, window_size) int64
            # y_batch: (batch_size,)             int64
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()

            # logits: (batch_size, num_keys)
            logits, _ = model(X_batch)

            # loss: () scalar
            loss: torch.Tensor = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss: float = running_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            print(f"  Epoch {epoch + 1:>4d}/{epochs}  -  loss: {avg_loss:.4f}")

        # Mid-training checkpoint (skip the final epoch — caller saves that)
        is_save_epoch: bool = (epoch + 1) % save_every == 0
        is_final_epoch: bool = (epoch + 1) == epochs
        if on_checkpoint is not None and is_save_epoch and not is_final_epoch:
            on_checkpoint(model, optimiser, epoch + 1, avg_loss)

    return epoch_losses, optimiser


# ======================================================================== #
#  Main entry point                                                         #
# ======================================================================== #

def main() -> None:
    """Parse arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train DeepLog LSTM")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--mode", type=str, default=os.environ.get("DEEPLOG_MODE", "mock"),
        choices=["mock", "full"],
        help="'mock' for synthetic data, 'full' for real data.",
    )
    args = parser.parse_args()

    # ---- Device selection with VRAM cap ----
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_CUDA_MEMORY_MAX)
        device = torch.device("cuda")
        print(f"[device] CUDA - VRAM capped at {_CUDA_MEMORY_MAX * 100:.0f}%")
    else:
        device = torch.device("cpu")
        print("[device] CPU")

    # ---- Load or generate sequences ----
    if args.mode == "mock":
        print("[data] Generating mock sequences (safe for interactive use) ...")
        num_keys: int = 5
        sequences: list[list[int]] = generate_mock_sequences(num_keys=num_keys)
    else:
        print(f"[data] Loading training keys from {TRAIN_KEYS_PATH} ...")
        if not TRAIN_KEYS_PATH.exists():
            print(f"ERROR: {TRAIN_KEYS_PATH} not found. Run the Drain "
                  "parser first.", file=sys.stderr)
            sys.exit(1)
        with open(TRAIN_KEYS_PATH, "r", encoding="utf-8") as fh:
            payload: dict = json.load(fh)
        sequences = payload["sequences"]
        num_keys = payload["num_keys"]

    print(f"       {len(sequences)} sessions, vocabulary size = {num_keys}")

    # ---- Dataset & DataLoader ----
    dataset = LogKeyDataset(sequences, window_size=DEFAULT_WINDOW_SIZE)
    print(f"[dataset] {len(dataset)} sliding-window samples "
          f"(window={DEFAULT_WINDOW_SIZE})")

    dataloader = DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
    )

    # ---- Model ----
    model = DeepLogModel(
        num_keys=num_keys,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
    )

    # ---- Resume from checkpoint (if it exists) ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = MODEL_DIR / "deeplog.pt"
    start_epoch: int = 0
    optimiser_state: dict | None = None

    if ckpt_path.exists():
        print(f"[resume] Loading checkpoint from {ckpt_path} ...")
        ckpt: dict = torch.load(ckpt_path, map_location=device, weights_only=True)

        # ---- num_keys mismatch guard ----
        ckpt_num_keys: int | None = ckpt.get("num_keys")
        if ckpt_num_keys is not None and ckpt_num_keys != num_keys:
            raise ValueError(
                f"num_keys mismatch: checkpoint was trained with "
                f"{ckpt_num_keys} keys but the current Drain vocabulary "
                f"has {num_keys} keys. The embedding and output layers are "
                f"incompatible. Delete the stale checkpoint and retrain "
                f"from scratch:\n"
                f"    rm {ckpt_path}\n"
                f"    python src/models/train_deeplog.py"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        optimiser_state = ckpt.get("optimizer_state_dict")
        prev_loss: float | None = ckpt.get("final_loss")
        print(f"         Resuming from epoch {start_epoch}"
              f" (last loss: {prev_loss:.4f})" if prev_loss is not None
              else f"         Resuming from epoch {start_epoch}")

    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"[model] DeepLogModel - {total_params:,} parameters\n")

    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} (requested {args.epochs}). Nothing to do.")
        return

    # ---- Mid-training checkpoint callback ----
    def _save_checkpoint(
        mdl: nn.Module,
        opt: torch.optim.Adam,
        epoch: int,
        avg_loss: float,
    ) -> None:
        torch.save(
            {
                "model_state_dict": mdl.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "num_keys": num_keys,
                "window_size": DEFAULT_WINDOW_SIZE,
                "embedding_dim": DEFAULT_EMBEDDING_DIM,
                "hidden_size": DEFAULT_HIDDEN_SIZE,
                "num_layers": DEFAULT_NUM_LAYERS,
                "dropout": DEFAULT_DROPOUT,
                "final_loss": avg_loss,
            },
            ckpt_path,
        )
        print(f"  [checkpoint] Saved at epoch {epoch}  (loss: {avg_loss:.4f})")

    # ---- Train ----
    print(f"Training from epoch {start_epoch} to {args.epochs} ...")
    losses, optimiser = train(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=DEFAULT_LR,
        device=device,
        start_epoch=start_epoch,
        optimiser_state=optimiser_state,
        on_checkpoint=_save_checkpoint,
    )

    # ---- Save checkpoint ----
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "epoch": args.epochs,
            "num_keys": num_keys,
            "window_size": DEFAULT_WINDOW_SIZE,
            "embedding_dim": DEFAULT_EMBEDDING_DIM,
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "num_layers": DEFAULT_NUM_LAYERS,
            "dropout": DEFAULT_DROPOUT,
            "final_loss": losses[-1] if losses else None,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved -> {ckpt_path}")
    print(f"Final loss: {losses[-1]:.4f}" if losses else "No training done.")


if __name__ == "__main__":
    main()
