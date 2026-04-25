"""Build DeepLog training keys from preprocessed CloudTrail sessions.

This script bridges Phase 1 (preprocess.py) and Phase 3 (train_deeplog.py).
It reads ``data/processed/train_sessions.csv``, runs Drain on the flattened
event corpus to learn a global template vocabulary, re-splits the resulting
log-key sequence by ``sessionId``, and writes the artifacts that
``train_deeplog.py`` expects on disk.

Outputs
-------
* ``data/processed/train_log_keys.json``  — schema:
      ``{"sequences": list[list[int]], "num_keys": int}``
* ``data/processed/train_templates.json`` — schema:
      ``{str(template_id): template_str}``  (JSON keys must be strings)

Skill 3 — Data Leakage Watchdog
-------------------------------
The script refuses to run on any CSV whose basename contains ``"test"``.
Drain must be fit on training sessions only; the test split must be parsed
at inference time using the frozen ``train_templates.json`` vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to sys.path so ``python src/pipeline/build_train_keys.py``
# works without installation (mirrors how the other pipeline scripts run).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.main import event_to_log_string  # noqa: E402
from src.parser.drain_parser import DrainParser  # noqa: E402


_DEFAULT_SESSIONS_CSV: Path = _PROJECT_ROOT / "data" / "processed" / "train_sessions.csv"
_DEFAULT_OUT_KEYS: Path = _PROJECT_ROOT / "data" / "processed" / "train_log_keys.json"
_DEFAULT_OUT_TEMPLATES: Path = _PROJECT_ROOT / "data" / "processed" / "train_templates.json"

# Flat CSV columns produced by preprocess.py that feed event_to_log_string
_STRING_COLUMNS: tuple[str, ...] = (
    "eventSource",
    "eventName",
    "sourceIPAddress",
    "userIdentity.arn",
)


def _row_to_event_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Reshape a flat CSV row into the nested dict shape expected by
    ``event_to_log_string`` (which reads ``userIdentity.arn`` from a nested
    ``userIdentity`` dict, matching the raw CloudTrail JSON schema)."""
    return {
        "eventSource": row.get("eventSource", "") or "",
        "eventName": row.get("eventName", "") or "",
        "sourceIPAddress": row.get("sourceIPAddress", "") or "",
        "userIdentity": {"arn": row.get("userIdentity.arn", "") or ""},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sessions_csv",
        type=Path,
        default=Path(os.environ.get("TRAIN_SESSIONS_CSV", str(_DEFAULT_SESSIONS_CSV))),
        help="Input train_sessions.csv produced by preprocess.py",
    )
    parser.add_argument(
        "--out_keys",
        type=Path,
        default=Path(os.environ.get("TRAIN_KEYS_PATH", str(_DEFAULT_OUT_KEYS))),
        help="Output JSON file consumed by train_deeplog.py",
    )
    parser.add_argument(
        "--out_templates",
        type=Path,
        default=Path(os.environ.get("TRAIN_TEMPLATES_PATH", str(_DEFAULT_OUT_TEMPLATES))),
        help="Output JSON file with the frozen Drain template vocabulary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Skill 3 — Data Leakage Watchdog
    if "test" in args.sessions_csv.name.lower():
        raise ValueError(
            f"Refusing to build train keys from a test split: {args.sessions_csv}. "
            "Drain must fit on training sessions only."
        )
    if not args.sessions_csv.exists():
        raise FileNotFoundError(f"Sessions CSV not found: {args.sessions_csv}")

    print(f"[load] {args.sessions_csv}")
    df = pd.read_csv(args.sessions_csv, low_memory=False)
    print(f"       {len(df):,} rows, {df['sessionId'].nunique():,} sessions")

    # Drop rows missing the session key; coerce string-field NaNs so Drain
    # output is deterministic.
    df = df.dropna(subset=["sessionId"]).copy()
    for col in _STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Order only — sessionSeqNo gaps are fine; stable sort breaks ties on CSV order.
    df = df.sort_values(["sessionId", "sessionSeqNo"], kind="stable")

    all_log_strings: list[str] = []   # length N_total
    session_lengths: list[int] = []   # length S
    skipped_empty: int = 0

    for _session_id, group in df.groupby("sessionId", sort=False):
        if len(group) == 0:
            skipped_empty += 1
            continue
        strings = [event_to_log_string(_row_to_event_dict(r)) for r in group.to_dict("records")]
        all_log_strings.extend(strings)
        session_lengths.append(len(strings))

    if skipped_empty:
        print(f"[warn] Skipped {skipped_empty} empty session group(s)")

    n_total: int = len(all_log_strings)
    print(f"[drain] Fitting Drain on {n_total:,} events ...")
    parser_obj = DrainParser()
    log_keys, templates = parser_obj.fit_transform(all_log_strings)
    n_templates: int = len(templates)
    print(f"        Extracted {n_templates} unique template(s)")

    # Re-split the flat key list into per-session sequences (mirrors
    # src/main.py:604-608).
    sequences: list[list[int]] = []
    offset: int = 0
    for length in session_lengths:
        sequences.append(log_keys[offset : offset + length])
        offset += length

    args.out_keys.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_keys, "w", encoding="utf-8") as fh:
        json.dump({"sequences": sequences, "num_keys": n_templates}, fh)
    print(f"[write] {args.out_keys}")

    args.out_templates.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_templates, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in templates.items()}, fh, indent=2)
    print(f"[write] {args.out_templates}")

    print(
        f"[done] {len(sequences)} sessions, {n_total:,} events, {n_templates} templates"
    )


if __name__ == "__main__":
    main()
