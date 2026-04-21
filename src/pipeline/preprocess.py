"""Preprocess AWS CloudTrail JSONL logs into structured train/test CSV datasets.

Pipeline step: Raw AWS JSON Logs → Drain Parser.
This script handles the initial data cleaning, train/test splitting, noise
removal, and session grouping before logs are forwarded to the Drain parser
for template extraction.

Usage:
    python src/pipeline/preprocess.py

Environment Variables:
    INPUT_PATH  – Path to merged JSONL file  (default: data/interim/flaws_merged.jsonl)
    OUTPUT_DIR  – Directory for output CSVs  (default: data/processed/)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Final

import pandas as pd

# ---------------------------------------------------------------------------
# Configurable paths — Docker / env-variable friendly (per CLAUDE.md)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

INPUT_PATH: Final[Path] = Path(
    os.environ.get(
        "INPUT_PATH",
        str(_PROJECT_ROOT / "data" / "interim" / "flaws_merged.jsonl"),
    )
)
OUTPUT_DIR: Final[Path] = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(_PROJECT_ROOT / "data" / "processed"),
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOISE_USER_AGENTS: Final[list[str]] = ["CloudMapper", "boto3"]
NOISE_INVOKED_BY: Final[list[str]] = ["amazonaws.com"]
SESSION_KEYS: Final[list[str]] = ["sourceIPAddress", "userIdentity.arn"]
TEST_USERNAMES: Final[list[str]] = ["backup", "Level6"]


# ======================================================================== #
#  Stage 1 — Load                                                          #
# ======================================================================== #

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Stream a JSONL file line-by-line and return parsed records.

    Each line in the file is expected to be a self-contained JSON object
    representing a single CloudTrail event (previously extracted from the
    ``Records`` array of the raw JSON files).

    Args:
        path: Absolute or relative path to the JSONL input file.

    Returns:
        A list of dictionaries, one per event record.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped: str = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# ======================================================================== #
#  Stage 2 — Flatten                                                       #
# ======================================================================== #

def flatten_records(
    records: list[dict[str, Any]],
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Flatten nested CloudTrail records into a tabular DataFrame.
 
    Why two passes?
    ---------------
    1.9 M records × 1 046 columns × 8 bytes ≈ 15 GiB.  Accumulating all
    normalised chunks in a list and calling ``pd.concat`` at the end keeps
    the entire dataset in RAM twice (once as chunks, once as the result),
    which exhausts memory even on a 16–32 GiB machine.
 
    Instead we use a *disk-spill* strategy:
 
    Pass 1 — scan the first chunk to discover which columns are non-null
             and build a stable column schema.  This avoids the jagged-
             column problem where different event types produce different
             fields, which would cause ``pd.concat`` to create a massive
             sparse union.
 
    Pass 2 — normalise each chunk, reindex it to the fixed schema, append
             it to a temporary CSV file, then immediately ``del`` the chunk
             so only *one* chunk lives in RAM at a time.
 
    Finally, ``pd.read_csv`` streams the temp file back into a single
    DataFrame and the temp file is removed.
 
    Args:
        records:    List of raw CloudTrail event dictionaries.
        chunk_size: Rows per batch (50 000 ≈ 300–400 MB per chunk).
 
    Returns:
        A :class:`~pandas.DataFrame` with one row per event and
        dot-separated column names for formerly nested fields.
    """
    import tempfile
 
    total: int = len(records)
 
    # ------------------------------------------------------------------
    # Pass 1: derive a stable, non-null column schema from the first chunk
    # ------------------------------------------------------------------
    sample: pd.DataFrame = pd.json_normalize(records[:chunk_size], sep=".")
    sample = sample.dropna(axis=1, how="all")
    schema_cols: list[str] = list(sample.columns)
    del sample
 
    print(f"      Schema derived from first chunk: {len(schema_cols)} columns.")
 
    # ------------------------------------------------------------------
    # Pass 2: stream every chunk to a temp CSV file
    # ------------------------------------------------------------------
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".csv", prefix="preprocess_")
    os.close(tmp_fd)                          # close the raw OS handle
    tmp_path: Path = Path(tmp_path_str)
 
    try:
        for start in range(0, total, chunk_size):
            end: int = min(start + chunk_size, total)
            batch: list[dict[str, Any]] = records[start:end]
 
            chunk: pd.DataFrame = pd.json_normalize(batch, sep=".")
            # Reindex: fills missing columns with NaN, drops unknown extras
            chunk = chunk.reindex(columns=schema_cols)
 
            write_header: bool = start == 0
            chunk.to_csv(
                tmp_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            del chunk          # free immediately — only one chunk in RAM
 
            print(
                f"      Flattened {end:>{len(str(total))},} / {total:,} records …",
                end="\r",
            )
 
        print()  # newline after progress line
 
        # --------------------------------------------------------------
        # Read the temp CSV back in one shot (pandas streams it)
        # --------------------------------------------------------------
        print(f"      Reading flattened data from temp file …")
        df: pd.DataFrame = pd.read_csv(tmp_path, low_memory=False)
 
    finally:
        tmp_path.unlink(missing_ok=True)   # always clean up
 
    print(f"      Final shape: {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ======================================================================== #
#  Stage 3 — Temporal sort                                                 #
# ======================================================================== #

def convert_and_sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert *eventTime* to UTC datetime and sort chronologically.

    Args:
        df: DataFrame containing a string ``eventTime`` column.

    Returns:
        Chronologically sorted DataFrame with a reset integer index.
    """
    df["eventTime"] = pd.to_datetime(df["eventTime"], utc=True, format="ISO8601")
    df = df.sort_values("eventTime").reset_index(drop=True)
    return df


# ======================================================================== #
#  Stage 4 — Null handling                                                 #
# ======================================================================== #

def fill_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``NaN`` in every object (string) column with ``'None'``.

    Numeric and datetime columns are left untouched.

    Args:
        df: Any pandas DataFrame.

    Returns:
        The same DataFrame with string nulls filled.
    """
    object_cols: pd.Index = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("None")
    return df


# ======================================================================== #
#  Stage 5 — Train / Test split                                            #
# ======================================================================== #

def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the full DataFrame into *train* and *test* subsets.

    * **train_df** — events where ``userIdentity.type == 'Root'``
      (normal administrative baseline).
    * **test_df** — events where ``userIdentity.userName`` is
      ``'backup'`` or ``'Level6'``, **or** ``userIdentity.type``
      is ``'AssumedRole'`` (anomalous / attack-simulation traffic).

    Args:
        df: Cleaned, full DataFrame.

    Returns:
        A ``(train_df, test_df)`` tuple of independent copies.
    """
    train_df: pd.DataFrame = df.loc[
        df["userIdentity.type"] == "Root"
    ].copy()

    test_mask: pd.Series = (
        df["userIdentity.userName"].isin(TEST_USERNAMES)
        | (df["userIdentity.type"] == "AssumedRole")
    )
    test_df: pd.DataFrame = df.loc[test_mask].copy()

    return train_df, test_df


# ======================================================================== #
#  Stage 6 — Noise filtering                                               #
# ======================================================================== #

def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows generated by automated tooling or internal AWS services.

    The following rows are dropped:

    * ``userAgent`` contains *CloudMapper* or *boto3*.
    * ``userIdentity.invokedBy`` contains *amazonaws.com*.

    Column-existence checks are performed so the function is safe to
    call even when particular columns are absent from the data.

    Args:
        df: A train or test DataFrame.

    Returns:
        Filtered DataFrame with a fresh integer index.
    """
    # --- userAgent noise ---
    if "userAgent" in df.columns:
        for pattern in NOISE_USER_AGENTS:
            df = df[~df["userAgent"].str.contains(pattern, case=False, na=False)]

    # --- invokedBy noise (nested under userIdentity in raw JSON) ---
    invoked_col: str = "userIdentity.invokedBy"
    if invoked_col in df.columns:
        for pattern in NOISE_INVOKED_BY:
            df = df[~df[invoked_col].str.contains(pattern, case=False, na=False)]

    return df.reset_index(drop=True)


# ======================================================================== #
#  Stage 7 — Session grouping                                             #
# ======================================================================== #

def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Group records into chronological sessions.

    A *session* is defined by the composite key
    ``(sourceIPAddress, userIdentity.arn)``.  Each row receives:

    * ``sessionId``    — unique integer identifying the session.
    * ``sessionSeqNo`` — 0-based position of the event inside its
      session (ordered by ``eventTime``).

    Args:
        df: A filtered train or test DataFrame.

    Returns:
        DataFrame augmented with ``sessionId`` and ``sessionSeqNo``,
        sorted by ``(sessionId, eventTime)``.
    """
    df = df.sort_values("eventTime").reset_index(drop=True)

    present_keys: list[str] = [k for k in SESSION_KEYS if k in df.columns]

    if not present_keys:
        df["sessionId"] = 0
        df["sessionSeqNo"] = df.groupby("sessionId").cumcount()
        return df

    df["sessionId"] = df.groupby(present_keys, sort=False).ngroup()
    df = df.sort_values(["sessionId", "eventTime"]).reset_index(drop=True)
    df["sessionSeqNo"] = df.groupby("sessionId").cumcount()

    return df


# ======================================================================== #
#  I/O helper                                                              #
# ======================================================================== #

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to a CSV file, creating parent dirs as needed.

    Args:
        df: DataFrame to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ======================================================================== #
#  Pipeline orchestrator                                                   #
# ======================================================================== #

def main() -> None:
    """Execute the full preprocessing pipeline end-to-end."""
    print(f"[1/7] Loading records from {INPUT_PATH} ...")
    records: list[dict[str, Any]] = load_jsonl(INPUT_PATH)
    print(f"      Loaded {len(records):,} raw records.\n")

    print("[2/7] Flattening nested JSON with pd.json_normalize ...")
    df: pd.DataFrame = flatten_records(records)
    # Free raw list immediately to reclaim memory
    del records
    print(f"      Result: {len(df):,} rows x {len(df.columns)} columns.\n")

    print("[3/7] Converting eventTime to datetime & sorting chronologically ...")
    df = convert_and_sort_by_time(df)
    print(f"      Time range: {df['eventTime'].iloc[0]}  →  {df['eventTime'].iloc[-1]}\n")

    print("[4/7] Filling missing string values with 'None' ...")
    df = fill_missing_strings(df)
    print("      Done.\n")

    print("[5/7] Splitting into train / test sets ...")
    train_df, test_df = split_train_test(df)
    print(f"      Train (Root):                       {len(train_df):>10,} rows")
    print(f"      Test  (backup/Level6/AssumedRole):  {len(test_df):>10,} rows\n")

    # Free the large combined frame
    del df

    print("[6/7] Filtering automated noise ...")
    train_df = filter_noise(train_df)
    test_df = filter_noise(test_df)
    print(f"      Train after noise filter: {len(train_df):>10,} rows")
    print(f"      Test  after noise filter: {len(test_df):>10,} rows\n")

    print("[7/7] Assigning chronological session IDs ...")
    train_df = assign_sessions(train_df)
    test_df = assign_sessions(test_df)
    print(f"      Train sessions: {train_df['sessionId'].nunique():,}")
    print(f"      Test  sessions: {test_df['sessionId'].nunique():,}\n")

    # ---- Persist ----------------------------------------------------------
    train_path: Path = OUTPUT_DIR / "train_sessions.csv"
    test_path: Path = OUTPUT_DIR / "test_sessions.csv"

    save_csv(train_df, train_path)
    save_csv(test_df, test_path)

    print(f"Saved train data -> {train_path}")
    print(f"Saved test data  -> {test_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
