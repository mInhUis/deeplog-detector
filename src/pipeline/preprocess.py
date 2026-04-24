"""Preprocess AWS CloudTrail JSONL logs into structured train/test CSV datasets.

Pipeline step: Raw AWS JSON Logs -> Drain Parser.
This script handles the initial data cleaning, train/test splitting, noise
removal, and session grouping before logs are forwarded to the Drain parser
for template extraction.

Usage:
    python src/pipeline/preprocess.py

Environment Variables:
    INPUT_PATH  - Path to merged JSONL file  (default: data/interim/flaws_merged.jsonl)
    OUTPUT_DIR  - Directory for output CSVs  (default: data/processed/)
"""

from __future__ import annotations

import json
import os
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any, Final, Generator

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

# Columns always kept regardless of fill rate — required by downstream stages.
REQUIRED_COLS: Final[list[str]] = [
    "eventTime", "eventName", "eventSource", "eventType",
    "sourceIPAddress", "userAgent", "awsRegion",
    "requestID", "errorCode", "errorMessage",
    "userIdentity.type", "userIdentity.userName", "userIdentity.arn",
    "userIdentity.invokedBy", "userIdentity.accountId",
    "requestParameters.bucketName", "requestParameters.roleName",
    "requestParameters.userName", "requestParameters.keyId",
]

# Columns with fill rate below this threshold are dropped.
# Reduces ~1046 columns to ~40-60, shrinking the DataFrame from ~15 GB to < 2 GB.
# Raise to 0.10 if still OOM; lower to 0.01 to keep more detail for Drain.
COL_FILL_THRESHOLD: Final[float] = 0.05

# Records normalised per batch — each batch uses ~300-400 MB RAM.
CHUNK_SIZE: Final[int] = 50_000


# ======================================================================== #
#  Stage 1 — Load (streaming generator — no list materialisation)         #
# ======================================================================== #

def stream_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield parsed records one-by-one from a JSONL file.

    Unlike the previous ``load_jsonl`` approach, this generator never
    holds more than one record in memory at a time.  Loading all 1.9 M
    CloudTrail records into a Python list consumed ~3-4 GiB before Stage 2
    even started, leaving too little headroom on a 12.7 GiB Colab instance.

    Args:
        path: Absolute or relative path to the JSONL input file.

    Yields:
        One parsed dictionary per non-empty line.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped: str = line.strip()
            if stripped:
                yield json.loads(stripped)


def count_lines(path: Path) -> int:
    """Count non-empty lines in *path* without loading content into RAM.

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of non-empty lines (= number of records).
    """
    count: int = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


# ======================================================================== #
#  Stage 2 — Flatten                                                       #
# ======================================================================== #

def _derive_schema(path: Path) -> list[str]:
    """Sample the first chunk to build a memory-safe column schema.

    Reads only the first ``CHUNK_SIZE`` records to discover which columns
    exist and what their fill rates are.  Retains a column when *either*:

    * fill rate >= ``COL_FILL_THRESHOLD``, **or**
    * it is listed in ``REQUIRED_COLS`` (pipeline-critical fields for
      Drain / DeepLog).

    This typically reduces ~1 046 raw columns to ~40-60.

    Args:
        path: Path to the JSONL file.

    Returns:
        Ordered list of column names forming the stable schema.
    """
    sample_records: list[dict[str, Any]] = list(
        islice(stream_jsonl(path), CHUNK_SIZE)
    )
    sample: pd.DataFrame = pd.json_normalize(sample_records, sep=".")
    del sample_records

    fill_rates: pd.Series = sample.notna().mean()
    schema: list[str] = [
        col for col in sample.columns
        if fill_rates[col] >= COL_FILL_THRESHOLD or col in REQUIRED_COLS
    ]
    dropped: int = len(sample.columns) - len(schema)
    del sample

    print(
        f"      Schema: kept {len(schema)} columns, "
        f"dropped {dropped} sparse columns "
        f"(fill rate < {COL_FILL_THRESHOLD:.0%})."
    )
    return schema


def flatten_records(path: Path, total: int) -> pd.DataFrame:
    """Stream-flatten all CloudTrail records into a tabular DataFrame.

    Why this approach is necessary on 12.7 GiB Colab RAM
    -----------------------------------------------------
    The previous implementation had two memory bombs:

    1. ``load_jsonl`` materialised all 1.9 M records into a Python list
       (~3-4 GiB consumed before any processing started).
    2. ``pd.json_normalize`` on the full list tried to allocate a single
       (1 939 207 x 1 046) object array = ~15 GiB in one shot.

    This function eliminates both by:

    * Never building a full-dataset list — reads the file as a generator.
    * Processing ``CHUNK_SIZE`` records at a time (~400 MB each).
    * Appending each chunk directly to a temp CSV and deleting it, so
      only one chunk is in RAM at a time.
    * Pruning ~1 000 sparse columns via ``_derive_schema`` before writing,
      so the final ``pd.read_csv`` loads a compact ~1-2 GiB file.

    Args:
        path:  Path to the source JSONL file.
        total: Total record count (used only for progress display).

    Returns:
        A :class:`~pandas.DataFrame` with one row per event and
        dot-separated column names for formerly nested fields.
    """
    # Step 1 — build pruned column schema from first chunk
    schema_cols: list[str] = _derive_schema(path)

    # Step 2 — stream file -> chunks -> temp CSV, one chunk in RAM at a time
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".csv", prefix="preprocess_")
    os.close(tmp_fd)
    tmp_path: Path = Path(tmp_path_str)

    try:
        processed: int = 0
        first_write: bool = True
        stream: Generator[dict[str, Any], None, None] = stream_jsonl(path)

        while True:
            batch: list[dict[str, Any]] = list(islice(stream, CHUNK_SIZE))
            if not batch:
                break

            chunk: pd.DataFrame = pd.json_normalize(batch, sep=".")
            del batch                            # free raw dicts immediately

            chunk = chunk.reindex(columns=schema_cols)   # prune & align

            # Un-nest list-valued cells before spilling to CSV so that the
            # round-trip does not stringify them as "['x']" / "['x','y']".
            for col in chunk.columns:
                if col in {"userAgent", "resources", "eventTime"}:
                    continue
                if chunk[col].dtype == "object":
                    is_list = chunk[col].apply(lambda x: isinstance(x, list))
                    if is_list.any():
                        chunk.loc[is_list, col] = chunk.loc[is_list, col].apply(
                            lambda x: str(x[0]) if len(x) == 1 else ",".join(map(str, x))
                        )

            chunk.to_csv(
                tmp_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            del chunk                            # free before next iteration
            first_write = False

            processed += CHUNK_SIZE
            shown: int = min(processed, total)
            print(
                f"      Flattened {shown:>{len(str(total))},} / {total:,} records ...",
                end="\r",
            )

        print()

        # Step 3 — read back the compact pruned CSV
        print("      Reading pruned data into DataFrame ...")
        df: pd.DataFrame = pd.read_csv(tmp_path, low_memory=False)

    finally:
        tmp_path.unlink(missing_ok=True)    # always clean up temp file

    print(f"      Final shape: {len(df):,} rows x {len(df.columns)} columns.")
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
#  Stage 4 — Null handling & Type cleaning                                 #
# ======================================================================== #

def fill_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN in all non-time columns with 'None' and extract nested lists."""

    # 1. Un-nest lists (fixes the requestParameters.maxResults Bug)
    for col in df.columns:
        if col in {"userAgent", "resources", "eventTime"}:
            continue
        if df[col].dtype == "object":
            is_list = df[col].apply(lambda x: isinstance(x, list))
            if is_list.any():
                df.loc[is_list, col] = df.loc[is_list, col].apply(
                    lambda x: str(x[0]) if len(x) == 1 else ",".join(map(str, x))
                )

    # 2. Fix NaN Bug: Pandas infers empty columns as float64, missing the 'object' check.
    # We target all columns except eventTime, fill NaNs, and force them to strings.
    cols_to_fill = df.columns.difference(["eventTime"])
    df[cols_to_fill] = df[cols_to_fill].fillna("None").astype(str)

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
    if "userAgent" in df.columns:
        for pattern in NOISE_USER_AGENTS:
            df = df[~df["userAgent"].str.contains(pattern, case=False, na=False)]

    invoked_col: str = "userIdentity.invokedBy"
    if invoked_col in df.columns:
        for pattern in NOISE_INVOKED_BY:
            df = df[~df[invoked_col].str.contains(pattern, case=False, na=False)]

    return df.reset_index(drop=True)


# ======================================================================== #
#  Stage 7 — Session grouping                                              #
# ======================================================================== #

def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Group records into chronological sessions and restore global temporal sort."""
    df = df.sort_values("eventTime").reset_index(drop=True)

    present_keys: list[str] = [k for k in SESSION_KEYS if k in df.columns]

    if not present_keys:
        df["sessionId"] = 0
        df["sessionSeqNo"] = df.groupby("sessionId").cumcount()
        return df

    # Group and assign sequential numbers within the session
    df["sessionId"] = df.groupby(present_keys, sort=False).ngroup()
    df = df.sort_values(["sessionId", "eventTime"]).reset_index(drop=True)
    df["sessionSeqNo"] = df.groupby("sessionId").cumcount()

    # Fix: Restore global temporal sorting before writing to CSV.
    # Secondary keys (sessionId, sessionSeqNo) break eventTime ties
    # deterministically so intra-session order is preserved without
    # relying on sort stability.
    df = df.sort_values(["eventTime", "sessionId", "sessionSeqNo"]).reset_index(drop=True)

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
    print(f"[1/7] Counting records in {INPUT_PATH} ...")
    total: int = count_lines(INPUT_PATH)
    print(f"      Found {total:,} records.\n")

    print("[2/7] Flattening nested JSON (streaming, disk-spill) ...")
    df: pd.DataFrame = flatten_records(INPUT_PATH, total)
    print(f"      Result: {len(df):,} rows x {len(df.columns)} columns.\n")

    print("[3/7] Converting eventTime to datetime & sorting chronologically ...")
    df = convert_and_sort_by_time(df)
    print(f"      Time range: {df['eventTime'].iloc[0]}  ->  {df['eventTime'].iloc[-1]}\n")

    print("[4/7] Filling missing string values with 'None' ...")
    df = fill_missing_strings(df)
    print("      Done.\n")

    print("[5/7] Splitting into train / test sets ...")
    train_df, test_df = split_train_test(df)
    print(f"      Train (Root):                       {len(train_df):>10,} rows")
    print(f"      Test  (backup/Level6/AssumedRole):  {len(test_df):>10,} rows\n")
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

    train_path: Path = OUTPUT_DIR / "train_sessions.csv"
    test_path: Path = OUTPUT_DIR / "test_sessions.csv"

    save_csv(train_df, train_path)
    save_csv(test_df, test_path)

    print(f"Saved train data -> {train_path}")
    print(f"Saved test data  -> {test_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()