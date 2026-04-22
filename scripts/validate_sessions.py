"""Validate train_sessions.csv and test_sessions.csv against Phase 1 acceptance criteria.

Runs a PASS/FAIL report for:
  1. Tabular structure (no nested JSON strings)
  2. Temporal sorting by eventTime
  3. Null handling (string cols filled with "None")
  4. Train/test split logic (Root vs. backup/Level6/AssumedRole)
  5. Noise filtering (userAgent, userIdentity.invokedBy)
  6. Session grouping (sessionId + sessionSeqNo monotonic from 0, sorted by eventTime)

Usage:
    python scripts/validate_sessions.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(
    os.environ.get(
        "PROCESSED_DIR",
        Path(__file__).resolve().parent.parent / "data" / "processed",
    )
)
TRAIN_CSV = PROCESSED_DIR / "train_sessions.csv"
TEST_CSV = PROCESSED_DIR / "test_sessions.csv"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f"\n         -> {self.detail}"
        return line


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_tabular_structure(df: pd.DataFrame) -> CheckResult:
    """No cell should be a JSON-looking nested string ({...} or [...])."""
    offenders: list[str] = []
    for col in df.select_dtypes(include=["object"]).columns:
        # userAgent is legitimately wrapped in [..] (e.g. "[S3Console/0.4]"),
        # and the `resources` column is a list serialisation — skip those.
        if col in {"userAgent", "resources"}:
            continue
        series = df[col].astype(str).str.strip()
        mask = series.str.startswith("{") | (
            series.str.startswith("[") & ~series.str.startswith("[MOCK")
        )
        if mask.any():
            sample = series[mask].iloc[0][:80]
            offenders.append(f"{col} (e.g. {sample!r})")
    if offenders:
        return CheckResult(
            "Tabular structure — no nested JSON strings",
            False,
            f"{len(offenders)} column(s) contain nested-looking values: "
            + "; ".join(offenders[:5]),
        )
    return CheckResult("Tabular structure — no nested JSON strings", True)


def check_temporal_sorting(df: pd.DataFrame) -> CheckResult:
    if "eventTime" not in df.columns:
        return CheckResult("Temporal sorting by eventTime", False, "eventTime column missing")
    times = pd.to_datetime(df["eventTime"], utc=True, errors="coerce")
    if times.isna().any():
        n_bad = int(times.isna().sum())
        return CheckResult(
            "Temporal sorting by eventTime",
            False,
            f"{n_bad} unparseable eventTime values",
        )
    if not times.is_monotonic_increasing:
        # find first offending index
        diffs = times.diff().dt.total_seconds()
        first_bad = int((diffs < 0).idxmax())
        return CheckResult(
            "Temporal sorting by eventTime",
            False,
            f"first out-of-order row at index {first_bad}",
        )
    return CheckResult("Temporal sorting by eventTime", True)


def check_null_handling(df: pd.DataFrame) -> CheckResult:
    """Object columns must have no NaN/None; missing values should be the literal string 'None'."""
    offenders: list[str] = []
    for col in df.select_dtypes(include=["object"]).columns:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            offenders.append(f"{col} ({n_null} NaN)")
    if offenders:
        return CheckResult(
            "Null handling — object columns filled with 'None'",
            False,
            "columns with NaN: " + ", ".join(offenders[:10]),
        )
    return CheckResult("Null handling — object columns filled with 'None'", True)


def check_train_split_logic(df: pd.DataFrame) -> CheckResult:
    col = "userIdentity.type"
    if col not in df.columns:
        return CheckResult("Train split — userIdentity.type == 'Root' only", False, f"{col} missing")
    bad = df[df[col] != "Root"]
    if not bad.empty:
        uniques = bad[col].unique().tolist()
        return CheckResult(
            "Train split — userIdentity.type == 'Root' only",
            False,
            f"{len(bad)} non-Root rows; unique types: {uniques[:5]}",
        )
    return CheckResult("Train split — userIdentity.type == 'Root' only", True)


def check_test_split_logic(df: pd.DataFrame) -> CheckResult:
    uname_col = "userIdentity.userName"
    type_col = "userIdentity.type"
    missing = [c for c in (uname_col, type_col) if c not in df.columns]
    if missing:
        return CheckResult(
            "Test split — backup/Level6 or AssumedRole",
            False,
            f"missing columns: {missing}",
        )
    mask = (
        df[uname_col].isin(["backup", "Level6"])
        | (df[type_col] == "AssumedRole")
    )
    bad = df[~mask]
    if not bad.empty:
        sample_types = bad[type_col].value_counts().head(5).to_dict()
        sample_users = bad[uname_col].value_counts().head(5).to_dict()
        return CheckResult(
            "Test split — backup/Level6 or AssumedRole",
            False,
            f"{len(bad)} violating rows; types={sample_types}, users={sample_users}",
        )
    return CheckResult("Test split — backup/Level6 or AssumedRole", True)


def check_noise_filter(df: pd.DataFrame) -> CheckResult:
    issues: list[str] = []

    ua_col = "userAgent"
    if ua_col in df.columns:
        ua = df[ua_col].astype(str).str.lower()
        n_cm = int(ua.str.contains("cloudmapper", na=False).sum())
        n_boto = int(ua.str.contains("boto3", na=False).sum())
        if n_cm:
            issues.append(f"{n_cm} rows contain 'CloudMapper' in userAgent")
        if n_boto:
            issues.append(f"{n_boto} rows contain 'boto3' in userAgent")
    else:
        issues.append("userAgent column missing")

    inv_col = "userIdentity.invokedBy"
    if inv_col in df.columns:
        inv = df[inv_col].astype(str).str.lower()
        n_aws = int(inv.str.contains("amazonaws.com", na=False).sum())
        if n_aws:
            issues.append(f"{n_aws} rows contain 'amazonaws.com' in userIdentity.invokedBy")
    else:
        issues.append("userIdentity.invokedBy column missing")

    if issues:
        return CheckResult("Noise filtering — userAgent / invokedBy", False, "; ".join(issues))
    return CheckResult("Noise filtering — userAgent / invokedBy", True)


def check_session_grouping(df: pd.DataFrame) -> CheckResult:
    required = {"sessionId", "sessionSeqNo", "eventTime"}
    missing = required - set(df.columns)
    if missing:
        return CheckResult(
            "Session grouping — sessionId/sessionSeqNo integrity",
            False,
            f"missing columns: {sorted(missing)}",
        )

    times = pd.to_datetime(df["eventTime"], utc=True, errors="coerce")
    if times.isna().any():
        return CheckResult(
            "Session grouping — sessionId/sessionSeqNo integrity",
            False,
            "unparseable eventTime values",
        )

    bad_seq: list[object] = []
    bad_time: list[object] = []

    work = df.assign(_t=times).reset_index(drop=True)
    for sid, grp in work.groupby("sessionId", sort=False):
        seq = grp["sessionSeqNo"].tolist()
        expected = list(range(len(seq)))
        if seq != expected:
            bad_seq.append(sid)
            if len(bad_seq) >= 5:
                pass  # collect a few, then move on
        t = grp["_t"].tolist()
        if t != sorted(t):
            bad_time.append(sid)

    issues: list[str] = []
    if bad_seq:
        issues.append(f"{len(bad_seq)} session(s) with non-sequential sessionSeqNo (e.g. {bad_seq[:3]})")
    if bad_time:
        issues.append(f"{len(bad_time)} session(s) not sorted by eventTime (e.g. {bad_time[:3]})")
    if issues:
        return CheckResult(
            "Session grouping — sessionId/sessionSeqNo integrity",
            False,
            "; ".join(issues),
        )
    return CheckResult("Session grouping — sessionId/sessionSeqNo integrity", True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
SHARED_CHECKS: list[tuple[str, Callable[[pd.DataFrame], CheckResult]]] = [
    ("Tabular structure",   check_tabular_structure),
    ("Temporal sorting",    check_temporal_sorting),
    ("Null handling",       check_null_handling),
    ("Noise filtering",     check_noise_filter),
    ("Session grouping",    check_session_grouping),
]


def run_suite(name: str, df: pd.DataFrame, split_check: Callable[[pd.DataFrame], CheckResult]) -> list[CheckResult]:
    print(f"\n=== {name}  ({len(df):,} rows, {len(df.columns)} cols) ===")
    results = [fn(df) for _, fn in SHARED_CHECKS]
    results.append(split_check(df))
    for r in results:
        print(r)
    return results


def main() -> None:
    print(f"Loading: {TRAIN_CSV}")
    print(f"Loading: {TEST_CSV}")
    # Disable pandas' default NA inference: "None" is in the default
    # na_values list, which would convert the literal "None" fill-values
    # written by preprocess.py back into NaN and trigger a false failure
    # in check_null_handling. Only treat empty strings as missing.
    train = pd.read_csv(TRAIN_CSV, low_memory=False, keep_default_na=False, na_values=[""])
    test = pd.read_csv(TEST_CSV, low_memory=False, keep_default_na=False, na_values=[""])

    train_results = run_suite("train_sessions.csv", train, check_train_split_logic)
    test_results = run_suite("test_sessions.csv", test, check_test_split_logic)

    # ----- summary -----
    all_results = [("train", r) for r in train_results] + [("test", r) for r in test_results]
    total = len(all_results)
    passed = sum(1 for _, r in all_results if r.passed)
    failed = total - passed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")

    if failed:
        print("\n  Failed checks:")
        for split, r in all_results:
            if not r.passed:
                print(f"    - [{split}] {r.name}")
        raise SystemExit(1)

    print("\n  All acceptance criteria satisfied.")


if __name__ == "__main__":
    main()
