"""Unit tests for src/pipeline/preprocess.py.

Uses a small, mocked dataset of exactly 4 CloudTrail events to verify every
stage of the preprocessing pipeline without touching real data or heavy I/O.

Event A — Normal Root administrator action           → expected in train_df
Event B — Attacker action (userName='Level6')         → expected in test_df
Event C — Attacker action (userName='backup')         → expected in test_df
Event D — Noisy scanner action (userAgent='CloudMapper') → expected NOWHERE
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pytest

from src.pipeline.preprocess import (
    assign_sessions,
    convert_and_sort_by_time,
    count_lines,
    fill_missing_strings,
    filter_noise,
    flatten_records,
    save_csv,
    split_train_test,
    stream_jsonl,
)

# ======================================================================== #
#  Mock data                                                                #
# ======================================================================== #

EVENT_A: Final[dict[str, Any]] = {
    "eventTime": "2017-02-12T20:00:00Z",
    "eventName": "ListBuckets",
    "eventSource": "s3.amazonaws.com",
    "sourceIPAddress": "10.0.0.1",
    "userAgent": "aws-console",
    "userIdentity": {
        "type": "Root",
        "arn": "arn:aws:iam::111111111111:root",
        "accountId": "111111111111",
        "principalId": "111111111111",
    },
    "awsRegion": "us-east-1",
    "eventID": "aaaa-aaaa",
    "eventType": "AwsApiCall",
    "requestParameters": None,
    "responseElements": None,
}

EVENT_B: Final[dict[str, Any]] = {
    "eventTime": "2017-02-12T21:00:00Z",
    "eventName": "GetObject",
    "eventSource": "s3.amazonaws.com",
    "sourceIPAddress": "192.168.1.100",
    "userAgent": "aws-cli/1.0",
    "userIdentity": {
        "type": "IAMUser",
        "arn": "arn:aws:iam::111111111111:user/Level6",
        "accountId": "111111111111",
        "principalId": "AIDA000000000000LEVEL6",
        "userName": "Level6",
    },
    "awsRegion": "us-east-1",
    "eventID": "bbbb-bbbb",
    "eventType": "AwsApiCall",
    "requestParameters": None,
    "responseElements": None,
}

EVENT_C: Final[dict[str, Any]] = {
    "eventTime": "2017-02-12T19:00:00Z",
    "eventName": "PutObject",
    "eventSource": "s3.amazonaws.com",
    "sourceIPAddress": "192.168.1.200",
    "userAgent": "aws-sdk-java",
    "userIdentity": {
        "type": "IAMUser",
        "arn": "arn:aws:iam::111111111111:user/backup",
        "accountId": "111111111111",
        "principalId": "AIDA000000000000BACKUP",
        "userName": "backup",
    },
    "awsRegion": "us-east-1",
    "eventID": "cccc-cccc",
    "eventType": "AwsApiCall",
    "requestParameters": None,
    "responseElements": None,
}

EVENT_D: Final[dict[str, Any]] = {
    "eventTime": "2017-02-12T18:00:00Z",
    "eventName": "DescribeInstances",
    "eventSource": "ec2.amazonaws.com",
    "sourceIPAddress": "10.0.0.1",
    "userAgent": "CloudMapper",
    "userIdentity": {
        "type": "Root",
        "arn": "arn:aws:iam::111111111111:root",
        "accountId": "111111111111",
        "principalId": "111111111111",
    },
    "awsRegion": "us-east-1",
    "eventID": "dddd-dddd",
    "eventType": "AwsApiCall",
    "requestParameters": None,
    "responseElements": None,
}

MOCK_RECORDS: Final[list[dict[str, Any]]] = [EVENT_A, EVENT_B, EVENT_C, EVENT_D]


# ======================================================================== #
#  Test helpers                                                             #
# ======================================================================== #

def _flatten_list(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten an in-memory list of records through flatten_records.

    flatten_records now expects a JSONL file path (streaming pipeline).
    This helper serialises the list to a self-cleaning NamedTemporaryFile
    so individual tests stay free of tmp_path boilerplate.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
        tmp = Path(fh.name)
    try:
        return flatten_records(tmp, len(records))
    finally:
        tmp.unlink(missing_ok=True)


# ======================================================================== #
#  Fixtures                                                                 #
# ======================================================================== #

@pytest.fixture()
def mock_jsonl(tmp_path: Path) -> Path:
    """Write the four mock events to a temporary JSONL file.

    The file is automatically cleaned up by pytest's ``tmp_path`` fixture
    when the test session ends.
    """
    jsonl_file: Path = tmp_path / "mock_cloudtrail.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as fh:
        for record in MOCK_RECORDS:
            fh.write(json.dumps(record) + "\n")
    return jsonl_file


@pytest.fixture()
def flat_df(mock_jsonl: Path) -> pd.DataFrame:
    """Return a flattened, time-sorted, null-filled DataFrame for reuse."""
    total: int = count_lines(mock_jsonl)
    df: pd.DataFrame = flatten_records(mock_jsonl, total)
    df = convert_and_sort_by_time(df)
    df = fill_missing_strings(df)
    return df


# ======================================================================== #
#  Stage 1 — stream_jsonl / count_lines                                     #
# ======================================================================== #

class TestStreamJsonl:
    """Tests for the streaming JSONL generator and line counter."""

    def test_yields_correct_count(self, mock_jsonl: Path) -> None:
        """All four lines should be yielded as four dicts."""
        records: list[dict[str, Any]] = list(stream_jsonl(mock_jsonl))
        assert len(records) == 4

    def test_preserves_nested_structure(self, mock_jsonl: Path) -> None:
        """Nested dicts like userIdentity must survive the round-trip."""
        records = list(stream_jsonl(mock_jsonl))
        first_type: str = records[0]["userIdentity"]["type"]
        assert first_type in {"Root", "IAMUser"}

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A missing file must raise FileNotFoundError, not silently return."""
        with pytest.raises(FileNotFoundError):
            list(stream_jsonl(tmp_path / "nonexistent.jsonl"))

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank or whitespace-only lines must be silently skipped."""
        jsonl_file: Path = tmp_path / "blanks.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(EVENT_A) + "\n")
            fh.write("\n")
            fh.write("   \n")
            fh.write(json.dumps(EVENT_B) + "\n")
        assert len(list(stream_jsonl(jsonl_file))) == 2

    def test_count_lines_matches_yield_count(self, mock_jsonl: Path) -> None:
        """count_lines must agree with the number of records yielded."""
        assert count_lines(mock_jsonl) == len(list(stream_jsonl(mock_jsonl)))


# ======================================================================== #
#  Stage 2 — flatten_records                                                #
# ======================================================================== #

class TestFlattenRecords:
    """Tests for pd.json_normalize flattening."""

    def test_row_count(self, mock_jsonl: Path) -> None:
        """One row per input record."""
        df: pd.DataFrame = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        assert len(df) == 4

    def test_nested_keys_become_dot_separated(self, mock_jsonl: Path) -> None:
        """userIdentity.type must exist as a flattened column."""
        df = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        assert "userIdentity.type" in df.columns
        assert "userIdentity.arn" in df.columns

    def test_original_nested_dict_column_absent(self, mock_jsonl: Path) -> None:
        """The raw 'userIdentity' dict column must not survive flattening."""
        df = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        assert "userIdentity" not in df.columns


# ======================================================================== #
#  Stage 3 — convert_and_sort_by_time                                       #
# ======================================================================== #

class TestConvertAndSort:
    """Tests for datetime conversion and chronological ordering."""

    def test_eventtime_is_datetime(self, mock_jsonl: Path) -> None:
        """eventTime must be converted to a datetime64 dtype."""
        df = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        df = convert_and_sort_by_time(df)
        assert pd.api.types.is_datetime64_any_dtype(df["eventTime"])

    def test_chronological_order(self, mock_jsonl: Path) -> None:
        """Rows must be sorted earliest-first: D(18h), C(19h), A(20h), B(21h)."""
        df = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        df = convert_and_sort_by_time(df)
        event_ids: list[str] = df["eventID"].tolist()
        assert event_ids == ["dddd-dddd", "cccc-cccc", "aaaa-aaaa", "bbbb-bbbb"]

    def test_index_is_reset(self, mock_jsonl: Path) -> None:
        """Index must be a clean 0..n-1 range after sorting."""
        df = flatten_records(mock_jsonl, count_lines(mock_jsonl))
        df = convert_and_sort_by_time(df)
        assert list(df.index) == [0, 1, 2, 3]


# ======================================================================== #
#  Stage 4 — fill_missing_strings                                           #
# ======================================================================== #

class TestFillMissingStrings:
    """Tests for null handling in string columns."""

    def test_no_nans_in_object_columns(self, flat_df: pd.DataFrame) -> None:
        """After filling, no object column should contain NaN."""
        object_cols: pd.Index = flat_df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            assert flat_df[col].isna().sum() == 0, f"NaN found in column: {col}"

    def test_missing_username_filled(self, flat_df: pd.DataFrame) -> None:
        """Root events lack userName; those cells must now read 'None'."""
        root_rows: pd.DataFrame = flat_df[flat_df["userIdentity.type"] == "Root"]
        assert (root_rows["userIdentity.userName"] == "None").all()


# ======================================================================== #
#  Stage 5 — split_train_test                                               #
# ======================================================================== #

class TestSplitTrainTest:
    """Tests for the train/test splitting logic."""

    def test_train_contains_only_root(self, flat_df: pd.DataFrame) -> None:
        """train_df must contain *only* Root-type events (A and D)."""
        train_df, _ = split_train_test(flat_df)
        assert (train_df["userIdentity.type"] == "Root").all()

    def test_train_has_events_a_and_d(self, flat_df: pd.DataFrame) -> None:
        """Before noise filtering, train_df should hold events A and D."""
        train_df, _ = split_train_test(flat_df)
        assert set(train_df["eventID"].tolist()) == {"aaaa-aaaa", "dddd-dddd"}

    def test_test_contains_level6_and_backup(self, flat_df: pd.DataFrame) -> None:
        """test_df must contain events B (Level6) and C (backup)."""
        _, test_df = split_train_test(flat_df)
        assert set(test_df["eventID"].tolist()) == {"bbbb-bbbb", "cccc-cccc"}

    def test_test_usernames(self, flat_df: pd.DataFrame) -> None:
        """test_df userNames must be exactly {'Level6', 'backup'}."""
        _, test_df = split_train_test(flat_df)
        assert set(test_df["userIdentity.userName"].tolist()) == {"Level6", "backup"}

    def test_splits_are_independent_copies(self, flat_df: pd.DataFrame) -> None:
        """Mutating train_df must not affect test_df or vice-versa."""
        train_df, test_df = split_train_test(flat_df)
        train_df["eventName"] = "MUTATED"
        assert "MUTATED" not in test_df["eventName"].values


# ======================================================================== #
#  Stage 6 — filter_noise                                                   #
# ======================================================================== #

class TestFilterNoise:
    """Tests for automated-noise removal."""

    def test_cloudmapper_dropped_from_train(self, flat_df: pd.DataFrame) -> None:
        """Event D (CloudMapper) must be removed from train_df."""
        train_df, _ = split_train_test(flat_df)
        train_df = filter_noise(train_df)
        assert "dddd-dddd" not in train_df["eventID"].values

    def test_train_retains_only_event_a(self, flat_df: pd.DataFrame) -> None:
        """After noise filtering, train_df must contain only Event A."""
        train_df, _ = split_train_test(flat_df)
        train_df = filter_noise(train_df)
        assert len(train_df) == 1
        assert train_df["eventID"].iloc[0] == "aaaa-aaaa"

    def test_test_unaffected_by_noise_filter(self, flat_df: pd.DataFrame) -> None:
        """Events B and C have clean userAgents; test_df must keep both."""
        _, test_df = split_train_test(flat_df)
        test_df = filter_noise(test_df)
        assert len(test_df) == 2
        assert set(test_df["eventID"].tolist()) == {"bbbb-bbbb", "cccc-cccc"}

    def test_event_d_absent_everywhere(self, flat_df: pd.DataFrame) -> None:
        """Event D must not appear in train_df OR test_df after filtering."""
        train_df, test_df = split_train_test(flat_df)
        train_df = filter_noise(train_df)
        test_df = filter_noise(test_df)
        all_ids: set[str] = set(train_df["eventID"].tolist() + test_df["eventID"].tolist())
        assert "dddd-dddd" not in all_ids

    def test_boto3_noise_dropped(self) -> None:
        """A row with 'boto3' in userAgent must also be dropped."""
        noisy: dict[str, Any] = {**EVENT_A, "userAgent": "boto3/1.26.0", "eventID": "noisy"}
        df = _flatten_list([EVENT_A, noisy])
        df = convert_and_sort_by_time(df)
        df = fill_missing_strings(df)
        df = filter_noise(df)
        assert "noisy" not in df["eventID"].values
        assert len(df) == 1

    def test_index_reset_after_filter(self, flat_df: pd.DataFrame) -> None:
        """Index must be contiguous 0..n-1 after rows are dropped."""
        train_df, _ = split_train_test(flat_df)
        train_df = filter_noise(train_df)
        assert list(train_df.index) == list(range(len(train_df)))


# ======================================================================== #
#  Stage 7 — assign_sessions                                                #
# ======================================================================== #

class TestAssignSessions:
    """Tests for chronological session grouping."""

    def test_session_columns_added(self, flat_df: pd.DataFrame) -> None:
        """sessionId and sessionSeqNo columns must be present."""
        _, test_df = split_train_test(flat_df)
        test_df = filter_noise(test_df)
        test_df = assign_sessions(test_df)
        assert "sessionId" in test_df.columns
        assert "sessionSeqNo" in test_df.columns

    def test_distinct_ips_get_distinct_sessions(self, flat_df: pd.DataFrame) -> None:
        """Events B and C have different IPs and ARNs, so two sessions."""
        _, test_df = split_train_test(flat_df)
        test_df = filter_noise(test_df)
        test_df = assign_sessions(test_df)
        assert test_df["sessionId"].nunique() == 2

    def test_same_ip_and_arn_share_session(self) -> None:
        """Two events from the same IP+ARN must share one sessionId."""
        twin: dict[str, Any] = {
            **EVENT_A,
            "eventTime": "2017-02-12T20:05:00Z",
            "eventID": "aaaa-twin",
        }
        df = _flatten_list([EVENT_A, twin])
        df = convert_and_sort_by_time(df)
        df = fill_missing_strings(df)
        df = assign_sessions(df)
        assert df["sessionId"].nunique() == 1
        assert df["sessionSeqNo"].tolist() == [0, 1]

    def test_session_seq_starts_at_zero(self, flat_df: pd.DataFrame) -> None:
        """Every session's sequence numbering must begin at 0."""
        train_df, _ = split_train_test(flat_df)
        train_df = filter_noise(train_df)
        train_df = assign_sessions(train_df)
        for _, group in train_df.groupby("sessionId"):
            assert group["sessionSeqNo"].iloc[0] == 0


# ======================================================================== #
#  I/O — save_csv                                                           #
# ======================================================================== #

class TestSaveCsv:
    """Tests for CSV persistence."""

    def test_csv_written_to_disk(self, flat_df: pd.DataFrame, tmp_path: Path) -> None:
        """Output CSV file must exist after save_csv."""
        out: Path = tmp_path / "sub" / "output.csv"
        save_csv(flat_df, out)
        assert out.exists()

    def test_csv_roundtrip_row_count(self, flat_df: pd.DataFrame, tmp_path: Path) -> None:
        """Reading the CSV back must yield the same number of rows."""
        out: Path = tmp_path / "output.csv"
        save_csv(flat_df, out)
        reloaded: pd.DataFrame = pd.read_csv(out)
        assert len(reloaded) == len(flat_df)


# ======================================================================== #
#  End-to-end integration                                                   #
# ======================================================================== #

class TestEndToEnd:
    """Full pipeline integration: load → flatten → sort → fill → split → filter → sessions."""

    def test_full_pipeline(self, mock_jsonl: Path, tmp_path: Path) -> None:
        """Run every stage in sequence and verify final train/test content."""
        # Stage 1 — count & stream
        total: int = count_lines(mock_jsonl)
        assert total == 4

        # Stage 2 — Flatten
        df: pd.DataFrame = flatten_records(mock_jsonl, total)
        assert "userIdentity.type" in df.columns

        # Stage 3 — Sort
        df = convert_and_sort_by_time(df)
        assert df["eventID"].iloc[0] == "dddd-dddd"  # earliest

        # Stage 4 — Nulls
        df = fill_missing_strings(df)

        # Stage 5 — Split
        train_df, test_df = split_train_test(df)

        # Stage 6 — Noise
        train_df = filter_noise(train_df)
        test_df = filter_noise(test_df)

        # Stage 7 — Sessions
        train_df = assign_sessions(train_df)
        test_df = assign_sessions(test_df)

        # ---- Final assertions ----

        # Train must contain ONLY Event A
        assert len(train_df) == 1
        assert train_df["eventID"].iloc[0] == "aaaa-aaaa"
        assert train_df["eventName"].iloc[0] == "ListBuckets"

        # Test must contain ONLY Events B and C
        assert len(test_df) == 2
        assert set(test_df["eventID"].tolist()) == {"bbbb-bbbb", "cccc-cccc"}
        assert set(test_df["userIdentity.userName"].tolist()) == {"Level6", "backup"}

        # Event D must be completely absent
        all_ids: set[str] = set(
            train_df["eventID"].tolist() + test_df["eventID"].tolist()
        )
        assert "dddd-dddd" not in all_ids

        # Persist and verify files exist
        save_csv(train_df, tmp_path / "train.csv")
        save_csv(test_df, tmp_path / "test.csv")
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()
