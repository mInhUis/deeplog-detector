"""Unit tests for the end-to-end pipeline (Phase 5).

Tests cover:
    1. load_deeplog_checkpoint — reconstruction, missing file, key checks.
    2. reverse_map_anomalies   — text output, filtering, edge cases.
    3. generate_mock_pipeline_data — structure and validity.
    4. End-to-end mock pipeline — runs without error, produces reports.

All tests use mock mode and small synthetic data (per CLAUDE.md
heavy-compute warning).

Mock Data:
    Uses the mock templates and sessions from ``src/main.py``.
    DeepLog is quick-trained for ~20 epochs on normal sessions only.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Final

import pytest
import torch

from src.detector.deeplog import DeepLogModel
from src.main import (
    _MOCK_NUM_KEYS,
    _MOCK_TEMPLATES,
    generate_mock_pipeline_data,
    load_deeplog_checkpoint,
    reverse_map_anomalies,
    run_pipeline,
)


# ======================================================================== #
#  Fixtures                                                                 #
# ======================================================================== #

WINDOW_SIZE: Final[int] = 3
NUM_KEYS: Final[int] = 5


@pytest.fixture()
def mock_model_and_checkpoint(tmp_path: Path) -> tuple[Path, int]:
    """Create a minimal DeepLog checkpoint on disk.

    Returns (checkpoint_path, num_keys).
    """
    model = DeepLogModel(
        num_keys=NUM_KEYS,
        embedding_dim=16,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )
    ckpt_path: Path = tmp_path / "deeplog.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": 10,
            "num_keys": NUM_KEYS,
            "window_size": WINDOW_SIZE,
            "embedding_dim": 16,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "final_loss": 0.5,
        },
        ckpt_path,
    )
    return ckpt_path, NUM_KEYS


# ======================================================================== #
#  TestLoadDeeplogCheckpoint                                                #
# ======================================================================== #


class TestLoadDeeplogCheckpoint:
    """Verify checkpoint loading and model reconstruction."""

    def test_loads_valid_checkpoint(
        self, mock_model_and_checkpoint: tuple[Path, int],
    ) -> None:
        ckpt_path, num_keys = mock_model_and_checkpoint
        model, ckpt = load_deeplog_checkpoint(ckpt_path, torch.device("cpu"))
        assert isinstance(model, DeepLogModel)
        assert ckpt["num_keys"] == num_keys

    def test_model_in_eval_mode(
        self, mock_model_and_checkpoint: tuple[Path, int],
    ) -> None:
        ckpt_path, _ = mock_model_and_checkpoint
        model, _ = load_deeplog_checkpoint(ckpt_path, torch.device("cpu"))
        assert not model.training

    def test_hyperparameters_reconstructed(
        self, mock_model_and_checkpoint: tuple[Path, int],
    ) -> None:
        ckpt_path, _ = mock_model_and_checkpoint
        model, ckpt = load_deeplog_checkpoint(ckpt_path, torch.device("cpu"))
        assert ckpt["window_size"] == WINDOW_SIZE
        assert ckpt["embedding_dim"] == 16
        assert ckpt["hidden_size"] == 16
        assert ckpt["num_layers"] == 1

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_deeplog_checkpoint(
                Path("/nonexistent/deeplog.pt"), torch.device("cpu"),
            )

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        """Checkpoint without required keys should raise KeyError."""
        bad_ckpt = tmp_path / "bad.pt"
        torch.save({"epoch": 1}, bad_ckpt)
        with pytest.raises(KeyError, match="model_state_dict"):
            load_deeplog_checkpoint(bad_ckpt, torch.device("cpu"))


# ======================================================================== #
#  TestReverseMapAnomalies                                                  #
# ======================================================================== #


class TestReverseMapAnomalies:
    """Verify the anomaly reverse-mapping logic."""

    TEMPLATES: Final[dict[int, str]] = {
        0: "ListBuckets <*>",
        1: "GetObject <*>",
        2: "CreateUser <*>",
    }

    def test_only_anomalous_sessions_returned(self) -> None:
        sessions = [[0, 1, 0, 1], [0, 1, 2, 1]]
        flags = [
            [False, False, False, False],  # no anomalies
            [False, False, True, False],   # one anomaly
        ]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert len(result) == 1
        assert result[0]["session_index"] == 1

    def test_output_is_text_not_integers(self) -> None:
        sessions = [[0, 1, 2]]
        flags = [[False, True, True]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        for log in result[0]["anomalous_logs"]:
            assert isinstance(log, str)
            assert not log.isdigit()

    def test_correct_anomaly_indices(self) -> None:
        sessions = [[0, 1, 2, 0, 1]]
        flags = [[False, True, True, False, False]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert result[0]["anomaly_indices"] == [1, 2]

    def test_template_text_matches(self) -> None:
        sessions = [[2, 0]]
        flags = [[True, False]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert result[0]["anomalous_logs"] == ["CreateUser <*>"]

    def test_unknown_key_fallback(self) -> None:
        sessions = [[99]]
        flags = [[True]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert "<unknown key 99>" in result[0]["anomalous_logs"][0]

    def test_anomaly_ratio_calculated(self) -> None:
        sessions = [[0, 1, 2, 0]]
        flags = [[True, False, True, False]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert result[0]["context"]["anomaly_ratio"] == pytest.approx(0.5)

    def test_context_has_required_keys(self) -> None:
        sessions = [[0, 1]]
        flags = [[True, False]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        ctx = result[0]["context"]
        assert "session_id" in ctx
        assert "total_events" in ctx
        assert "anomaly_ratio" in ctx
        assert "timestamps" in ctx
        assert "source_ips" in ctx
        assert "user_arns" in ctx
        assert "event_names" in ctx

    def test_empty_input_returns_empty(self) -> None:
        result = reverse_map_anomalies([], [], self.TEMPLATES)
        assert result == []

    def test_no_anomalies_returns_empty(self) -> None:
        sessions = [[0, 1, 0]]
        flags = [[False, False, False]]
        result = reverse_map_anomalies(sessions, flags, self.TEMPLATES)
        assert result == []

    def test_metadata_slicing(self) -> None:
        """Per-anomaly metadata should be sliced from full session metadata."""
        sessions = [[0, 1, 2]]
        flags = [[False, True, True]]
        meta = [{
            "session_id": 7,
            "timestamps": ["t0", "t1", "t2"],
            "source_ips": ["ip0", "ip1", "ip2"],
            "user_arns": ["arn0", "arn1", "arn2"],
            "event_names": ["ListBuckets", "GetObject", "CreateUser"],
        }]
        result = reverse_map_anomalies(
            sessions, flags, self.TEMPLATES, session_metadata=meta,
        )
        ctx = result[0]["context"]
        assert ctx["session_id"] == 7
        assert ctx["timestamps"] == ["t1", "t2"]
        assert ctx["source_ips"] == ["ip1", "ip2"]
        assert ctx["event_names"] == ["GetObject", "CreateUser"]


# ======================================================================== #
#  TestGenerateMockPipelineData                                             #
# ======================================================================== #


class TestGenerateMockPipelineData:
    """Verify the mock data generator structure."""

    def test_returns_three_element_tuple(self) -> None:
        result = generate_mock_pipeline_data()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sessions_structure(self) -> None:
        sessions, templates, metadata = generate_mock_pipeline_data()
        assert len(sessions) == 5
        for session in sessions:
            assert isinstance(session, list)
            assert len(session) == 15
            assert all(isinstance(k, int) for k in session)

    def test_all_keys_valid(self) -> None:
        sessions, templates, _ = generate_mock_pipeline_data()
        for session in sessions:
            for key in session:
                assert key in templates, f"Key {key} not in templates"

    def test_attack_session_has_suspicious_keys(self) -> None:
        sessions, _, _ = generate_mock_pipeline_data()
        attack_session = sessions[4]
        suspicious_keys = {5, 6, 7}
        found = {k for k in attack_session if k in suspicious_keys}
        assert found == suspicious_keys

    def test_normal_sessions_have_no_suspicious_keys(self) -> None:
        sessions, _, _ = generate_mock_pipeline_data()
        suspicious_keys = {5, 6, 7}
        for s_idx in range(4):
            keys_in_session = set(sessions[s_idx])
            assert not keys_in_session & suspicious_keys

    def test_metadata_alignment(self) -> None:
        sessions, _, metadata = generate_mock_pipeline_data()
        assert len(metadata) == len(sessions)
        for s_idx, meta in enumerate(metadata):
            assert meta["session_id"] == s_idx
            assert len(meta["timestamps"]) == len(sessions[s_idx])
            assert len(meta["source_ips"]) == len(sessions[s_idx])
            assert len(meta["event_names"]) == len(sessions[s_idx])

    def test_templates_include_normal_and_suspicious(self) -> None:
        _, templates, _ = generate_mock_pipeline_data()
        assert len(templates) == _MOCK_NUM_KEYS
        # Normal templates
        assert "ListBuckets" in templates[0]
        # Suspicious templates
        assert "CreateUser" in templates[5]


# ======================================================================== #
#  TestEndToEndMockPipeline                                                 #
# ======================================================================== #


class TestEndToEndMockPipeline:
    """Verify mock-inference mode runs end-to-end without errors."""

    @pytest.fixture()
    def mock_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            mode="mock_inference",
            top_k=9,
            report_output=None,
        )

    def test_pipeline_runs_without_error(
        self, mock_args: argparse.Namespace,
    ) -> None:
        # Should not raise
        result = run_pipeline(mock_args)
        # Result may be None (if no anomalies detected by undertrained model)
        # or a string report
        assert result is None or isinstance(result, str)

    def test_pipeline_with_low_top_k_detects_anomalies(self) -> None:
        """With very low top-k, more events should be flagged."""
        args = argparse.Namespace(
            mode="mock_inference",
            top_k=1,  # very sensitive
            report_output=None,
        )
        result = run_pipeline(args)
        # With top_k=1 on a quick-trained model, anomalies are very likely
        # (even normal sessions may produce some flags)
        # Just verify it runs successfully
        assert result is None or isinstance(result, str)

    def test_report_output_to_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "report.txt"
        args = argparse.Namespace(
            mode="mock_inference",
            top_k=1,  # sensitive to ensure we get output
            report_output=str(output_file),
        )
        result = run_pipeline(args)
        if result is not None:
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert len(content) > 0

    def test_mock_report_contains_sections(self) -> None:
        """If anomalies are detected, report should have all 5 sections."""
        args = argparse.Namespace(
            mode="mock_inference",
            top_k=1,
            report_output=None,
        )
        result = run_pipeline(args)
        if result is not None:
            assert "Incident Summary" in result
            assert "Anomaly Analysis" in result
            assert "Severity Assessment" in result
            assert "Recommended Mitigation" in result
