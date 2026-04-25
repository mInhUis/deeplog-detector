"""Unit tests for the Llama-3 incident-response responder (Phase 4).

Tests cover:
    1. ResponderConfig   — defaults, immutability, custom values.
    2. LlamaResponder    — mock-mode report generation, section structure,
                           input reflection, edge cases.
    3. Prompt building   — Llama-3 chat-format tokens, system/user content.
    4. Dependency guard  — RuntimeError when unsloth is missing.

Uses mock mode exclusively (per CLAUDE.md heavy-compute warning).
No GPU memory is allocated during any test.

Mock Data:
    3 anomalous log templates simulating a privilege-escalation attack.
    Context dict with session metadata (timestamps, IPs, ARNs).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final
from unittest.mock import patch

import pytest

from src.responder.llama_inference import (
    LlamaResponder,
    ResponderConfig,
    _build_prompt,
    _build_user_message,
    _generate_mock_report,
)


# ======================================================================== #
#  Mock data                                                                #
# ======================================================================== #

MOCK_ANOMALOUS_LOGS: Final[list[str]] = [
    "CreateUser <*>",
    "AttachUserPolicy <*> <*>",
    "PutBucketPolicy <*>",
]

MOCK_CONTEXT: Final[dict[str, Any]] = {
    "session_id": 42,
    "total_events": 15,
    "anomaly_ratio": 0.2,
    "timestamps": [
        "2017-06-05T12:00:00Z",
        "2017-06-05T12:01:00Z",
        "2017-06-05T12:02:00Z",
    ],
    "source_ips": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
    "user_arns": [
        "arn:aws:iam::123456:user/attacker",
        "arn:aws:iam::123456:user/attacker",
        "arn:aws:iam::123456:user/attacker",
    ],
    "event_names": ["CreateUser", "AttachUserPolicy", "PutBucketPolicy"],
}


# ======================================================================== #
#  TestResponderConfig                                                      #
# ======================================================================== #


class TestResponderConfig:
    """Verify default values, immutability, and custom overrides."""

    def test_defaults(self) -> None:
        cfg = ResponderConfig()
        assert cfg.model_name == "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        assert cfg.max_new_tokens == 1024
        assert cfg.temperature == pytest.approx(0.3)
        assert cfg.top_p == pytest.approx(0.9)
        assert cfg.repetition_penalty == pytest.approx(1.15)
        assert cfg.max_seq_length == 2048
        assert cfg.mock_mode is False

    def test_frozen_immutability(self) -> None:
        cfg = ResponderConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.temperature = 0.9  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = ResponderConfig(
            model_name="custom/model",
            max_new_tokens=512,
            temperature=0.7,
            mock_mode=True,
        )
        assert cfg.model_name == "custom/model"
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == pytest.approx(0.7)
        assert cfg.mock_mode is True


# ======================================================================== #
#  TestLlamaResponderMock                                                   #
# ======================================================================== #


class TestLlamaResponderMock:
    """Test mock-mode report generation (no GPU, no unsloth)."""

    @pytest.fixture()
    def responder(self) -> LlamaResponder:
        return LlamaResponder(ResponderConfig(mock_mode=True))

    def test_report_is_string(self, responder: LlamaResponder) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_has_all_five_sections(
        self, responder: LlamaResponder,
    ) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "## 1. Incident Summary" in report
        assert "## 2. Anomaly Analysis" in report
        assert "## 3. Severity Assessment" in report
        assert "## 4. Attack Pattern Classification" in report
        assert "## 5. Recommended Mitigation" in report

    def test_report_reflects_session_id(
        self, responder: LlamaResponder,
    ) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "42" in report

    def test_report_reflects_event_count(
        self, responder: LlamaResponder,
    ) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "3" in report  # 3 anomalous events

    def test_report_contains_mock_marker(
        self, responder: LlamaResponder,
    ) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "[MOCK MODE" in report

    def test_model_stays_none(self, responder: LlamaResponder) -> None:
        responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert responder._model is None
        assert responder._tokenizer is None
        assert responder._loaded is True

    def test_report_contains_event_names(
        self, responder: LlamaResponder,
    ) -> None:
        report = responder.generate_report(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "CreateUser" in report
        assert "AttachUserPolicy" in report

    def test_empty_logs(self, responder: LlamaResponder) -> None:
        report = responder.generate_report([], {"session_id": 0})
        assert isinstance(report, str)
        assert "## 1. Incident Summary" in report
        assert "0 anomalous event" in report

    def test_missing_context_keys(self, responder: LlamaResponder) -> None:
        """Context with no keys should not raise; defaults are used."""
        report = responder.generate_report(
            ["SomeEvent <*>"], {},
        )
        assert isinstance(report, str)
        assert len(report) > 0


# ======================================================================== #
#  TestMockReportSeverity                                                   #
# ======================================================================== #


class TestMockReportSeverity:
    """Verify severity heuristic thresholds."""

    def test_high_severity(self) -> None:
        report = _generate_mock_report(
            ["Event1"], {"anomaly_ratio": 0.5},
        )
        assert "**High**" in report

    def test_medium_severity(self) -> None:
        report = _generate_mock_report(
            ["Event1"], {"anomaly_ratio": 0.2},
        )
        assert "**Medium**" in report

    def test_low_severity(self) -> None:
        report = _generate_mock_report(
            ["Event1"], {"anomaly_ratio": 0.05},
        )
        assert "**Low**" in report


# ======================================================================== #
#  TestPromptBuilding                                                       #
# ======================================================================== #


class TestPromptBuilding:
    """Verify Llama-3 Instruct chat-format prompt structure."""

    def test_system_prompt_contains_analyst_role(self) -> None:
        prompt = _build_prompt(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "Cloud Security Analyst" in prompt

    def test_user_message_contains_event_data(self) -> None:
        msg = _build_user_message(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "CreateUser <*>" in msg
        assert "10.0.0.1" in msg
        assert "42" in msg  # session_id

    def test_prompt_has_llama3_chat_tokens(self) -> None:
        prompt = _build_prompt(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
        assert "<|eot_id|>" in prompt

    def test_prompt_includes_all_anomalous_logs(self) -> None:
        prompt = _build_prompt(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        for log in MOCK_ANOMALOUS_LOGS:
            assert log in prompt

    def test_user_message_anomaly_ratio(self) -> None:
        msg = _build_user_message(MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT)
        assert "20.0%" in msg  # 0.2 formatted as percentage


# ======================================================================== #
#  TestDependencyHandling                                                   #
# ======================================================================== #


class TestDependencyHandling:
    """Verify graceful handling of missing unsloth dependency."""

    def test_real_mode_raises_without_unsloth(self) -> None:
        """Non-mock responder should raise RuntimeError on _load_model."""
        responder = LlamaResponder(ResponderConfig(mock_mode=False))
        with patch.dict("sys.modules", {"unsloth": None}):
            with pytest.raises(RuntimeError, match="Unsloth is required"):
                responder._load_model()

    def test_mock_mode_bypasses_unsloth(self) -> None:
        """Mock mode should never attempt to import unsloth."""
        responder = LlamaResponder(ResponderConfig(mock_mode=True))
        with patch.dict("sys.modules", {"unsloth": None}):
            # Should not raise
            report = responder.generate_report(
                MOCK_ANOMALOUS_LOGS, MOCK_CONTEXT,
            )
            assert isinstance(report, str)
