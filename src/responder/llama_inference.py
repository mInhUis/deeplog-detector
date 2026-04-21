"""Llama-3 incident-response report generator (Phase 4).

**Why an LLM after DeepLog? (Advisor Skill 2)**
DeepLog's LSTM detects *that* an anomaly occurred — a mathematical deviation
in the log-key sequence.  It cannot explain *what* the anomaly means in a
security context or *how* to remediate it.  Llama-3-8b bridges this semantic
gap: given the human-readable Drain templates of flagged events, it produces
a structured incident report with severity assessment, attack classification,
and actionable mitigation steps.

Data Flow:
    Anomaly flags (list[list[bool]])
        → reverse-map keys to Drain templates (in main.py)
        → anomalous_logs: list[str]  +  context: dict
        → LlamaResponder.generate_report()
        → Structured incident report (str)

Tensor shapes (real-mode inference):
    input_ids  : (1, seq_len) int64   — tokenised prompt
    logits     : (1, seq_len, vocab_size) float32
    output_ids : (1, seq_len + max_new_tokens) int64

Heavy Compute Warning (CLAUDE.md):
    Real Llama-3 inference requires a CUDA GPU with ≥6 GB VRAM.
    Default to ``mock_mode=True`` for interactive development and testing.
    The mock path never imports unsloth or allocates GPU memory.

Environment Variables:
    RESPONDER_MODEL     – HuggingFace model ID
                          (default: unsloth/llama-3-8b-Instruct-bnb-4bit)
    RESPONDER_MOCK      – "true" to force mock mode (default: "false")
    RESPONDER_MAX_TOKENS – Max new tokens to generate (default: 1024)
    CUDA_MEMORY_MAX     – Max VRAM fraction (0.0–1.0, default: 0.5)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponderConfig:
    """Immutable configuration for the Llama-3 responder.

    Attributes:
        model_name:         HuggingFace model ID for the 4-bit quantised
                            Llama-3-8b Instruct model.
        max_new_tokens:     Maximum tokens the model may generate per report.
        temperature:        Sampling temperature.  0.3 keeps output
                            deterministic enough for reproducible thesis
                            results while avoiding degenerate repetition.
        top_p:              Nucleus-sampling probability mass.
        repetition_penalty: Penalises repeated token sequences.
        max_seq_length:     Maximum context length.  2048 (not the default
                            4096) to reduce KV-cache VRAM usage on consumer
                            GPUs.
        mock_mode:          When True, generate_report() returns a template-
                            based mock without loading the model.  Safe for
                            interactive sessions and CI.
    """

    model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_seq_length: int = 2048
    mock_mode: bool = False


# ---------------------------------------------------------------------------
#  Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are a senior AWS Cloud Security Analyst performing autonomous "
    "incident response.  A DeepLog LSTM anomaly-detection model has flagged "
    "the following CloudTrail events as statistically anomalous — their "
    "log-key transitions deviate from the learned normal baseline.\n\n"
    "Analyse the flagged events and produce a structured incident report "
    "with EXACTLY the following five sections:\n\n"
    "1. **Incident Summary** — 1–2 sentences describing the overall incident.\n"
    "2. **Anomaly Analysis** — For each flagged event, explain why it is "
    "suspicious in the context of AWS security.\n"
    "3. **Severity Assessment** — Rate as Critical / High / Medium / Low "
    "with a brief justification.\n"
    "4. **Attack Pattern Classification** — Map to MITRE ATT&CK tactics "
    "and techniques where applicable.\n"
    "5. **Recommended Mitigation** — Numbered, actionable remediation steps "
    "an incident responder should execute immediately.\n\n"
    "Be concise, precise, and actionable.  Do not speculate beyond what the "
    "evidence supports."
)


def _build_user_message(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Format the user-turn content from reverse-mapped anomaly data.

    Args:
        anomalous_logs: Drain template strings for each flagged event.
        context:        Metadata dict with optional keys: ``session_id``,
                        ``total_events``, ``anomaly_ratio``, ``timestamps``,
                        ``source_ips``, ``user_arns``, ``event_names``.

    Returns:
        A formatted string ready to be placed in the Llama-3 user turn.
    """
    session_id: int | str = context.get("session_id", "N/A")
    total: int = context.get("total_events", len(anomalous_logs))
    ratio: float = context.get("anomaly_ratio", 1.0)
    timestamps: list[str] = context.get("timestamps", [])
    source_ips: list[str] = context.get("source_ips", [])
    user_arns: list[str] = context.get("user_arns", [])
    event_names: list[str] = context.get("event_names", [])

    header: str = (
        f"Session ID: {session_id}\n"
        f"Total events in session: {total} | "
        f"Anomalous: {len(anomalous_logs)} ({ratio:.1%})\n\n"
        "Flagged events:\n"
    )

    lines: list[str] = []
    for idx, log_text in enumerate(anomalous_logs):
        ts: str = timestamps[idx] if idx < len(timestamps) else "N/A"
        ip: str = source_ips[idx] if idx < len(source_ips) else "N/A"
        arn: str = user_arns[idx] if idx < len(user_arns) else "N/A"
        evt: str = event_names[idx] if idx < len(event_names) else "N/A"
        lines.append(
            f"  {idx + 1}. [{ts}] {evt}\n"
            f"     Template: {log_text}\n"
            f"     Source IP: {ip} | Identity: {arn}"
        )

    return header + "\n".join(lines)


def _build_prompt(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Construct the full Llama-3 Instruct chat-format prompt.

    Uses the ``<|begin_of_text|>`` / ``<|start_header_id|>`` / ``<|eot_id|>``
    special-token format required by the Llama-3 Instruct family.

    Returns:
        Ready-to-tokenise prompt string.
    """
    user_msg: str = _build_user_message(anomalous_logs, context)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{_SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ---------------------------------------------------------------------------
#  Mock report generator
# ---------------------------------------------------------------------------

def _generate_mock_report(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Produce a template-based mock report with the same 5-section structure.

    The mock uses heuristics (anomaly ratio, event names) to fill in
    plausible content so the full pipeline produces visually meaningful
    output during development and thesis demos.
    """
    session_id: int | str = context.get("session_id", "N/A")
    ratio: float = context.get("anomaly_ratio", 0.0)
    event_names: list[str] = context.get("event_names", [])
    timestamps: list[str] = context.get("timestamps", [])
    source_ips: list[str] = context.get("source_ips", [])
    user_arns: list[str] = context.get("user_arns", [])

    # --- Severity heuristic ---
    if ratio > 0.3:
        severity = "High"
        severity_reason = (
            f"{ratio:.0%} of session events are anomalous, indicating a "
            "sustained deviation from normal behaviour."
        )
    elif ratio > 0.15:
        severity = "Medium"
        severity_reason = (
            f"{ratio:.0%} anomaly ratio suggests intermittent suspicious "
            "activity within an otherwise normal session."
        )
    else:
        severity = "Low"
        severity_reason = (
            f"Only {ratio:.0%} of events are anomalous; this may be benign "
            "but warrants monitoring."
        )

    # --- Attack pattern heuristic ---
    name_set: set[str] = {n.lower() for n in event_names}
    if {"createuser", "attachuserpolicy"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1136 (Create Account), T1098 (Account "
            "Manipulation) — Privilege Escalation / Persistence."
        )
    elif {"putbucketpolicy", "putobject"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1530 (Data from Cloud Storage Object) — "
            "Data Exfiltration / Resource Manipulation."
        )
    elif {"assumerole", "getcalleridentity"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1550.001 (Use Alternate Authentication "
            "Material: Application Access Token) — Lateral Movement."
        )
    else:
        attack_pattern = (
            "No direct MITRE ATT&CK mapping identified from event names "
            "alone.  Manual review recommended."
        )

    # --- Build event lines ---
    event_lines: list[str] = []
    for i, log in enumerate(anomalous_logs):
        ts = timestamps[i] if i < len(timestamps) else "N/A"
        ip = source_ips[i] if i < len(source_ips) else "N/A"
        arn = user_arns[i] if i < len(user_arns) else "N/A"
        evt = event_names[i] if i < len(event_names) else "N/A"
        event_lines.append(
            f"  - [{ts}] **{evt}** from {ip} (identity: {arn})\n"
            f"    Template: `{log}`\n"
            f"    This event deviates from the learned normal sequence "
            f"pattern and was not predicted by the DeepLog model."
        )

    events_block: str = "\n".join(event_lines) if event_lines else (
        "  No detailed event data available."
    )

    return (
        f"## 1. Incident Summary\n\n"
        f"DeepLog flagged {len(anomalous_logs)} anomalous event(s) in "
        f"session {session_id}.  The flagged actions deviate from the "
        f"normal CloudTrail activity baseline learned during training.\n\n"
        f"## 2. Anomaly Analysis\n\n"
        f"{events_block}\n\n"
        f"## 3. Severity Assessment\n\n"
        f"**{severity}** — {severity_reason}\n\n"
        f"## 4. Attack Pattern Classification\n\n"
        f"{attack_pattern}\n\n"
        f"## 5. Recommended Mitigation\n\n"
        f"1. Immediately review the IAM activity for the affected "
        f"identities.\n"
        f"2. Rotate credentials for any compromised access keys.\n"
        f"3. Enable CloudTrail log-file validation if not already active.\n"
        f"4. Restrict overly permissive IAM policies identified in the "
        f"flagged events.\n"
        f"5. Escalate to the security operations team for deeper forensic "
        f"investigation.\n\n"
        f"---\n"
        f"[MOCK MODE — This report was generated without Llama-3 inference]"
    )


# ---------------------------------------------------------------------------
#  LlamaResponder
# ---------------------------------------------------------------------------

class LlamaResponder:
    """Generates structured incident reports from anomalous CloudTrail events.

    In **mock mode** (default for interactive/CI use), reports are produced
    from a deterministic template without loading any model.  In **real mode**,
    the 4-bit quantised Llama-3-8b Instruct model is loaded via Unsloth and
    inference is run on GPU.

    Usage::

        responder = LlamaResponder(ResponderConfig(mock_mode=True))
        report = responder.generate_report(
            anomalous_logs=["CreateUser <*>", "AttachUserPolicy <*>"],
            context={"session_id": 42, "anomaly_ratio": 0.25},
        )
        print(report)
    """

    def __init__(self, config: ResponderConfig | None = None) -> None:
        self._config: ResponderConfig = config or ResponderConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    #  Lazy model loading                                                  #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Load the quantised Llama-3 model via Unsloth (lazy, idempotent).

        Skips entirely in mock mode.  Applies the ``CUDA_MEMORY_MAX``
        VRAM cap before allocation and wraps the load in an OOM guard.

        Raises:
            RuntimeError: If ``unsloth`` is not installed (with guidance
                on how to install it or use mock mode instead).
            RuntimeError: If CUDA runs out of memory.
        """
        if self._loaded:
            return

        if self._config.mock_mode:
            self._loaded = True
            return

        # --- Dependency guard ---
        try:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Unsloth is required for real Llama-3 inference but is not "
                "installed.  Options:\n"
                "  1. Install: pip install \"unsloth[colab-new] @ "
                "git+https://github.com/unslothai/unsloth.git\"\n"
                "  2. Use mock mode: python src/main.py --mode mock_inference\n"
                "  3. Set env var: RESPONDER_MOCK=true"
            ) from exc

        # --- VRAM cap ---
        import torch

        if torch.cuda.is_available():
            vram_frac: float = float(
                os.environ.get("CUDA_MEMORY_MAX", "0.5"),
            )
            torch.cuda.set_per_process_memory_fraction(vram_frac)

        # --- Model loading with OOM guard ---
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._config.model_name,
                max_seq_length=self._config.max_seq_length,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        except (RuntimeError, Exception) as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory while loading Llama-3.  Options:\n"
                    "  1. Close other GPU applications.\n"
                    "  2. Reduce CUDA_MEMORY_MAX (currently "
                    f"{os.environ.get('CUDA_MEMORY_MAX', '0.5')}).\n"
                    "  3. Use mock mode: --mode mock_inference"
                ) from exc
            raise

        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True

    # ------------------------------------------------------------------ #
    #  Report generation                                                   #
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        anomalous_logs: list[str],
        context: dict[str, Any],
    ) -> str:
        """Generate an incident report from reverse-mapped anomaly data.

        Args:
            anomalous_logs: Drain template strings for each flagged event
                            (e.g. ``["CreateUser <*>", "AttachUserPolicy <*>"]``).
            context:        Metadata dict with optional keys:
                            ``session_id``, ``total_events``,
                            ``anomaly_ratio``, ``timestamps``,
                            ``source_ips``, ``user_arns``, ``event_names``.

        Returns:
            Structured incident report string with five sections:
            Incident Summary, Anomaly Analysis, Severity Assessment,
            Attack Pattern Classification, and Recommended Mitigation.
        """
        if self._config.mock_mode:
            self._load_model()  # marks _loaded, no-op
            return _generate_mock_report(anomalous_logs, context)

        self._load_model()

        import torch

        prompt: str = _build_prompt(anomalous_logs, context)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self._model.device)

        # input_ids: (1, seq_len) int64
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                do_sample=True,
            )

        # output_ids: (1, seq_len + max_new_tokens) int64
        full_text: str = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True,
        )

        # Extract assistant response (after the last assistant header)
        marker: str = "assistant"
        marker_idx: int = full_text.rfind(marker)
        if marker_idx != -1:
            report: str = full_text[marker_idx + len(marker) :].strip()
        else:
            report = full_text.strip()

        return report
