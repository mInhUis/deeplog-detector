"""Responder package — Llama-3 incident-report generation.

Public API:
    LlamaResponder   — Generates structured incident reports from anomalous
                        CloudTrail events using Llama-3-8b (4-bit quantised).
    ResponderConfig   — Immutable configuration dataclass for the responder.
"""

from src.responder.llama_inference import LlamaResponder, ResponderConfig

__all__: list[str] = [
    "LlamaResponder",
    "ResponderConfig",
]
