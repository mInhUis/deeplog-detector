"""Detector package — DeepLog LSTM anomaly detection.

Public API:
    DeepLogModel   — LSTM nn.Module for next-log-key prediction.
    LogKeyDataset  — Sliding-window PyTorch Dataset.
    detect_anomalies    — Top-k inference over test sessions.
    evaluate_predictions — Precision / Recall / F1 scorer.
"""

from src.detector.dataset import LogKeyDataset
from src.detector.deeplog import DeepLogModel
from src.detector.detect import detect_anomalies, evaluate_predictions

__all__: list[str] = [
    "DeepLogModel",
    "LogKeyDataset",
    "detect_anomalies",
    "evaluate_predictions",
]
