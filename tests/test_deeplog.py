"""Unit tests for the DeepLog detector package.

Tests cover:
    1. LogKeyDataset  — sliding window construction, shapes, data leakage.
    2. DeepLogModel   — forward pass tensor shapes, gradient flow.
    3. detect_anomalies — top-k anomaly flagging on a mock-trained model.
    4. evaluate_predictions — Precision / Recall / F1 arithmetic.

Uses a small, mocked dataset of 5–10 log keys per session (per CLAUDE.md
heavy-compute warning) to verify logic without hardware strain.

Mock Dataset (3 sessions of deterministic repeating patterns):
    Session 0: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  (len=15)
    Session 1: [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]  (len=15)
    Session 2: [2, 3]                                             (len=2)

With window_size=3:
    • Sessions 0 and 1 produce (15 - 3) = 12 samples each → 24 total.
    • Session 2 is too short (2 < 3+1) → 0 samples.
    • Total expected samples: 24.

Tensor shapes verified:
    X : (batch_size, window_size)  int64
    y : (batch_size,)              int64
    logits : (batch_size, num_keys) float32
"""

from __future__ import annotations

from typing import Final

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.dataset import LogKeyDataset
from src.detector.deeplog import DeepLogModel
from src.detector.detect import detect_anomalies, evaluate_predictions


# ======================================================================== #
#  Mock data                                                                #
# ======================================================================== #

NUM_KEYS: Final[int] = 5
WINDOW_SIZE: Final[int] = 3

MOCK_SESSIONS: Final[list[list[int]]] = [
    # Session 0: repeating 0-4 cycle, length 15
    [i % NUM_KEYS for i in range(15)],
    # Session 1: offset by 1, length 15
    [(i + 1) % NUM_KEYS for i in range(15)],
    # Session 2: too short to form a single sample with window_size=3
    [2, 3],
]

# Sessions 0 and 1 each yield (15 - 3) = 12 samples; session 2 yields 0.
EXPECTED_TOTAL_SAMPLES: Final[int] = 24


# ======================================================================== #
#  Fixtures                                                                 #
# ======================================================================== #

@pytest.fixture()
def dataset() -> LogKeyDataset:
    """A LogKeyDataset built from the mock sessions."""
    return LogKeyDataset(MOCK_SESSIONS, window_size=WINDOW_SIZE)


@pytest.fixture()
def model() -> DeepLogModel:
    """A DeepLogModel with small dimensions for fast testing."""
    return DeepLogModel(
        num_keys=NUM_KEYS,
        embedding_dim=8,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture()
def trained_model(dataset: LogKeyDataset) -> DeepLogModel:
    """A DeepLogModel trained for a few epochs on mock data.

    We train just enough for the model to learn the deterministic
    repeating pattern — this lets us verify that detect_anomalies
    flags out-of-pattern keys.
    """
    mdl = DeepLogModel(
        num_keys=NUM_KEYS,
        embedding_dim=16,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(mdl.parameters(), lr=1e-2)

    mdl.train()
    for _ in range(80):  # enough epochs for a trivial pattern
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            logits, _ = mdl(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

    return mdl


# ======================================================================== #
#  1. LogKeyDataset tests                                                   #
# ======================================================================== #

class TestLogKeyDataset:
    """Verify sliding window construction and data shapes."""

    def test_total_sample_count(self, dataset: LogKeyDataset) -> None:
        """Dataset length must equal sum of (session_len - w) for valid sessions."""
        assert len(dataset) == EXPECTED_TOTAL_SAMPLES

    def test_short_session_skipped(self) -> None:
        """Sessions shorter than window_size + 1 must yield zero samples."""
        ds = LogKeyDataset([[0, 1]], window_size=3)
        assert len(ds) == 0

    def test_single_sample_session(self) -> None:
        """A session of exactly window_size + 1 yields exactly 1 sample."""
        ds = LogKeyDataset([[0, 1, 2, 3]], window_size=3)
        assert len(ds) == 1

    def test_empty_input(self) -> None:
        """Empty session list → empty dataset, no crash."""
        ds = LogKeyDataset([], window_size=5)
        assert len(ds) == 0

    def test_window_size_property(self, dataset: LogKeyDataset) -> None:
        """The window_size property must reflect the constructor value."""
        assert dataset.window_size == WINDOW_SIZE

    def test_invalid_window_size_raises(self) -> None:
        """window_size < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            LogKeyDataset([[0, 1, 2]], window_size=0)

    def test_x_shape(self, dataset: LogKeyDataset) -> None:
        """X must be a 1-D int64 tensor of shape (window_size,)."""
        X, _ = dataset[0]
        assert X.shape == (WINDOW_SIZE,)
        assert X.dtype == torch.long

    def test_y_shape(self, dataset: LogKeyDataset) -> None:
        """y must be a scalar int64 tensor of shape ()."""
        _, y = dataset[0]
        assert y.shape == ()
        assert y.dtype == torch.long

    def test_data_leakage_target_after_window(
        self, dataset: LogKeyDataset
    ) -> None:
        """DATA-LEAKAGE CHECK: y must equal the key immediately after X.

        For the first sample of session 0 with window_size=3:
            X = [0, 1, 2]   (positions 0, 1, 2)
            y = 3            (position 3)
        The target is strictly *after* all input positions.
        """
        X, y = dataset[0]
        # Session 0 starts [0, 1, 2, 3, ...]
        assert X.tolist() == [0, 1, 2]
        assert y.item() == 3

    def test_no_cross_session_leakage(self) -> None:
        """Sliding windows must NOT span session boundaries.

        Two sessions [0, 1, 2, 3] and [10, 11, 12, 13] with window_size=3.
        The last sample of session 0 should be X=[1,2,3], y would not exist
        because position 4 doesn't exist in session 0.
        Session 1's first sample should start fresh at [10, 11, 12].
        """
        ds = LogKeyDataset(
            [[0, 1, 2, 3], [10, 11, 12, 13]],
            window_size=3,
        )
        # Session 0: 1 sample → X=[0,1,2], y=3
        # Session 1: 1 sample → X=[10,11,12], y=13
        assert len(ds) == 2

        X0, y0 = ds[0]
        assert X0.tolist() == [0, 1, 2]
        assert y0.item() == 3

        X1, y1 = ds[1]
        assert X1.tolist() == [10, 11, 12]
        assert y1.item() == 13

    def test_dataloader_batching(self, dataset: LogKeyDataset) -> None:
        """DataLoader must collate into proper batch dimensions.

        Expected batch shapes:
            X_batch: (batch_size, window_size) int64
            y_batch: (batch_size,)             int64
        """
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (4, WINDOW_SIZE)
        assert y_batch.shape == (4,)


# ======================================================================== #
#  2. DeepLogModel tests                                                    #
# ======================================================================== #

class TestDeepLogModel:
    """Verify forward-pass shapes, gradient flow, and properties."""

    def test_output_logit_shape(self, model: DeepLogModel) -> None:
        """logits must have shape (batch_size, num_keys).

        Tensor flow:
            (2, 3) int64 → Embedding → (2, 3, 8) → LSTM → (2, 3, 8)
            → last step (2, 8) → Linear → (2, 5)
        """
        X = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
        # X: (2, 3)
        logits, _ = model(X)
        # logits: (2, num_keys=5)
        assert logits.shape == (2, NUM_KEYS)

    def test_output_dtype_float32(self, model: DeepLogModel) -> None:
        """Logits must be float32 (standard for cross-entropy input)."""
        X = torch.tensor([[0, 1, 2]], dtype=torch.long)
        logits, _ = model(X)
        assert logits.dtype == torch.float32

    def test_hidden_state_shapes(self, model: DeepLogModel) -> None:
        """Hidden state (h_n, c_n) shapes must be (num_layers, batch, hidden).

        Expected:
            h_n: (1, 2, 8)
            c_n: (1, 2, 8)
        """
        X = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
        _, (h_n, c_n) = model(X)
        assert h_n.shape == (model.num_layers, 2, model.hidden_size)
        assert c_n.shape == (model.num_layers, 2, model.hidden_size)

    def test_gradient_flows_to_all_parameters(
        self, model: DeepLogModel
    ) -> None:
        """A backward pass must produce non-None gradients for all params."""
        X = torch.tensor([[0, 1, 2]], dtype=torch.long)
        y = torch.tensor([3], dtype=torch.long)

        logits, _ = model(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_single_sample_batch(self, model: DeepLogModel) -> None:
        """Batch size of 1 must work without broadcasting errors."""
        X = torch.tensor([[0, 1, 2]], dtype=torch.long)
        logits, _ = model(X)
        assert logits.shape == (1, NUM_KEYS)

    def test_sequence_length_one(self) -> None:
        """A sequence of length 1 (single token) must still produce output."""
        mdl = DeepLogModel(num_keys=3, embedding_dim=4, hidden_size=4,
                           num_layers=1)
        X = torch.tensor([[2]], dtype=torch.long)
        logits, _ = mdl(X)
        assert logits.shape == (1, 3)

    def test_hidden_size_property(self, model: DeepLogModel) -> None:
        """hidden_size property must match constructor arg."""
        assert model.hidden_size == 8

    def test_num_layers_property(self, model: DeepLogModel) -> None:
        """num_layers property must match constructor arg."""
        assert model.num_layers == 1

    def test_custom_initial_hidden(self, model: DeepLogModel) -> None:
        """Passing explicit (h_0, c_0) must not raise and shapes must match."""
        batch_size = 2
        h_0 = torch.zeros(model.num_layers, batch_size, model.hidden_size)
        c_0 = torch.zeros(model.num_layers, batch_size, model.hidden_size)
        X = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)

        logits, (h_n, c_n) = model(X, hidden=(h_0, c_0))
        assert logits.shape == (batch_size, NUM_KEYS)
        assert h_n.shape == h_0.shape
        assert c_n.shape == c_0.shape


# ======================================================================== #
#  3. Anomaly detection tests                                               #
# ======================================================================== #

class TestDetectAnomalies:
    """Verify top-k detection logic on a mock-trained model."""

    def test_output_structure_mirrors_input(
        self, trained_model: DeepLogModel
    ) -> None:
        """detect_anomalies must return one flag-list per session."""
        test_sessions: list[list[int]] = [
            [0, 1, 2, 3, 4, 0, 1],
            [1, 2, 3],
        ]
        flags = detect_anomalies(
            trained_model, test_sessions,
            window_size=WINDOW_SIZE, top_k=3,
        )
        assert len(flags) == len(test_sessions)
        for f, s in zip(flags, test_sessions):
            assert len(f) == len(s)

    def test_flags_are_booleans(self, trained_model: DeepLogModel) -> None:
        """Every flag must be a Python bool."""
        flags = detect_anomalies(
            trained_model, [MOCK_SESSIONS[0]],
            window_size=WINDOW_SIZE, top_k=3,
        )
        for f in flags[0]:
            assert isinstance(f, bool)

    def test_first_w_events_always_normal(
        self, trained_model: DeepLogModel
    ) -> None:
        """The first window_size events lack a full context window and
        must always be marked False (normal)."""
        flags = detect_anomalies(
            trained_model, [MOCK_SESSIONS[0]],
            window_size=WINDOW_SIZE, top_k=3,
        )
        assert flags[0][:WINDOW_SIZE] == [False] * WINDOW_SIZE

    def test_normal_pattern_mostly_unflagged(
        self, trained_model: DeepLogModel
    ) -> None:
        """A repeating normal pattern the model was trained on should have
        few or no anomaly flags (with generous top_k)."""
        normal_session: list[int] = [i % NUM_KEYS for i in range(20)]
        flags = detect_anomalies(
            trained_model, [normal_session],
            window_size=WINDOW_SIZE, top_k=NUM_KEYS,  # k = vocab → nothing flagged
        )
        # With k = num_keys, every key is in top-k → zero anomalies
        assert not any(flags[0])

    def test_injected_anomaly_detected(
        self, trained_model: DeepLogModel
    ) -> None:
        """Injecting an out-of-pattern key should be flagged with small k.

        The model learned patterns like [0,1,2]→3, [1,2,3]→4, etc.
        We inject a break: [..., 0, 1, 2, 0, ...] where position after
        [0,1,2] should be 3 but we put 0.  With top_k=1 the model must
        flag the break point.
        """
        # Normal prefix + anomalous break
        anomalous: list[int] = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]
        #                       ^-- position 8 breaks the [0,1,2]→3 pattern
        flags = detect_anomalies(
            trained_model, [anomalous],
            window_size=WINDOW_SIZE, top_k=1,
        )
        # Position 8 (key=0, after window [0,1,2]) should be anomalous
        # The model should predict 3 as top-1 after seeing [0,1,2]
        assert flags[0][8] is True

    def test_short_session_all_normal(
        self, trained_model: DeepLogModel
    ) -> None:
        """A session shorter than window_size must be all-normal flags."""
        short: list[int] = [0, 1]
        flags = detect_anomalies(
            trained_model, [short],
            window_size=WINDOW_SIZE, top_k=3,
        )
        assert flags[0] == [False, False]


# ======================================================================== #
#  4. Evaluation metrics tests                                              #
# ======================================================================== #

class TestEvaluatePredictions:
    """Verify Precision / Recall / F1 / Accuracy computation.

    Imbalanced Metric Focus (Advisor Skill 4):
    These tests verify that the metrics are computed correctly for
    scenarios typical of rare-event anomaly detection.
    """

    def test_perfect_predictions(self) -> None:
        """Perfect match → P=1, R=1, F1=1, Acc=1."""
        pred = [[True, False, True, False]]
        truth = [[True, False, True, False]]
        m = evaluate_predictions(pred, truth)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_all_false_negatives(self) -> None:
        """Model predicts all normal, but all are anomalies → R=0."""
        pred = [[False, False, False]]
        truth = [[True, True, True]]
        m = evaluate_predictions(pred, truth)
        assert m["recall"] == pytest.approx(0.0)
        assert m["fn"] == 3
        assert m["tp"] == 0

    def test_all_false_positives(self) -> None:
        """Model flags everything, nothing is actually anomalous → P=0."""
        pred = [[True, True, True]]
        truth = [[False, False, False]]
        m = evaluate_predictions(pred, truth)
        assert m["precision"] == pytest.approx(0.0)
        assert m["fp"] == 3
        assert m["tp"] == 0

    def test_mixed_scenario(self) -> None:
        """Hand-computed metrics for a known confusion matrix.

        Pred:  [T, T, F, F, T]
        Truth: [T, F, T, F, T]
        TP=2, FP=1, FN=1, TN=1
        P = 2/3, R = 2/3, F1 = 2/3, Acc = 3/5
        """
        pred = [[True, True, False, False, True]]
        truth = [[True, False, True, False, True]]
        m = evaluate_predictions(pred, truth)
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3)
        assert m["accuracy"] == pytest.approx(3 / 5)

    def test_empty_input(self) -> None:
        """Empty predictions → all metrics default to 0.0."""
        m = evaluate_predictions([], [])
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_highly_imbalanced_accuracy_is_misleading(self) -> None:
        """Demonstrate why accuracy is misleading for rare anomalies.

        99 normals + 1 anomaly.  Model predicts ALL normal.
        Accuracy = 99% — looks great!
        Recall = 0% — the one anomaly was missed.
        This is exactly why we prioritise Recall and F1 (Skill 4).
        """
        pred = [[False] * 100]
        truth = [[False] * 99 + [True]]
        m = evaluate_predictions(pred, truth)
        assert m["accuracy"] == pytest.approx(0.99)
        assert m["recall"] == pytest.approx(0.0)  # missed the anomaly!

    def test_multi_session_aggregation(self) -> None:
        """Metrics aggregate correctly across multiple sessions."""
        pred = [[True, False], [False, True]]
        truth = [[True, True], [False, True]]
        # TP=2 (pos 0 sess 0, pos 1 sess 1)
        # FN=1 (pos 1 sess 0 — missed)
        # TN=1 (pos 0 sess 1)
        # FP=0
        m = evaluate_predictions(pred, truth)
        assert m["tp"] == 2
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["fp"] == 0
