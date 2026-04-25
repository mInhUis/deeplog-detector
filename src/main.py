"""End-to-end Cloud Log Analytics & Incident Response pipeline (Phase 5).

**Why a single orchestrator script?**
Phases 1–4 each solve one piece of the problem in isolation — parsing,
detection, and response.  This script is the *glue* that enforces the
strict pipeline order mandated by the thesis architecture:

    Raw AWS JSON → Drain Parser → DeepLog LSTM → Reverse-Map → Llama-3

By centralising the flow in ``run_pipeline()`` we guarantee that every
component receives correctly shaped data from its predecessor and that
the critical embedding/template safety check is always executed before
inference begins.

Usage:
    # Local testing (prints the Llama-3 prompt without loading the model):
    python src/main.py --input_file data/raw/cloudtrail.json \\
                       --deeplog_ckpt models/deeplog.pt \\
                       --mock_llm

    # Full pipeline (requires GPU + Unsloth):
    python src/main.py --input_file data/raw/cloudtrail.json \\
                       --deeplog_ckpt models/deeplog.pt

Environment Variables:
    CUDA_MEMORY_MAX     – Max VRAM fraction (0.0–1.0, default: 0.5)
    RESPONDER_MODEL     – HuggingFace model ID for Llama-3
    RESPONDER_MAX_TOKENS – Max tokens to generate (default: 1024)

Heavy Compute Warning (CLAUDE.md):
    When ``--mock_llm`` is omitted the real Llama-3-8b model is loaded
    via Unsloth with 4-bit quantisation.  This requires a CUDA GPU with
    ≥6 GB VRAM.  Always use ``--mock_llm`` for interactive development.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src.*`` imports resolve when the
# script is invoked directly (e.g. ``python src/main.py``).
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.parser.drain_parser import DrainParser
from src.detector.dataset import LogKeyDataset
from src.detector.deeplog import DeepLogModel
from src.detector.detect import detect_anomalies
from src.models.train_deeplog import train as _train_deeplog
from src.responder.llama_inference import (
    LlamaResponder,
    ResponderConfig,
    _build_prompt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TOP_K: Final[int] = 9
_CUDA_MEMORY_MAX: Final[float] = float(
    os.environ.get("CUDA_MEMORY_MAX", "0.5"),
)

# ---------------------------------------------------------------------------
# Mock-inference fixtures (used by ``--mode mock_inference`` and tests).
# Keys 0–4 are benign read-only AWS API calls; keys 5–7 are the
# privilege-escalation pattern we want DeepLog to flag in session 4.
# ---------------------------------------------------------------------------
_MOCK_NUM_KEYS: Final[int] = 8
_MOCK_TEMPLATES: Final[dict[int, str]] = {
    0: "ListBuckets <*>",
    1: "GetObject <*>",
    2: "DescribeInstances <*>",
    3: "ListUsers <*>",
    4: "GetCallerIdentity <*>",
    5: "CreateUser <*>",
    6: "AttachUserPolicy <*>",
    7: "CreateAccessKey <*>",
}
_MOCK_WINDOW_SIZE: Final[int] = 3


# ======================================================================== #
#  CLI                                                                      #
# ======================================================================== #

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the pipeline.

    Returns:
        Configured :class:`argparse.ArgumentParser` with three arguments:
        ``--input_file``, ``--deeplog_ckpt``, and ``--mock_llm``.
    """
    parser = argparse.ArgumentParser(
        description="Cloud Log Analytics — End-to-End Incident Response Pipeline",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "mock_inference"],
        help=(
            "Pipeline mode. 'full' loads real CloudTrail data and a "
            "trained DeepLog checkpoint. 'mock_inference' uses a "
            "synthetic 5-session dataset and a quick-trained model "
            "(safe for interactive runs per CLAUDE.md)."
        ),
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help=(
            "Path to raw CloudTrail JSON (.json array or .jsonl). "
            "Required for --mode full."
        ),
    )
    parser.add_argument(
        "--deeplog_ckpt",
        type=str,
        default=None,
        help=(
            "Path to saved DeepLog model checkpoint (e.g. models/deeplog.pt). "
            "Required for --mode full."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=(
            "Top-k threshold for DeepLog next-key anomaly detection. "
            "An event is flagged when its true key is not among the "
            "model's top-k predictions."
        ),
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default=None,
        help="Optional file path to write the generated report(s) to.",
    )
    parser.add_argument(
        "--mock_llm",
        action="store_true",
        default=False,
        help=(
            "Bypass Llama-3 inference and print the formatted prompt that "
            "would have been sent to the model.  Crucial for local testing."
        ),
    )
    return parser


# ======================================================================== #
#  Data loading & session grouping                                          #
# ======================================================================== #

def load_cloudtrail_events(path: Path) -> list[dict[str, Any]]:
    """Load CloudTrail events from a JSON array or JSONL file.

    Handles two standard CloudTrail export formats:

    1. **JSON object** with a top-level ``"Records"`` key::

           {"Records": [{"eventName": "ListBuckets", ...}, ...]}

    2. **JSONL** — one self-contained JSON object per line.

    Args:
        path: Path to the input file.

    Returns:
        Flat list of event dictionaries.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text: str = path.read_text(encoding="utf-8").strip()

    # --- Attempt 1: standard JSON (array or {"Records": [...]}) ---
    try:
        data: Any = json.loads(text)
        if isinstance(data, dict) and "Records" in data:
            return data["Records"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # --- Attempt 2: JSONL (one object per line) ---
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped: str = line.strip()
        if stripped:
            events.append(json.loads(stripped))
    return events


def event_to_log_string(event: dict[str, Any]) -> str:
    """Convert a single CloudTrail event dict into a Drain-parseable string.

    The resulting string contains the four most discriminative fields from
    a CloudTrail record.  Drain's pre-compiled regex patterns will
    automatically mask variable components (IP addresses, ARNs, hex IDs)
    into ``<*>`` wildcards during template extraction.

    Format::

        "{eventSource} {eventName} {sourceIPAddress} {userIdentity.arn}"

    Args:
        event: A single CloudTrail event dictionary.

    Returns:
        A single-line string ready for ``DrainParser.fit_transform()``.
    """
    source: str = event.get("eventSource", "")
    name: str = event.get("eventName", "")
    ip: str = event.get("sourceIPAddress", "")
    identity: Any = event.get("userIdentity", {})
    arn: str = identity.get("arn", "") if isinstance(identity, dict) else ""
    return f"{source} {name} {ip} {arn}"


def group_events_into_sessions(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group CloudTrail events into sessions by source identity.

    Uses the same session keys as ``preprocess.py``
    (``sourceIPAddress`` + ``userIdentity.arn``) to ensure consistency
    across the full pipeline.

    Args:
        events: Flat list of CloudTrail event dictionaries.

    Returns:
        List of sessions, each a list of event dicts belonging to the
        same ``(sourceIPAddress, userIdentity.arn)`` combination.
    """
    session_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list,
    )

    for event in events:
        ip: str = event.get("sourceIPAddress", "unknown")
        identity: Any = event.get("userIdentity", {})
        arn: str = (
            identity.get("arn", "unknown")
            if isinstance(identity, dict)
            else "unknown"
        )
        key: tuple[str, str] = (ip, arn)
        session_map[key].append(event)

    return list(session_map.values())


def _extract_session_metadata(
    session_events: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Extract per-session metadata needed by the incident responder.

    For each session, collects timestamps, source IPs, user ARNs, and
    event names — the context fields that ``LlamaResponder`` and
    ``_build_prompt`` use to enrich the report.

    Args:
        session_events: Grouped event lists (output of
                        ``group_events_into_sessions``).

    Returns:
        One metadata dict per session with keys: ``session_id``,
        ``timestamps``, ``source_ips``, ``user_arns``, ``event_names``.
    """
    metadata: list[dict[str, Any]] = []

    for s_idx, session in enumerate(session_events):
        metadata.append({
            "session_id": s_idx,
            "timestamps": [
                e.get("eventTime", "N/A") for e in session
            ],
            "source_ips": [
                e.get("sourceIPAddress", "N/A") for e in session
            ],
            "user_arns": [
                e.get("userIdentity", {}).get("arn", "N/A")
                if isinstance(e.get("userIdentity"), dict)
                else "N/A"
                for e in session
            ],
            "event_names": [
                e.get("eventName", "N/A") for e in session
            ],
        })

    return metadata


# ======================================================================== #
#  Mock pipeline data                                                       #
# ======================================================================== #

def generate_mock_pipeline_data() -> tuple[
    list[list[int]], dict[int, str], list[dict[str, Any]],
]:
    """Build a deterministic mini dataset for ``--mode mock_inference``.

    Produces 5 sessions of length 15 over the 8-key vocabulary in
    ``_MOCK_TEMPLATES``. Sessions 0–3 use only the 5 benign keys
    ``{0,1,2,3,4}``; session 4 (the "attack") embeds every suspicious
    key ``{5,6,7}`` so an undertrained DeepLog flags it under low
    ``top_k``. Determinism is enforced by hand-crafted patterns rather
    than ``random``, so test assertions hold across Python runs.

    Returns:
        Tuple of ``(sessions, templates, metadata)`` mirroring the
        shapes produced by the real Drain + ``_extract_session_metadata``
        path so the rest of the pipeline can consume them unchanged.
    """
    normal_pool: list[int] = [0, 1, 2, 3, 4]
    sessions: list[list[int]] = []

    # Sessions 0–3: rotate the 5-key benign pool to keep each session
    # distinct but still anomaly-free relative to itself.
    for s_idx in range(4):
        rotated: list[int] = (
            normal_pool[s_idx:] + normal_pool[:s_idx]
        )
        sessions.append([rotated[i % 5] for i in range(15)])

    # Session 4: privilege-escalation pattern. Suspicious keys 5/6/7
    # appear at fixed positions and are surrounded by benign reads.
    attack: list[int] = [
        0, 1, 2, 0, 1,   # benign warm-up
        5, 6, 7,         # CreateUser → AttachUserPolicy → CreateAccessKey
        3, 4, 0,         # benign cool-down
        5, 6, 7, 0,      # repeat pattern + trailing benign
    ]
    sessions.append(attack)

    templates: dict[int, str] = dict(_MOCK_TEMPLATES)

    metadata: list[dict[str, Any]] = []
    for s_idx, session in enumerate(sessions):
        metadata.append({
            "session_id": s_idx,
            "timestamps": [
                f"2026-04-25T00:00:{j:02d}Z" for j in range(len(session))
            ],
            "source_ips": [f"10.0.0.{s_idx}" for _ in session],
            "user_arns": [
                f"arn:aws:iam::000000000000:user/mock{s_idx}"
                for _ in session
            ],
            "event_names": [
                templates[k].split(" ", 1)[0] for k in session
            ],
        })

    return sessions, templates, metadata


# ======================================================================== #
#  Checkpoint loader                                                        #
# ======================================================================== #

def load_deeplog_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DeepLogModel, dict[str, Any]]:
    """Load a DeepLog checkpoint and reconstruct the exact model architecture.

    The checkpoint dict (saved by ``train_deeplog.py``) stores all
    hyperparameters (``num_keys``, ``embedding_dim``, ``hidden_size``,
    ``num_layers``, ``dropout``) alongside the model weights, enabling
    exact architecture reconstruction without external configuration.

    Args:
        checkpoint_path: Path to ``deeplog.pt``.
        device:          Torch device to map tensors onto.

    Returns:
        Tuple of ``(model, checkpoint_dict)`` where the model is already
        in eval mode on the requested device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If required hyperparameters are missing.
    """
    path: Path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DeepLog checkpoint not found: {path}\n"
            f"Train first: python src/models/train_deeplog.py --mode full"
        )

    ckpt: dict[str, Any] = torch.load(
        path, map_location=device, weights_only=False,
    )

    # Validate required checkpoint keys
    required_keys: list[str] = ["model_state_dict", "num_keys", "window_size"]
    for key in required_keys:
        if key not in ckpt:
            raise KeyError(
                f"Checkpoint is missing required key '{key}'. "
                f"Available keys: {list(ckpt.keys())}"
            )

    # Reconstruct the exact architecture from saved hyperparameters
    model: DeepLogModel = DeepLogModel(
        num_keys=ckpt["num_keys"],
        embedding_dim=ckpt.get("embedding_dim", 64),
        hidden_size=ckpt.get("hidden_size", 64),
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


# ======================================================================== #
#  Reverse-map anomalies                                                    #
# ======================================================================== #

def reverse_map_anomalies(
    sessions: list[list[int]],
    anomaly_flags: list[list[bool]],
    templates: dict[int, str],
    session_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Translate DeepLog anomaly flags back to human-readable Drain templates.

    For each session that contains at least one anomalous event, this
    function reverse-maps the integer log keys at flagged positions
    through the Drain ``templates`` dictionary, producing the text
    strings that the Llama-3 responder needs.

    Args:
        sessions:         Per-session log-key sequences (same structure as
                          the ``detect_anomalies`` input).
        anomaly_flags:    Output of ``detect_anomalies()`` — nested bool
                          flags mirroring *sessions*.
        templates:        Drain template map ``{key_int: template_str}``.
        session_metadata: Per-session metadata dicts (from
                          ``_extract_session_metadata``).

    Returns:
        List of dicts (one per anomalous session) each containing:
            ``session_index``   : int
            ``anomalous_logs``  : list[str]  (Drain template text)
            ``anomaly_indices`` : list[int]  (positions within the session)
            ``context``         : dict       (metadata for the responder)
    """
    results: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = (
        session_metadata if session_metadata is not None else []
    )

    for s_idx, (seq, flags) in enumerate(zip(sessions, anomaly_flags)):
        anomaly_indices: list[int] = [
            i for i, flag in enumerate(flags) if flag
        ]
        if not anomaly_indices:
            continue

        # Reverse-map integer keys → Drain template text
        anomalous_logs: list[str] = [
            templates.get(seq[i], f"<unknown key {seq[i]}>")
            for i in anomaly_indices
        ]

        # Build context dict for the responder
        meta: dict[str, Any] = (
            metadata[s_idx] if s_idx < len(metadata) else {}
        )
        context: dict[str, Any] = {
            "session_id": meta.get("session_id", s_idx),
            "total_events": len(seq),
            "anomaly_ratio": (
                len(anomaly_indices) / len(seq) if seq else 0.0
            ),
        }

        # Slice per-anomaly metadata from full session metadata
        for field_name in (
            "timestamps", "source_ips", "user_arns", "event_names",
        ):
            full_list: list[str] = meta.get(field_name, [])
            context[field_name] = [
                full_list[i] if i < len(full_list) else "N/A"
                for i in anomaly_indices
            ]

        results.append({
            "session_index": s_idx,
            "anomalous_logs": anomalous_logs,
            "anomaly_indices": anomaly_indices,
            "context": context,
        })

    return results


# ======================================================================== #
#  Pipeline orchestrator                                                    #
# ======================================================================== #

def run_pipeline(args: argparse.Namespace) -> str | None:
    """Execute the full Cloud Log Analytics & Incident Response pipeline.

    Pipeline flow:
        1. Load raw CloudTrail JSON → group into sessions → Drain Parser.
        2. Load the trained DeepLog model from the checkpoint.
        3. **Critical safety check**: embedding size == template count.
        4. Run DeepLog top-k anomaly detection on the integer sequences.
        5. Reverse-map flagged keys back to Drain template text.
        6. Generate incident reports via Llama-3 (or print the prompt if
           ``--mock_llm`` is set).

    Args:
        args: Parsed CLI namespace with ``input_file``, ``deeplog_ckpt``,
              and ``mock_llm`` attributes.

    Returns:
        The concatenated report/prompt output as a string (useful for
        testing), or ``None`` if no anomalies were found.

    Raises:
        FileNotFoundError: If input file or checkpoint is missing.
        ValueError: If the model's embedding table does not match the
                    number of Drain templates (stale model).
    """
    if getattr(args, "mode", None) == "mock_inference":
        return _run_mock_pipeline(args)

    input_path: Path = Path(args.input_file)
    ckpt_path: Path = Path(args.deeplog_ckpt)

    # ================================================================== #
    #  Step 1 — Load raw JSON, group into sessions, run Drain             #
    # ================================================================== #
    print("[Step 1/5] Loading CloudTrail events and running Drain parser ...")

    events: list[dict[str, Any]] = load_cloudtrail_events(input_path)
    if not events:
        print("         WARNING: Input file contains zero events.")
        return None
    print(f"          Loaded {len(events)} events from {input_path}")

    # Group events by (sourceIPAddress, userIdentity.arn)
    session_events: list[list[dict[str, Any]]] = group_events_into_sessions(
        events,
    )
    print(f"          Grouped into {len(session_events)} session(s)")

    # Convert each event to a Drain-parseable string, tracking session
    # boundaries so the flat log_keys output can be split back.
    all_log_strings: list[str] = []
    session_lengths: list[int] = []

    for session in session_events:
        strings: list[str] = [event_to_log_string(e) for e in session]
        all_log_strings.extend(strings)
        session_lengths.append(len(strings))

    # Extract per-session metadata for the responder
    session_metadata: list[dict[str, Any]] = _extract_session_metadata(
        session_events,
    )

    # Run Drain on the full corpus to build a global template vocabulary.
    # Input:  list[str] of length N (all events, flattened)
    # Output: log_keys list[int] of length N, templates dict[int, str]
    parser: DrainParser = DrainParser()
    log_keys: list[int]
    templates: dict[int, str]
    log_keys, templates = parser.fit_transform(all_log_strings)
    print(f"          Drain extracted {len(templates)} unique template(s)")

    # Split the flat log_keys sequence back into per-session chunks
    sessions: list[list[int]] = []
    offset: int = 0
    for length in session_lengths:
        sessions.append(log_keys[offset : offset + length])
        offset += length

    # ================================================================== #
    #  Step 2 — Load DeepLog model from checkpoint                        #
    # ================================================================== #
    print("[Step 2/5] Loading DeepLog model from checkpoint ...")

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_CUDA_MEMORY_MAX)

    model: DeepLogModel
    ckpt: dict[str, Any]
    model, ckpt = load_deeplog_checkpoint(ckpt_path, device)

    window_size: int = ckpt["window_size"]
    print(
        f"          Device: {device} | window_size: {window_size} | "
        f"epoch: {ckpt.get('epoch', '?')} | "
        f"loss: {ckpt.get('final_loss', '?')}"
    )

    # ================================================================== #
    #  Step 3 — CRITICAL SAFETY CHECK                                     #
    #                                                                      #
    #  The DeepLog embedding table was sized for the vocabulary that       #
    #  existed at training time.  If Drain produces a different number     #
    #  of templates on new data, the model's weight matrices are           #
    #  incompatible and inference would produce garbage.                   #
    # ================================================================== #
    num_embeddings: int = model._embedding.num_embeddings
    num_templates: int = len(templates)

    if num_embeddings != num_templates:
        raise ValueError(
            f"Embedding/template mismatch: the DeepLog model's embedding "
            f"table has {num_embeddings} entries but Drain produced "
            f"{num_templates} templates on the current input data. The "
            f"model was trained on a different log vocabulary and cannot "
            f"be used for inference. Delete the stale checkpoint and "
            f"retrain DeepLog on the current Drain output:\n"
            f"    rm {ckpt_path}\n"
            f"    python src/models/train_deeplog.py --mode full"
        )
    print(
        f"          Safety check PASSED: {num_embeddings} embeddings == "
        f"{num_templates} templates"
    )

    # ================================================================== #
    #  Step 4 — Anomaly detection                                         #
    # ================================================================== #
    # Cap top_k to the model's vocabulary size to prevent torch.topk
    # from raising when k > number of output classes.
    requested_top_k: int = int(getattr(args, "top_k", _DEFAULT_TOP_K))
    effective_top_k: int = min(requested_top_k, num_embeddings)
    print(
        f"[Step 3/5] Running DeepLog anomaly detection "
        f"(top_k={effective_top_k}) ..."
    )

    # detect_anomalies tensor shapes (per window):
    #   window_tensor : (1, window_size) int64
    #   logits        : (1, num_keys)    float32
    #   top_indices   : (1, top_k)       int64
    anomaly_flags: list[list[bool]] = detect_anomalies(
        model=model,
        sequences=sessions,
        window_size=window_size,
        top_k=effective_top_k,
        device=device,
    )

    total_anomalies: int = sum(
        flag for session_flags in anomaly_flags for flag in session_flags
    )
    anomalous_session_count: int = sum(
        1 for flags in anomaly_flags if any(flags)
    )
    print(
        f"          {total_anomalies} anomalous event(s) across "
        f"{anomalous_session_count} session(s)"
    )

    if total_anomalies == 0:
        print("\n[Done] No anomalies detected — no reports to generate.")
        return None

    # ================================================================== #
    #  Step 5 — Reverse-map anomalous keys to text                        #
    # ================================================================== #
    print("[Step 4/5] Reverse-mapping anomalous keys to Drain templates ...")

    mapped: list[dict[str, Any]] = reverse_map_anomalies(
        sessions=sessions,
        anomaly_flags=anomaly_flags,
        templates=templates,
        session_metadata=session_metadata,
    )
    print(f"          {len(mapped)} session(s) with anomalies to report")

    # ================================================================== #
    #  Step 6 — Generate reports (or print prompt in mock mode)           #
    # ================================================================== #
    print("[Step 5/5] Generating incident reports ...")

    output_parts: list[str] = []

    if args.mock_llm:
        # --mock_llm: print the exact Llama-3 prompt without loading
        # any model.  This lets the developer verify that the reverse-
        # mapped text and context metadata are correctly formatted
        # before committing GPU resources.
        for entry in mapped:
            prompt: str = _build_prompt(
                anomalous_logs=entry["anomalous_logs"],
                context=entry["context"],
            )
            separator: str = "\n" + "=" * 72
            header: str = (
                f"{separator}\n"
                f"  LLAMA-3 PROMPT  (--mock_llm)  |  "
                f"Session {entry['context']['session_id']}\n"
                f"{'=' * 72}"
            )
            print(header)
            print(prompt)
            print("=" * 72)
            output_parts.append(prompt)
    else:
        # Real Llama-3 inference via the LlamaResponder.
        responder: LlamaResponder = LlamaResponder(
            ResponderConfig(
                model_name=os.environ.get(
                    "RESPONDER_MODEL",
                    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                ),
                max_new_tokens=int(
                    os.environ.get("RESPONDER_MAX_TOKENS", "1024"),
                ),
                mock_mode=False,
            ),
        )
        for entry in mapped:
            report: str = responder.generate_report(
                anomalous_logs=entry["anomalous_logs"],
                context=entry["context"],
            )
            separator = "\n" + "=" * 72
            header = (
                f"{separator}\n"
                f"  INCIDENT REPORT  |  "
                f"Session {entry['context']['session_id']}\n"
                f"{'=' * 72}"
            )
            print(header)
            print(report)
            print("=" * 72)
            output_parts.append(report)

    full_output: str = ("\n\n" + "=" * 72 + "\n\n").join(output_parts)
    print("\n[Done] Pipeline complete.")
    return full_output


# ======================================================================== #
#  Mock-inference pipeline                                                  #
# ======================================================================== #

def _run_mock_pipeline(args: argparse.Namespace) -> str | None:
    """Run the end-to-end pipeline on the synthetic mock dataset.

    Skips CloudTrail loading, Drain, and the real Llama-3 model. The
    DeepLog model is freshly initialised and quick-trained for 20
    epochs on the four normal sessions only (semi-supervised baseline)
    before scoring all 5 sessions.

    Args:
        args: Namespace with optional ``top_k`` and ``report_output``.

    Returns:
        The joined 5-section incident report string, or ``None`` when
        DeepLog flags no anomalies (possible at high ``top_k``).
    """
    print("[mock] Generating synthetic sessions, templates, metadata ...")
    sessions, templates, metadata = generate_mock_pipeline_data()

    torch.manual_seed(0)
    device: torch.device = torch.device("cpu")

    model: DeepLogModel = DeepLogModel(
        num_keys=_MOCK_NUM_KEYS,
        embedding_dim=16,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    # Quick semi-supervised pre-training on the 4 normal sessions only.
    normal_sessions: list[list[int]] = sessions[:4]
    dataset: LogKeyDataset = LogKeyDataset(
        normal_sessions, window_size=_MOCK_WINDOW_SIZE,
    )
    loader: DataLoader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("[mock] Quick-training DeepLog (20 epochs, normal sessions) ...")
    _train_deeplog(
        model=model,
        dataloader=loader,
        epochs=20,
        lr=0.01,
        device=device,
    )
    model.eval()

    requested_top_k: int = int(getattr(args, "top_k", _DEFAULT_TOP_K))
    effective_top_k: int = min(requested_top_k, _MOCK_NUM_KEYS)
    print(f"[mock] Detecting anomalies (top_k={effective_top_k}) ...")
    anomaly_flags: list[list[bool]] = detect_anomalies(
        model=model,
        sequences=sessions,
        window_size=_MOCK_WINDOW_SIZE,
        top_k=effective_top_k,
        device=device,
    )

    total_anomalies: int = sum(
        flag for session_flags in anomaly_flags for flag in session_flags
    )
    if total_anomalies == 0:
        print("[mock] No anomalies detected.")
        return None

    mapped: list[dict[str, Any]] = reverse_map_anomalies(
        sessions=sessions,
        anomaly_flags=anomaly_flags,
        templates=templates,
        session_metadata=metadata,
    )
    if not mapped:
        return None

    responder: LlamaResponder = LlamaResponder(
        ResponderConfig(mock_mode=True),
    )
    output_parts: list[str] = []
    for entry in mapped:
        report: str = responder.generate_report(
            anomalous_logs=entry["anomalous_logs"],
            context=entry["context"],
        )
        output_parts.append(report)

    full_output: str = ("\n\n" + "=" * 72 + "\n\n").join(output_parts)

    report_output: str | None = getattr(args, "report_output", None)
    if report_output:
        Path(report_output).write_text(full_output, encoding="utf-8")
        print(f"[mock] Wrote report to {report_output}")

    print("[mock] Pipeline complete.")
    return full_output


# ======================================================================== #
#  Entry point                                                              #
# ======================================================================== #

def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser: argparse.ArgumentParser = build_arg_parser()
    args: argparse.Namespace = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
