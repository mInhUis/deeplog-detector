# Cloud Log Analytics & Incident Response Thesis

## Build & Run Commands
* Install dependencies: `pip install -r requirements.txt`
* Run tests: `pytest tests/`
* Run data preprocessing: `python src/pipeline/preprocess.py`
* Run Drain parser: `python src/parser/drain_parser.py`
* Train DeepLog (mock, safe for interactive): `python src/models/train_deeplog.py --mode mock`
* Train DeepLog (full data): `python src/models/train_deeplog.py --mode full`
* Run full pipeline mock: `python src/main.py --mode mock_inference`
* Run full pipeline (real data): `python src/main.py --mode full`
* Run full pipeline (sensitive): `python src/main.py --mode full --top-k 3`
* Build Docker Container: `docker build -t cloud-log-pipeline .`

## Project Architecture
* `data/raw/`: Raw AWS CloudTrail logs (JSON).
* `data/interim/`: Merged JSONL file (`flaws_merged.jsonl`) ready for preprocessing.
* `data/processed/`: Structured logs (`train_sessions.csv`, `test_sessions.csv`) and Drain outputs (`train_log_keys.json`).
* `models/`: Saved DeepLog checkpoints (`deeplog.pt`).
* `src/pipeline/preprocess.py`: **[COMPLETE — Phase 1]** Cleans, splits, and session-groups raw CloudTrail logs into train/test CSVs.
* `src/parser/drain_parser.py`: **[COMPLETE — Phase 2]** Drain algorithm; outputs `log_keys: list[int]` and `templates: dict[int, str]`.
* `src/detector/`: **[COMPLETE — Phase 3]** PyTorch DeepLog LSTM.
  * `dataset.py`: `LogKeyDataset` — sliding-window `(X, y)` pairs.
  * `deeplog.py`: `DeepLogModel` — Embedding → LSTM → Linear next-key classifier.
  * `detect.py`: `detect_anomalies()` (top-k inference) + `evaluate_predictions()` (P/R/F1). Includes stale-model embedding guard.
* `src/models/train_deeplog.py`: **[COMPLETE — Phase 3]** Training script with mock mode, VRAM cap, checkpoint resume, mid-training saves every 10 epochs, and `num_keys` mismatch guard.
* `src/responder/`: **[COMPLETE — Phase 4]** Llama-3-8b incident-response report generator.
  * `llama_inference.py`: `LlamaResponder` — lazy-loaded 4-bit quantised Llama-3 via Unsloth/QLoRA. `ResponderConfig` (frozen dataclass). Zero-shot prompt template forcing 5-section Cloud Security Analyst report. Mock mode for safe interactive use.
* `src/main.py`: **[COMPLETE — Phase 5]** End-to-end pipeline orchestrator.
  * CLI: `--input_file` (raw JSON) + `--deeplog_ckpt` (model weights) + `--mock_llm` (print prompt only).
  * `load_cloudtrail_events()` — loads JSON array or JSONL CloudTrail exports.
  * `event_to_log_string()` — converts CloudTrail dict → Drain-parseable string.
  * `group_events_into_sessions()` — groups by (sourceIPAddress, userIdentity.arn).
  * `load_deeplog_checkpoint()` — reconstructs model from saved hyperparameters.
  * `reverse_map_anomalies()` — translates DeepLog bool flags back to Drain template text + metadata.
  * `run_pipeline()` — runs Drain inline on raw JSON, critical embedding/template safety check, DeepLog detection, reverse-map, Llama-3 report (or prompt print with `--mock_llm`).
* `src/pipeline/`: Integration scripts connecting the Parser, Detector, and Responder.
* `tests/`: Unit tests — 138 tests passing across all completed phases.

## AI Agent Guidelines & Constraints
* **Frameworks:** Use Python 3.10+, PyTorch for neural networks, and Hugging Face `transformers`/`unsloth` for the LLM.
* **Heavy Compute & Hardware Warning:** NEVER attempt to run Llama-3 training or heavy deep learning training loops directly in the terminal during an interactive session. To prevent system freezes or hardware strain (e.g., GPU/display issues under heavy load), strictly use small, mocked datasets (e.g., 5-10 log lines) to verify script logic. Cap VRAM usage explicitly in scripts where possible.
* **Data Structure Optimization:** Ensure the Drain parser utilizes highly efficient string matching and array manipulations to handle the massive JSON payloads quickly.
* **Modularity:** Ensure the DeepLog model's output (integer sequences) can be cleanly reverse-mapped to text templates before being passed to the Llama-3 prompt.
* **Deployment:** Keep the architecture container-friendly. Ensure scripts use environment variables for file paths so they can be seamlessly wrapped in Docker containers.
* **Style:** Use strong typing (Python type hints), PEP8 formatting, and write docstrings for all core functions.

# Tech Stack & Tools
* **Language:** Python
* **Log Parsing:** Drain algorithm (to convert unstructured JSON logs into integer "Log Keys").
* **Anomaly Detection:** DeepLog architecture using an LSTM (Long Short-Term Memory) Recurrent Neural Network.
* **Incident Response:** Llama-3-8b (Large Language Model).
* **LLM Optimization:** QLoRA and Unsloth (strictly required to ensure the model runs locally on a consumer-grade laptop).

# Architecture & Workflow Decisions
* The pipeline strictly flows in this order: Raw AWS JSON Logs -> Drain Parser -> DeepLog LSTM -> If Anomaly Detected -> Reverse map keys to text -> Llama-3 Prompt -> Final Report.
* DeepLog is strictly for mathematical sequence detection (identifying deviations from normal behavior). It does not do semantic reasoning.
* Llama-3 is strictly for semantic reasoning and generating mitigation steps.
* Local execution is a priority. Avoid relying on external cloud APIs for the machine learning models.

## Phase Completion Status
| Phase | Component | Status | Key Output |
|-------|-----------|--------|------------|
| 1 | `src/pipeline/preprocess.py` | ✅ Complete | `train_sessions.csv`, `test_sessions.csv` |
| 2 | `src/parser/drain_parser.py` | ✅ Complete | `log_keys: list[int]`, `templates: dict[int, str]` |
| 3 | `src/detector/` + `src/models/train_deeplog.py` | ✅ Complete | `models/deeplog.pt`, anomaly flags + P/R/F1 |
| 4 | `src/responder/llama_inference.py` | ✅ Complete | 5-section incident report (mock + real Llama-3) |
| 5 | `src/main.py` (end-to-end pipeline) | ✅ Complete | CLI: `--input_file` / `--deeplog_ckpt` / `--mock_llm` |

## Phase 4 — Key Implementation Notes
* **Why Llama-3 after DeepLog (Advisor Skill 2):** DeepLog detects *that* an anomaly occurred (mathematical deviation). Llama-3 explains *what* it means and *how* to remediate — bridging the semantic gap.
* `LlamaResponder`: Lazy model loading (idempotent). Unsloth `FastLanguageModel.from_pretrained()` with `load_in_4bit=True`, `max_seq_length=2048`. VRAM capped via `CUDA_MEMORY_MAX`.
* Prompt: Zero-shot Llama-3 Instruct chat format. System role forces 5-section structure: Incident Summary, Anomaly Analysis, Severity Assessment (Critical/High/Medium/Low), Attack Pattern Classification (MITRE ATT&CK), Recommended Mitigation.
* Mock mode: Template-based report with same 5-section structure. Severity heuristic from anomaly ratio. Attack classification from event names. `[MOCK MODE]` marker appended.
* Dependency guard: `_load_model()` wraps unsloth import in try/except. Mock mode never imports unsloth or allocates GPU memory.
* Tensor shapes: `input_ids: (1, seq_len) int64` → `logits: (1, seq_len, vocab_size) float32` → `output_ids: (1, seq_len + max_new_tokens) int64`.

## Phase 5 — Key Implementation Notes
* CLI: `--input_file` (raw CloudTrail JSON/JSONL), `--deeplog_ckpt` (model weights path), `--mock_llm` (print Llama-3 prompt without loading model).
* `load_cloudtrail_events()`: Handles both `{"Records": [...]}` JSON and line-delimited JSONL.
* `event_to_log_string()`: Formats `"{eventSource} {eventName} {sourceIPAddress} {userIdentity.arn}"` — Drain's regex patterns mask the variable fields.
* `group_events_into_sessions()`: Groups by `(sourceIPAddress, userIdentity.arn)`, consistent with `preprocess.py` session keys.
* `load_deeplog_checkpoint()`: Reads saved hyperparameters (`num_keys`, `embedding_dim`, `hidden_size`, `num_layers`, `dropout`) from `deeplog.pt` and reconstructs the exact model architecture.
* **Critical safety check**: `model._embedding.num_embeddings == len(templates)` — raises `ValueError` with retrain instructions on mismatch.
* `reverse_map_anomalies()`: Pure function. For each session with ≥1 anomaly flag, reverse-maps `sessions[s][i]` → `templates[key]` and slices per-event metadata (timestamps, IPs, ARNs). Unknown keys → `"<unknown key N>"` fallback.
* `top_k` safety: Capped to `min(top_k, num_keys)` before calling `detect_anomalies()` to prevent `torch.topk` overflow.
* `--mock_llm` mode: Imports `_build_prompt()` from `llama_inference.py` and prints the exact chat-format prompt. No model loaded, no GPU memory allocated.
* Env vars: `CUDA_MEMORY_MAX`, `RESPONDER_MODEL`, `RESPONDER_MAX_TOKENS`.

## Phase 3 — Key Implementation Notes
* `DeepLogModel`: Embedding(num_keys, 64) → LSTM(64, 64, 2 layers) → Linear(64, num_keys). All tensor shapes documented inline.
* Training: semi-supervised on normal-only (Root user) sequences. Cross-entropy loss, Adam optimiser.
* Checkpoint safety: mid-training saves every 10 epochs; `num_keys` mismatch guard on resume; embedding size assertion before inference.
* Inference: top-k next-key check (default k=9). Anomaly = true next key not in top-k predictions.
* Evaluation: Precision, Recall, F1 prioritised over accuracy (Advisor Skill 4 — rare-event imbalance).

## ML Advisor & Thesis Mentor Skills
When generating machine learning code (especially PyTorch and model evaluation logic), you must strictly adhere to the following advisor protocols:

* **Skill 1: Tensor Shape Explicit Documentation:** Deep learning bugs are almost always dimension mismatches. For every PyTorch `nn.Module` or `forward()` pass you write, you must include inline comments explicitly stating the expected tensor shapes (e.g., `Input: (batch_size, seq_len, input_dim) -> Output: (batch_size, hidden_size)`).
* **Skill 2: The "Why" Before the "How":** Before writing the code for a complex ML component (like the LSTM architecture or the loss function), write a brief 2-3 sentence markdown explanation of *why* this specific mathematical approach is best for this specific anomaly detection task. 
* **Skill 3: Data Leakage Watchdog:** When splitting training and testing data, or writing the sliding window dataset logic, explicitly verify and state how the code guarantees no target data leaks into the training set.
* **Skill 4: Imbalanced Metric Focus:** When writing evaluation scripts, heavily prioritize Precision, Recall, and the F1-Score over raw Accuracy, acknowledging that cloud anomalies are highly imbalanced rare events.