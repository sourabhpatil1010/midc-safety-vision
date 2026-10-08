# MIDC AI Tracking System — Coding Assistant Context

**How to use this file:** Paste this into your coding assistant's project context (e.g. `CLAUDE.md` for Claude Code, `.cursorrules` for Cursor, or a pinned system message) so it understands the project without re-explaining it every session. Update it as the project evolves — treat it as a living contract between you, your teammate, and the assistant.

---

## 1\. Project Summary

Building an AI camera system for MIDC that solves:

1. Employees skipping required safety gear (PPE).  
2. Inaccurate attendance (marked times don't match actual presence).  
3. Productivity tracking via movement between work zones.  
4. Unique employee IDs tied to role, required PPE, and assigned zone.

Two people are working on this project. Code should be written so either person can pick up any file and understand it without the original author present.

---

## 2\. Approach — Read This Before Suggesting Alternatives

**We are using Approach A: fully local, self-managed fine-tuning.** Do not suggest hosted training platforms (Roboflow's hosted training, Azure Custom Vision, AutoML) as the primary training method — that was considered and explicitly rejected in favor of local control. Roboflow is used ONLY to download pre-labeled datasets; training itself happens locally via `ultralytics`.

Do not suggest cloud vision APIs (AWS Rekognition, Azure Face API, Google Vision) for face recognition or PPE detection — this is a deliberate choice to keep all employee biometric data and video processing local, for privacy and cost reasons. Do not suggest using an LLM (GPT-4V, Claude vision) for PPE detection or face matching — too slow and expensive for real-time video; LLMs are only being considered for a later reporting/alerting layer, not detection.

---

## 3\. Tech Stack — Use Only These, Unless Explicitly Asked to Change

### Computer Vision

- **PPE detection:** YOLOv8 (Ultralytics), fine-tuned locally  
- **Face recognition:** InsightFace (ArcFace), pre-trained, no training — enrollment-based  
- **Tracking:** ByteTrack (bundled in Ultralytics, used via `yolo detect track`)  
- **Video/stream handling:** OpenCV

### Backend (not yet built)

- **API:** FastAPI  
- **Database:** PostgreSQL  
- **ORM:** SQLAlchemy  
- **Validation:** Pydantic

### Agent/Intelligence layer (not yet started — confirm scope before building)

- **Orchestration:** LangGraph  
- **RAG:** LangChain \+ Chroma or FAISS  
- **LLM:** Claude or GPT API (not finalized)

### Frontend (not yet built)

- **Dashboard:** React  
- **API calls:** Axios

### Dev tooling — do not deviate from these

- **Package/env manager:** `uv` — always use `uv add`, `uv run`, never raw `pip install` or `python` directly  
- **Python version:** 3.12, pinned via `.python-version` — do not suggest 3.13 (torchaudio and some other packages lack wheels for it; we hit this already)  
- **GPU stack:** PyTorch 2.5.1+cu121, onnxruntime-gpu pinned `<1.27` (see gotcha below)

---

## 4\. Known Gotchas — Don't Re-Suggest These Mistakes

1. **`uv init` defaults to the newest Python** (3.13 at time of writing), which breaks `torchaudio` (no CUDA wheels for cp313). We don't use `torchaudio` at all — this project is vision-only, no audio. Python is pinned to 3.12.  
     
2. **`onnxruntime-gpu` versions ≥1.27 default to expecting CUDA 13**, but our PyTorch build uses CUDA 12.1. This causes a silent CPU fallback with DLL errors (`cublasLt64_13.dll not found`). Fixed by pinning `onnxruntime-gpu<1.27` (resolves to 1.26.0), which matches PyTorch's bundled CUDA 12.1 libraries. Do not suggest installing a system-wide CUDA Toolkit — it's unnecessary; PyTorch's bundled DLLs are reused automatically once the version is aligned.  
     
3. **`--index-url` vs `--default-index` with `uv add`:** using `--index-url` when adding torch replaces the *entire project's* default package index, breaking installation of every other package (we hit this with `ultralytics` failing to resolve). Always use `--default-index` for a one-off index override, or configure `[[tool.uv.index]]` with `explicit = true` in `pyproject.toml` scoped to specific packages via `[tool.uv.sources]`.  
     
4. **GTX 1650 (4GB VRAM) constraints:** stick to `yolov8n` or `yolov8s` for training and inference. `yolov8m` and larger will not fit reliably. If suggesting training commands, default to `batch=4` to `batch=8`, `imgsz=640`, and mention `amp=False` as a fallback if training produces NaN losses (16-series cards can be unstable with mixed precision).

---

## 5\. File Structure — Put New Files in the Right Place

midc-ai-tracking/

├── ai\_models/

│   ├── ppe\_detection/        \# YOLOv8 training \+ inference code

│   └── face\_recognition/     \# InsightFace enrollment \+ matching code

├── datasets/                 \# Downloaded Roboflow datasets (gitignored — large files)

├── data/

│   ├── employee\_photos/      \# Reference photos for enrollment

│   └── embeddings/           \# Stored face embeddings (employees.pkl)

├── scripts/                  \# One-off utility scripts (dataset download, bulk enrollment, etc.)

├── runs/                     \# YOLO's auto-generated output (predictions, training runs) — gitignored

├── backend/                  \# FastAPI app (not yet built)

│   ├── routers/               \# One file per resource: employees, attendance, violations, productivity

│   ├── models/                 \# SQLAlchemy models

│   └── schemas/                \# Pydantic schemas

├── dashboard/                 \# React app (separate Node.js project, not yet built)

└── pyproject.toml / uv.lock

When adding a new feature, put code in the existing folder that matches its purpose — don't create new top-level folders without a clear reason tied to this structure.

---

## 6\. Database Schema (planned — implement against this when building backend)

employees        (employee\_id PK, name, role, assigned\_zone, required\_ppe, face\_embedding)

attendance\_logs  (id PK, employee\_id FK, in\_time, out\_time, camera\_id)

violation\_logs   (id PK, employee\_id FK, violation\_type, timestamp, snapshot\_url, zone\_id)

zone\_logs        (id PK, employee\_id FK, zone\_id, entry\_time, exit\_time)

---

## 7\. Coding Conventions

- **Python style:** follow PEP 8, type hints on function signatures, docstrings on non-trivial functions  
- **Config over hardcoding:** zone definitions, PPE-per-role requirements, and confidence thresholds belong in config files (e.g. `config/zones.yaml`, `config/ppe_requirements.yaml`), not hardcoded in scripts — these will change as MIDC's layout/roles evolve  
- **No secrets in code:** Roboflow API keys, future DB credentials, and LLM API keys go in a `.env` file (gitignored), loaded via `python-dotenv` — never hardcoded or committed  
- **GPU device handling:** always make `device` a parameter/config value (e.g. `device=0`), not hardcoded, since production hardware will differ from the dev GTX 1650  
- **Commit messages / PRs:** since two people work on this, keep changes scoped — one feature/fix per commit, clear enough for the other person to understand without asking

---

## 8\. Current Status (keep updated)

| Item | Status |
| :---- | :---- |
| Dev environment (torch, ultralytics, roboflow) | Done, GPU confirmed |
| Face recognition environment (insightface, onnxruntime-gpu) | Done, GPU confirmed |
| Tracking (ByteTrack) | Tested on webcam |
| Face enrollment script | Written, needs full test run |
| PPE dataset | Not yet downloaded |
| PPE model training | Blocked on dataset |
| Backend (FastAPI \+ PostgreSQL) | Not started |
| Agent/RAG layer | Not started — scope unconfirmed |
| Dashboard (React) | Not started |

---

## 9\. What NOT to Do

- Don't suggest switching to a hosted/cloud training or inference service for PPE or face recognition — this was a deliberate decision, not an oversight  
- Don't suggest Python 3.13 or later without checking package compatibility first  
- Don't hardcode zone names, employee counts, or camera counts — these are not finalized  
- Don't add `torchaudio` or any audio-processing dependency — not needed for this project  
- Don't suggest a different package manager (pip/poetry/conda) — this project uses `uv` exclusively  
- Don't skip the `device=0` / GPU-provider parameterization "for simplicity" — production will run on different, likely multiple, GPUs