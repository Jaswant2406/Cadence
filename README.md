# CADENCE Security Analytics

CADENCE is a local-first behavioral security analytics workspace for analyzing
enterprise access telemetry. It combines adaptive entity baselines, a
conformal GRU autoencoder, Isolation Forest, MITRE ATT&CK mapping, SHAP
explanations, and network context in a responsive analyst dashboard.

The application can analyze an uploaded CSV or generate a reproducible
synthetic dataset internally. Ground-truth labels are retained strictly for
post-inference evaluation and never influence model training, alert
generation, attack-category prediction, risk scoring, or explanations.

## Key capabilities

- Validate and preview uploaded telemetry before analysis.
- Generate realistic synthetic enterprise activity when no CSV is supplied.
- Build per-entity behavioral profiles and five-event sequence windows.
- Detect anomalous behavior with two independent models:
  - Conformal GRU autoencoder
  - Isolation Forest
- Calculate evidence-based confidence, risk, and severity for every alert.
- Map behavioral alerts to supported MITRE ATT&CK techniques.
- Explain high-scoring Isolation Forest events with SHAP.
- Enrich alerts with offline network trust and history context.
- Explore alerts, telemetry, entities, model results, and methodology.
- Use a responsive sidebar that collapses on desktop and becomes a mobile
  drawer with backdrop and Escape-key support.

## Application workflow

```text
CSV upload or synthetic generation
              |
              v
Schema and sequence validation
              |
              v
Adaptive entity behavior profiles
              |
              v
Five-event sequence construction
              |
              v
Conformal GRU + Isolation Forest
              |
              v
Dynamic confidence and risk scoring
              |
              v
MITRE mapping + SHAP + network context
              |
              v
Analyst dashboard and evaluation views
```

## Alert scoring

CADENCE does not assign a fixed confidence or severity to an attack category.
Each alert is scored from its actual model evidence.

Raw GRU reconstruction errors and Isolation Forest anomaly scores are
normalized for the active run. The resulting alert evidence includes:

- normalized anomaly strength;
- conformal GRU score;
- Isolation Forest score;
- agreement or disagreement between the models; and
- consistency between the two model scores.

Prediction confidence is calculated dynamically:

```text
confidence =
    0.40 × anomaly strength
  + 0.20 × conformal score
  + 0.20 × Isolation Forest score
  + 0.20 × model-score consistency
  + agreement bonus
```

The final risk score is bounded to `0–100`:

```text
risk =
    0.30 × anomaly strength
  + 0.20 × prediction confidence
  + 0.20 × conformal score
  + 0.20 × Isolation Forest score
  + 0.10 × model agreement
```

Severity is derived consistently from risk and agreement:

| Severity | Rule |
|---|---|
| Critical | Both models agree and risk is at least 80 |
| High | Risk is at least 65 |
| Medium | Risk is at least 58 |
| Low | Risk is below 58 |

Risk is a prioritization signal, not proof of compromise.

## Supported behavioral categories

| Behavior | MITRE ATT&CK | Technique |
|---|---|---|
| Brute Force | T1110 | Brute Force |
| Credential Stuffing | T1110.004 | Credential Stuffing |
| Impossible Travel | T1078 | Valid Accounts |
| Insider Drift | T1078 | Valid Accounts |
| Lateral Movement | T1021 | Remote Services |
| Device Spoofing | T1036 | Masquerading |
| Low-and-Slow Exfiltration | T1041 | Exfiltration Over C2 Channel |

The Overview selects the highest-risk representative from distinct categories
before filling remaining positions, preventing the top-alert panel from being
dominated by one behavior.

## Overview metrics

All Overview values are derived from the complete run snapshot:

| Metric | Definition |
|---|---|
| Telemetry events | Total rows in the analyzed dataset |
| Detected alerts | Sequence windows flagged by either detector |
| Ground-truth anomaly rate | Ground-truth anomalous events divided by all dataset events |
| Adaptive threshold | Calibrated conformal GRU threshold |
| Average confidence | Mean prediction confidence across detected alerts |
| Ensemble precision | `TP / (TP + FP)` for the combined detector against sequence labels |
| Observed false-positive rate | `FP / (FP + TN)` across labeled normal windows |
| Models agreeing | Alerts flagged by both models divided by all detected alerts |

The ground-truth anomaly rate is:

```text
anomaly rate = ground-truth anomalous events / total dataset events × 100
```

Metrics requiring labels display as unavailable when the uploaded dataset does
not contain a complete `label` column.

## CSV input

Uploads must be UTF-8 CSV files no larger than 10 MB or 25,000 rows. Download
the current template from the start screen or:

```text
GET /api/imports/template
```

Required columns:

```text
entity_id
entity_type
timestamp
source_ip
geo_location
resource_accessed
auth_method
session_duration
command_sequence
device_fingerprint
```

Supported `entity_type` values:

```text
user
service_account
edge_device
```

The optional `label` column accepts `Normal`, `Benign`, or one of the seven
supported behavioral categories. If any label is missing, evaluation labels
are ignored for that upload.

At least three five-event entity windows must be available for GRU training.
Duplicate rows are reported and retained.

## Technology

### Backend

- Python
- FastAPI
- pandas and NumPy
- PyTorch
- scikit-learn
- SHAP

### Frontend

- React 19
- TypeScript
- Vite
- React Router
- TanStack Query
- Recharts
- Lucide icons

The frontend uses a maintained CSS design system rather than Tailwind. The
sidebar supports a 240px expanded state, an 80px icon-only desktop state with
tooltips, and a 240px mobile drawer.

## Prerequisites

- Python 3.14+
- Node.js 20+
- npm 10+

## Installation

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend
npm install
```

The Python requirements use the official CPU-only PyTorch package index.

## Run locally

Start the API from the repository root:

```bash
.venv/bin/uvicorn cadence_api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm run dev
```

Open:

- Dashboard: `http://127.0.0.1:5173`
- API documentation: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/api/health`

Choose **Generate Dataset** for a complete analysis without uploading a CSV.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service health and latest run |
| GET | `/api/imports/template` | Download the CSV template |
| POST | `/api/imports/csv` | Validate and stage a CSV |
| GET | `/api/imports/{import_id}` | Read an import result |
| POST | `/api/runs` | Start a generated or CSV-backed analysis |
| GET | `/api/runs/latest` | Read the latest run |
| GET | `/api/runs/{run_id}` | Read run status |
| GET | `/api/runs/{run_id}/summary` | Overview data |
| GET | `/api/runs/{run_id}/alerts` | Search and filter alerts |
| GET | `/api/runs/{run_id}/alerts/{alert_id}` | Alert investigation |
| GET | `/api/runs/{run_id}/entities/{entity_id}` | Entity history |
| GET | `/api/runs/{run_id}/telemetry` | Paginated telemetry |
| GET | `/api/runs/{run_id}/models` | Evaluation and model metadata |

## Verification

Backend tests:

```bash
.venv/bin/python -m pytest -q
```

Full backend QA:

```bash
.venv/bin/python qa_backend_run.py
```

Frontend checks:

```bash
cd frontend
npm run typecheck
npm test
npm run build
```

## Project structure

```text
.
├── app.py                  # Generation, profiling, models, scoring, mapping
├── cadence_api/            # FastAPI routes, validation, storage, views
├── network_intelligence/   # Offline network enrichment
├── frontend/               # React analyst console
├── tests/                  # Automated backend tests
├── qa_backend_run.py       # Full backend QA harness
├── requirements.txt
└── README.md
```

## Operational notes

- Models retrain locally for each analysis run.
- Uploaded data and completed snapshots are held in memory.
- Restarting the API clears imports and completed runs.
- Labels are evaluation-only and are not exposed as analyst evidence unless
  the evaluation context is explicitly enabled.
- SHAP explanations are generated for the highest-scoring Isolation Forest
  events when SHAP is available.
- The application is an analytical prototype and should not be treated as an
  autonomous incident-response system.
