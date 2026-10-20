 🚀 Live Demo

**CADENCE — Security Analytics API**

Explore the deployed application and its API documentation.

- 🌐 **Live Application:** [Open CADENCE](https://cadence-ik3o.onrender.com)
- 📚 **API Documentation:** [Swagger UI](https://cadence-ik3o.onrender.com/docs)


title: CADENCE Security Analytics
emoji: shield
colorFrom: slate
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CADENCE Security Analytics

CADENCE is a behavioral security analytics workspace for analyzing enterprise
access telemetry. It combines adaptive entity baselines, a conformal GRU
autoencoder, Isolation Forest, MITRE ATT&CK mapping, SHAP explanations, and
network context in a responsive analyst dashboard.

This repository is configured for deployment as a Hugging Face Docker Space.

## What the Space does

- Validates and previews uploaded telemetry before analysis.
- Generates reproducible synthetic telemetry when no CSV is supplied.
- Builds per-entity behavioral profiles and five-event sequence windows.
- Detects anomalous behavior with a conformal GRU autoencoder and Isolation Forest.
- Scores alerts with dynamic evidence-based confidence and risk.
- Maps alerts to supported MITRE ATT&CK techniques.
- Serves the React frontend and FastAPI backend from one Space.

## Local development

Backend:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn cadence_api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Hugging Face deployment

1. Create a new Hugging Face Space.
2. Choose `Docker` as the Space SDK.
3. Push this repository to the Space.
4. Hugging Face will build the Docker image and expose the app on port `7860`.

The Docker build:

- installs Python dependencies from `requirements.txt`;
- installs frontend dependencies from `frontend/package-lock.json`;
- builds the Vite app into `frontend/dist`; and
- runs `uvicorn cadence_api.main:app` on the Space-provided port.

## Runtime notes

- The Space serves the frontend and API from the same origin.
- Uploaded data and completed run snapshots are held in memory.
- Restarting the Space clears imports and completed runs.
- SHAP explanations are generated when the dependency is available.
- This project is an analytical prototype, not an autonomous response system.
