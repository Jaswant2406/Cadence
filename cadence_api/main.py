from __future__ import annotations

import csv
import io
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .csv_validation import MAX_FILE_BYTES, REQUIRED_COLUMNS, validate_csv_bytes
from .models import ImportResponse, PaginatedResponse, RunCreate, RunResponse
from .store import store
from .views import page


app = FastAPI(
    title="CADENCE Security Analytics API",
    description="Local API for behavioral anomaly detection and analyst dashboards.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def completed_snapshot(run_id: str) -> Dict[str, Any]:
    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.status != "completed" or run.snapshot is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run is {run.status}; analysis data is not available yet.",
        )
    return run.snapshot


@app.get("/api/health")
def health() -> Dict[str, Any]:
    latest = store.latest()
    return {
        "status": "ok",
        "service": "cadence-api",
        "latest_run": latest.public() if latest else None,
    }


@app.get("/api/imports/template")
def csv_template() -> StreamingResponse:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REQUIRED_COLUMNS)
    writer.writeheader()
    for hour in range(9, 17):
        writer.writerow(
            {
                "entity_id": "ENT-0001",
                "entity_type": "user",
                "timestamp": f"2026-07-01 {hour:02d}:00:00",
                "source_ip": "103.48.1.10",
                "geo_location": "New York, US",
                "resource_accessed": "sso://portal/login",
                "auth_method": "sso",
                "session_duration": str(16 + hour),
                "command_sequence": "login;open_app;logout",
                "device_fingerprint": "device-a1",
            }
        )
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cadence-template.csv"'},
    )


@app.post("/api/imports/csv", response_model=ImportResponse)
async def import_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only .csv files are supported.")
    payload = await file.read(MAX_FILE_BYTES + 1)
    validation = validate_csv_bytes(payload)
    response = {
        "filename": filename,
        "valid": validation.valid,
        "summary": validation.summary,
        "preview": validation.preview,
        "errors": validation.errors,
        "warnings": validation.warnings,
    }
    artifact = store.add_import(filename, validation.dataframe, response)
    return {"import_id": artifact.import_id, **response}


@app.get("/api/imports/{import_id}", response_model=ImportResponse)
def get_import(import_id: str) -> Dict[str, Any]:
    artifact = store.imports.get(import_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Import was not found.")
    return {"import_id": import_id, **artifact.response}


@app.post("/api/runs", response_model=RunResponse, status_code=202)
def create_run(request: RunCreate) -> Dict[str, Any]:
    if request.source == "csv" and not request.import_id:
        raise HTTPException(status_code=422, detail="import_id is required for CSV runs.")
    try:
        return store.start_run(request.source, request.import_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/latest", response_model=Optional[RunResponse])
def latest_run() -> Optional[Dict[str, Any]]:
    latest = store.latest()
    return latest.public() if latest else None


@app.get("/api/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> Dict[str, Any]:
    try:
        return store.get_run(run_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/summary")
def get_summary(run_id: str) -> Dict[str, Any]:
    return completed_snapshot(run_id)["summary"]


@app.get("/api/runs/{run_id}/alerts", response_model=PaginatedResponse)
def get_alerts(
    run_id: str,
    page_number: int = Query(1, alias="page", ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = "",
    severity: str = "",
    technique: str = "",
    min_risk: float = Query(0, ge=0, le=100),
) -> Dict[str, Any]:
    alerts = completed_snapshot(run_id)["alerts"]
    needle = search.strip().lower()
    filtered = [
        alert
        for alert in alerts
        if float(alert["combined_risk_score"]) >= min_risk
        and (not severity or alert["severity"].lower() == severity.lower())
        and (not technique or alert["technique_id"].lower() == technique.lower())
        and (
            not needle
            or needle in str(alert["entity_id"]).lower()
            or needle in str(alert["source_ip"]).lower()
            or needle in str(alert["resource_accessed"]).lower()
            or needle in str(alert["technique_name"]).lower()
        )
    ]
    return page(filtered, page_number, page_size)


@app.get("/api/runs/{run_id}/alerts/{alert_id}")
def get_alert(run_id: str, alert_id: str) -> Dict[str, Any]:
    snapshot = completed_snapshot(run_id)
    alert = next(
        (item for item in snapshot["alerts"] if item["alert_id"] == alert_id),
        None,
    )
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert was not found.")
    entity_events = [
        event for event in snapshot["telemetry"] if event["entity_id"] == alert["entity_id"]
    ]
    position = next(
        (
            index
            for index, event in enumerate(entity_events)
            if event["event_index"] == alert["event_index"]
        ),
        len(entity_events) - 1,
    )
    return {
        **alert,
        "surrounding_events": entity_events[max(0, position - 5) : position + 6],
        "fallback_factors": [
            {
                "name": "New device",
                "active": bool(alert["new_device_indicator"]),
            },
            {
                "name": "IP subnet changed",
                "active": bool(alert["ip_subnet_change"]),
            },
            {
                "name": "Elevated login frequency",
                "value": alert["login_frequency"],
            },
            {
                "name": "Time since previous login",
                "value": alert["time_since_previous_login"],
            },
        ],
    }


@app.get("/api/runs/{run_id}/entities/{entity_id}")
def get_entity(run_id: str, entity_id: str) -> Dict[str, Any]:
    snapshot = completed_snapshot(run_id)
    events = [
        event for event in snapshot["telemetry"] if event["entity_id"] == entity_id
    ]
    if not events:
        raise HTTPException(status_code=404, detail="Entity was not found.")
    features = [
        row for row in snapshot["engineered"] if row["entity_id"] == entity_id
    ]
    alerts = [
        alert for alert in snapshot["alerts"] if alert["entity_id"] == entity_id
    ]
    recent = features[-1] if features else {}
    return {
        "entity_id": entity_id,
        "entity_type": events[-1]["entity_type"],
        "event_count": len(events),
        "alert_count": len(alerts),
        "max_risk": max(
            (float(alert["combined_risk_score"]) for alert in alerts),
            default=0,
        ),
        "profile": {
            "preferred_login_hour": recent.get("preferred_login_hour"),
            "typical_session_duration": recent.get("typical_session_duration"),
            "common_ip_subnet": recent.get("common_ip_subnet"),
            "frequently_accessed_resources": recent.get(
                "frequently_accessed_resources", []
            ),
        },
        "resources": [
            {"name": name, "count": count}
            for name, count in _counts(event["resource_accessed"] for event in events)
        ],
        "auth_methods": [
            {"name": name, "count": count}
            for name, count in _counts(event["auth_method"] for event in events)
        ],
        "events": events,
        "alerts": alerts,
        "network_history": [
            {
                "event_index": event["event_index"],
                "timestamp": event["timestamp"],
                "source_ip": event["source_ip"],
                "network": snapshot["network_by_event"].get(event["event_index"]),
            }
            for event in events
        ],
    }


def _counts(values: Any) -> list[tuple[str, int]]:
    from collections import Counter

    return Counter(str(value) for value in values).most_common(8)


@app.get("/api/runs/{run_id}/telemetry", response_model=PaginatedResponse)
def get_telemetry(
    run_id: str,
    page_number: int = Query(1, alias="page", ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: str = "",
    evaluation: bool = False,
) -> Dict[str, Any]:
    telemetry = completed_snapshot(run_id)["telemetry"]
    needle = search.strip().lower()
    filtered = [
        event
        for event in telemetry
        if not needle
        or needle in str(event["entity_id"]).lower()
        or needle in str(event["source_ip"]).lower()
        or needle in str(event["resource_accessed"]).lower()
        or needle in str(event["command_sequence"]).lower()
    ]
    if not evaluation:
        filtered = [
            {key: value for key, value in event.items() if key != "label"}
            for event in filtered
        ]
    return page(filtered, page_number, page_size)


@app.get("/api/runs/{run_id}/models")
def get_models(run_id: str) -> Dict[str, Any]:
    return completed_snapshot(run_id)["models"]
