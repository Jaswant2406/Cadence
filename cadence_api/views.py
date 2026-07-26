from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from network_intelligence import NetworkIntelligenceEngine


def native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [native(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [native(item) for item in value]
    return str(value)


def build_snapshot(result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    telemetry_frame = result["telemetry_data"].copy().reset_index(drop=True)
    engineered = result["engineered_features"].copy().reset_index(drop=True)
    telemetry_frame.insert(0, "event_index", np.arange(len(telemetry_frame)))
    telemetry = native(telemetry_frame)

    network_by_event: Dict[int, Dict[str, Any]] = {}
    network_engine = NetworkIntelligenceEngine()
    ordered = telemetry_frame.copy()
    ordered["_parsed_timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    for _, row in ordered.sort_values("_parsed_timestamp").iterrows():
        try:
            network_by_event[int(row["event_index"])] = native(
                network_engine.analyze(
                    {"user_id": str(row["entity_id"]), "ip_address": str(row["source_ip"])}
                )
            )
        except Exception:
            network_by_event[int(row["event_index"])] = {
                "network_type": "Unknown",
                "network_trust_score": None,
                "reasons": ["Network enrichment unavailable"],
            }

    shap = result["shap_results"]
    shap_by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if shap.get("available") and not shap["explanations"].empty:
        for row in shap["explanations"].to_dict(orient="records"):
            metadata = {"incident_index", "entity_id", "anomaly_score"}
            contributors = sorted(
                (
                    {"feature": key, "value": float(value)}
                    for key, value in row.items()
                    if key not in metadata and pd.notna(value)
                ),
                key=lambda item: abs(item["value"]),
                reverse=True,
            )[:6]
            shap_by_entity[str(row["entity_id"])].append(
                {
                    "incident_index": int(row["incident_index"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "contributors": contributors,
                }
            )

    parsed_times = pd.to_datetime(telemetry_frame["timestamp"], errors="coerce")
    alerts: List[Dict[str, Any]] = []
    mitre_frame = result["mitre_results"]
    for index, incident in enumerate(mitre_frame.to_dict(orient="records")):
        entity_id = str(incident["entity_id"])
        window_end = pd.to_datetime(incident["window_end"], errors="coerce")
        candidates = telemetry_frame[
            (telemetry_frame["entity_id"].astype(str) == entity_id)
            & (parsed_times <= window_end)
        ]
        event = candidates.iloc[-1] if not candidates.empty else None
        event_index = int(event["event_index"]) if event is not None else -1
        explanation_options = shap_by_entity.get(entity_id, [])
        explanation = explanation_options[0] if explanation_options else None
        alert = {
            **native(incident),
            "alert_id": f"{run_id}-{index + 1:05d}",
            "event_index": event_index,
            "source_ip": native(event["source_ip"]) if event is not None else None,
            "device_fingerprint": native(event["device_fingerprint"]) if event is not None else None,
            "network": network_by_event.get(event_index),
            "explanation": explanation,
            "model_agreement": bool(incident["gru_prediction"] and incident["if_prediction"]),
        }
        alerts.append(alert)
    alerts.sort(key=lambda item: item["combined_risk_score"], reverse=True)

    severity_counts = Counter(alert["severity"] for alert in alerts)
    technique_counts = Counter(alert["technique_name"] for alert in alerts)
    agreement_counts = Counter(
        "Both models" if alert["model_agreement"] else
        "GRU only" if alert["gru_prediction"] else "Isolation Forest only"
        for alert in alerts
    )
    risk_by_day: Dict[str, List[float]] = defaultdict(list)
    for alert in alerts:
        key = str(alert["window_end"])[:10]
        risk_by_day[key].append(float(alert["combined_risk_score"]))

    labels_available = "label" in telemetry_frame.columns
    benign = telemetry_frame["label"].astype(str).str.lower().isin(["normal", "benign"]) if labels_available else None
    average_confidence = (
        round(float(np.mean([alert["prediction_confidence"] for alert in alerts])), 2)
        if alerts else 0.0
    )
    model_agreement_rate = (
        round(100.0 * sum(alert["model_agreement"] for alert in alerts) / len(alerts), 2)
        if alerts else 0.0
    )
    detection_precision = None
    false_positive_rate = None
    sequence_metadata = result["gru_predictions"].get("sequence_metadata")
    if (
        labels_available
        and isinstance(sequence_metadata, pd.DataFrame)
        and sequence_metadata["sequence_label"].notna().all()
    ):
        ground_truth = sequence_metadata["sequence_label"].to_numpy(dtype=bool)
        gru_flags = np.asarray(result["gru_predictions"]["anomaly_prediction"], dtype=bool)
        if_flags = np.asarray(
            result["isolation_forest_predictions"]["sequence_level"]["predictions"],
            dtype=bool,
        )
        ensemble_flags = gru_flags | if_flags
        true_positives = int(np.sum(ensemble_flags & ground_truth))
        false_positives = int(np.sum(ensemble_flags & ~ground_truth))
        true_negatives = int(np.sum(~ensemble_flags & ~ground_truth))
        predicted_positives = true_positives + false_positives
        actual_negatives = true_negatives + false_positives
        detection_precision = round(
            100.0 * true_positives / predicted_positives, 2
        ) if predicted_positives else 0.0
        false_positive_rate = round(
            100.0 * false_positives / actual_negatives, 2
        ) if actual_negatives else 0.0

    diverse_top_alerts: List[Dict[str, Any]] = []
    selected_alert_ids = set()
    represented_categories = set()
    for alert in alerts:
        category = alert["predicted_attack_category"]
        if category in represented_categories:
            continue
        diverse_top_alerts.append(alert)
        selected_alert_ids.add(alert["alert_id"])
        represented_categories.add(category)
        if len(diverse_top_alerts) == 6:
            break
    if len(diverse_top_alerts) < 6:
        diverse_top_alerts.extend(
            alert for alert in alerts
            if alert["alert_id"] not in selected_alert_ids
        )
        diverse_top_alerts = diverse_top_alerts[:6]

    summary = {
        "total_events": int(len(telemetry_frame)),
        "total_entities": int(telemetry_frame["entity_id"].nunique()),
        "total_alerts": len(alerts),
        "high_risk_alerts": sum(
            alert["severity"] in {"Critical", "High"} for alert in alerts
        ),
        "labels_available": labels_available,
        "anomaly_rate": round(float((~benign).mean()) * 100.0, 2) if benign is not None else None,
        "average_confidence": average_confidence,
        "detection_precision": detection_precision,
        "false_positive_rate": false_positive_rate,
        "model_agreement_rate": model_agreement_rate,
        "adaptive_threshold": native(result["adaptive_threshold"]),
        "severity_distribution": [
            {"name": severity, "value": severity_counts.get(severity, 0)}
            for severity in ["Critical", "High", "Medium", "Low"]
        ],
        "technique_distribution": [
            {"name": name, "value": value}
            for name, value in technique_counts.most_common()
        ],
        "model_agreement": [
            {"name": name, "value": value}
            for name, value in agreement_counts.items()
        ],
        "risk_timeline": [
            {
                "date": day,
                "alerts": len(scores),
                "average_risk": round(float(np.mean(scores)), 2),
                "max_risk": round(float(np.max(scores)), 2),
            }
            for day, scores in sorted(risk_by_day.items())
        ],
        "top_alerts": diverse_top_alerts,
    }

    model_metrics = native(result["evaluation_metrics"])
    models = {
        "metrics": model_metrics,
        "labels_available": labels_available,
        "adaptive_threshold": native(result["adaptive_threshold"]),
        "shap": {
            "available": bool(shap.get("available")),
            "reason": shap.get("reason"),
            "explained_incidents": int(len(shap.get("explanations", []))),
        },
    }
    return {
        "summary": summary,
        "alerts": alerts,
        "telemetry": telemetry,
        "engineered": native(engineered),
        "models": models,
        "network_by_event": network_by_event,
    }


def page(items: Iterable[Dict[str, Any]], page_number: int, page_size: int) -> Dict[str, Any]:
    materialized = list(items)
    total = len(materialized)
    pages = max(1, math.ceil(total / page_size))
    page_number = min(max(1, page_number), pages)
    start = (page_number - 1) * page_size
    return {
        "items": materialized[start : start + page_size],
        "page": page_number,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    }
