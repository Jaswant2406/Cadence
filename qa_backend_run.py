from __future__ import annotations

import json
import time
import tracemalloc
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import app


def _timer() -> float:
    return time.perf_counter()


def _record_result(
    name: str,
    start: float,
    passed: bool,
    warnings_list: List[str] | None = None,
    details: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    return {
        "status": "PASS" if passed else "FAIL",
        "execution_time_sec": round(time.perf_counter() - start, 4),
        "warnings": warnings_list or [],
        "details": details or {},
        "error": error,
    }


def run_qa() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    tracemalloc.start()
    overall_start = _timer()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        start = _timer()
        try:
            tg = app.TelemetryGenerator(app.CONFIG)
            telemetry = tg.generate()
            attack_ratio = float((telemetry["label"].astype(str).str.lower() != "normal").mean())
            attack_counts = telemetry["label"].value_counts().to_dict()
            configured_attacks = max(
                len(app.TelemetryGenerator.ATTACK_LABELS),
                int(round(app.CONFIG["N_LOGS"] * app.CONFIG["ATTACK_RATE"])),
            )
            minimum_expected_per_category = max(
                1,
                configured_attacks // len(app.TelemetryGenerator.ATTACK_LABELS),
            )
            required_columns = [
                "entity_id",
                "entity_type",
                "timestamp",
                "source_ip",
                "geo_location",
                "resource_accessed",
                "auth_method",
                "session_duration",
                "command_sequence",
                "device_fingerprint",
                "label",
            ]
            passed = (
                len(telemetry) == app.CONFIG["N_LOGS"]
                and abs(attack_ratio - app.CONFIG["ATTACK_RATE"]) <= 0.005
                and all(column in telemetry.columns for column in required_columns)
                and min(
                    attack_counts.get(label, 0) for label in app.TelemetryGenerator.ATTACK_LABELS
                ) >= minimum_expected_per_category
            )
            results["TelemetryGenerator"] = _record_result(
                "TelemetryGenerator",
                start,
                passed,
                details={
                    "rows": int(len(telemetry)),
                    "attack_ratio": attack_ratio,
                    "attack_counts": attack_counts,
                },
            )
        except Exception as exc:
            results["TelemetryGenerator"] = _record_result(start=start, name="TelemetryGenerator", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            engine = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"])
            behaviour = engine.fit_transform(telemetry)
            profiles = engine.get_entity_profiles()
            cold_df = pd.DataFrame(
                [
                    {
                        "entity_id": "COLD-1",
                        "entity_type": "user",
                        "timestamp": "2026-07-01 09:00:00",
                        "source_ip": "10.10.10.10",
                        "geo_location": "New York, US",
                        "resource_accessed": "share://sales/regional-forecast",
                        "auth_method": "sso",
                        "session_duration": 15,
                        "command_sequence": "login;open_app;logout",
                        "device_fingerprint": "cold-device",
                        "label": "Normal",
                    }
                ]
            )
            cold_prepared = engine._prepare_dataframe(cold_df)
            cold_features = engine._engineer_features(cold_prepared, update_profiles=False)
            cohort_preferred = float(engine.cohort_profiles["user"]["preferred_login_hour"])
            cold_preferred = float(cold_features.iloc[0]["preferred_login_hour"])
            passed = (
                bool(profiles)
                and not cold_features.isna().any().any()
                and behaviour["sequence_windows"].shape[1] == app.CONFIG["WINDOW_SIZE"]
                and np.isclose(cold_preferred, cohort_preferred)
            )
            results["AdaptiveBehaviourContextEngine"] = _record_result(
                "AdaptiveBehaviourContextEngine",
                start,
                passed,
                details={
                    "profile_count": len(profiles),
                    "flat_shape": tuple(behaviour["flat_features"].shape),
                    "sequence_shape": tuple(behaviour["sequence_windows"].shape),
                    "cold_start_preferred_login_hour": cold_preferred,
                    "cohort_preferred_login_hour": cohort_preferred,
                },
            )
        except Exception as exc:
            results["AdaptiveBehaviourContextEngine"] = _record_result(start=start, name="AdaptiveBehaviourContextEngine", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            backend = app.BackendEngine(config=app.CONFIG)
            prepared = engine._prepare_dataframe(telemetry)
            sequence_metadata = backend._build_sequence_metadata(prepared)
            sequence_labels = sequence_metadata["sequence_label"].to_numpy(dtype=int)
            sequence_windows = behaviour["sequence_windows"]
            benign_sequences = sequence_windows[sequence_labels == 0]
            gru = app.ConformalGRUAutoencoder(
                input_dim=int(sequence_windows.shape[2]),
                window_size=app.CONFIG["WINDOW_SIZE"],
                target_fpr=app.CONFIG["TARGET_FPR"],
                seed=app.CONFIG["SEED"],
                epochs=5,
            )
            gru.fit(benign_sequences)
            gru_predictions = gru.predict(sequence_windows)
            expected_threshold = float(np.quantile(gru.calibration_errors_, 1.0 - app.CONFIG["TARGET_FPR"], method="higher"))
            passed = (
                gru.threshold_ is not None
                and np.isclose(float(gru.threshold_), expected_threshold)
                and np.all((gru_predictions["risk_score"] >= 0) & (gru_predictions["risk_score"] <= 100))
            )
            results["ConformalGRUAutoencoder"] = _record_result(
                "ConformalGRUAutoencoder",
                start,
                passed,
                details={
                    "threshold": float(gru.threshold_ or 0.0),
                    "expected_threshold": expected_threshold,
                    "prediction_count": int(len(gru_predictions["anomaly_prediction"])),
                },
            )
        except Exception as exc:
            results["ConformalGRUAutoencoder"] = _record_result(start=start, name="ConformalGRUAutoencoder", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            benign_event_mask = prepared["label"].astype(str).str.lower().isin(["normal", "benign"]).to_numpy()
            iso = app.BaselineIsolationForest(
                contamination=app.CONFIG["ATTACK_RATE"],
                random_state=app.CONFIG["SEED"],
            )
            iso.fit(behaviour["flat_features"][benign_event_mask])
            iso_predictions = iso.predict(behaviour["flat_features"])
            passed = (
                len(iso_predictions["predictions"]) == len(telemetry)
                and len(iso_predictions["anomaly_scores"]) == len(telemetry)
                and np.all(np.isfinite(iso_predictions["anomaly_scores"]))
            )
            results["BaselineIsolationForest"] = _record_result(
                "BaselineIsolationForest",
                start,
                passed,
                details={
                    "prediction_count": int(len(iso_predictions["predictions"])),
                    "anomalies_flagged": int(np.sum(iso_predictions["predictions"])),
                },
            )
        except Exception as exc:
            results["BaselineIsolationForest"] = _record_result(start=start, name="BaselineIsolationForest", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            mapper = app.MITREMapper()
            mapping = mapper.map_anomaly(
                88.0,
                {
                    "command_sequence": "login_fail;password_reset_attempt",
                    "auth_method": "password",
                    "time_since_previous_login": 0.1,
                    "login_frequency": 12,
                },
            )
            passed = all(
                key in mapping for key in ["technique_id", "technique_name", "severity", "soc_recommendation"]
            )
            results["MITREMapper"] = _record_result(
                "MITREMapper",
                start,
                passed,
                details=mapping,
            )
        except Exception as exc:
            results["MITREMapper"] = _record_result(start=start, name="MITREMapper", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            engineered = backend._assemble_engineered_features(prepared, behaviour)
            shap_result = backend._generate_shap_results(
                model=iso,
                feature_matrix=behaviour["flat_features"],
                engineered_features=engineered,
                context_engine=engine,
                event_predictions=iso_predictions,
            )
            top_feature_count = int(shap_result["explanations"].shape[1] - 3) if shap_result["available"] else 0
            passed = (
                ("available" in shap_result)
                and (
                    (shap_result["available"] and top_feature_count >= 5)
                    or (not shap_result["available"] and "reason" in shap_result)
                )
            )
            results["SHAPExplainer"] = _record_result(
                "SHAPExplainer",
                start,
                passed,
                details={
                    "available": bool(shap_result["available"]),
                    "top_feature_columns": top_feature_count,
                    "reason": shap_result.get("reason"),
                },
            )
        except Exception as exc:
            results["SHAPExplainer"] = _record_result(start=start, name="SHAPExplainer", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            evaluator = app.EvaluationEngine()
            metrics = evaluator.evaluate_models(
                y_true=np.array([0, 0, 1, 1], dtype=int),
                gru_results={
                    "anomaly_prediction": np.array([False, True, True, False]),
                    "risk_score": np.array([0.2, 0.9, 0.8, 0.1], dtype=np.float32),
                },
                isolation_forest_results={
                    "predictions": np.array([False, False, True, True]),
                    "anomaly_scores": np.array([0.01, 0.02, 0.8, 0.9], dtype=np.float32),
                },
            )
            passed = (
                np.isclose(metrics["ConformalGRUAutoencoder"]["precision"], 0.5)
                and metrics["BaselineIsolationForest"]["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
            )
            results["EvaluationEngine"] = _record_result(
                "EvaluationEngine",
                start,
                passed,
                details=metrics,
            )
        except Exception as exc:
            results["EvaluationEngine"] = _record_result(start=start, name="EvaluationEngine", passed=False, error=str(exc))
            raise

        start = _timer()
        try:
            pipeline_output = backend.run_pipeline()
            required_keys = {
                "telemetry_data",
                "engineered_features",
                "gru_predictions",
                "isolation_forest_predictions",
                "risk_scores",
                "adaptive_threshold",
                "mitre_results",
                "shap_results",
                "evaluation_metrics",
            }
            passed = required_keys.issubset(pipeline_output.keys())
            current, peak = tracemalloc.get_traced_memory()
            results["BackendEngine.run_pipeline"] = _record_result(
                "BackendEngine.run_pipeline",
                start,
                passed,
                details={
                    "returned_keys": sorted(pipeline_output.keys()),
                    "adaptive_threshold": float(pipeline_output["adaptive_threshold"]),
                    "current_memory_bytes": int(current),
                    "peak_memory_bytes": int(peak),
                },
            )
        except Exception as exc:
            results["BackendEngine.run_pipeline"] = _record_result(start=start, name="BackendEngine.run_pipeline", passed=False, error=str(exc))
            raise

        edge_cases = {}
        edge_start = _timer()
        try:
            try:
                engine.fit_transform(pd.DataFrame())
                edge_cases["empty_telemetry"] = "FAIL"
            except Exception:
                edge_cases["empty_telemetry"] = "PASS"

            single_entity = telemetry.copy()
            single_entity["entity_id"] = "ENT-SINGLE"
            single_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(single_entity)
            edge_cases["single_entity_only"] = "PASS" if single_behaviour["flat_features"].shape[0] == len(single_entity) else "FAIL"

            all_benign = telemetry.copy()
            all_benign["label"] = "Normal"
            all_benign_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(all_benign)
            single_backend = app.BackendEngine(config=app.CONFIG)
            single_prepared = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"])._prepare_dataframe(all_benign)
            single_seq_meta = single_backend._build_sequence_metadata(single_prepared)
            edge_cases["all_benign_data"] = "PASS" if len(single_seq_meta) >= 0 and all_benign_behaviour["flat_features"].shape[0] == len(all_benign) else "FAIL"

            unlabeled = telemetry.drop(columns=["label"])
            unlabeled_output = app.BackendEngine(config=app.CONFIG).run_pipeline(unlabeled)
            edge_cases["unlabeled_inference"] = "PASS" if not unlabeled_output["evaluation_metrics"] else "FAIL"

            hundred_attack = dict(app.CONFIG)
            hundred_attack["ATTACK_RATE"] = 1.0
            try:
                app.TelemetryGenerator(hundred_attack).generate()
                edge_cases["100_percent_attacks"] = "FAIL"
            except Exception:
                edge_cases["100_percent_attacks"] = "PASS"

            missing_values = telemetry.copy()
            missing_values.loc[0, "auth_method"] = None
            missing_values.loc[1, "resource_accessed"] = None
            missing_values.loc[2, "device_fingerprint"] = None
            missing_values.loc[3, "source_ip"] = None
            mv_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(missing_values)
            edge_cases["missing_values"] = "PASS" if mv_behaviour["flat_features"].shape[0] == len(missing_values) else "FAIL"

            duplicate_records = pd.concat([telemetry, telemetry.iloc[[0]]], ignore_index=True)
            dup_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(duplicate_records)
            edge_cases["duplicate_records"] = "PASS" if dup_behaviour["flat_features"].shape[0] == len(duplicate_records) else "FAIL"

            invalid_ip = telemetry.copy()
            invalid_ip.loc[0, "source_ip"] = "999.999.999.999"
            invalid_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(invalid_ip)
            edge_cases["invalid_ip_addresses"] = "PASS" if invalid_behaviour["flat_features"].shape[0] == len(invalid_ip) else "FAIL"

            unknown_auth = telemetry.copy()
            unknown_auth.loc[0, "auth_method"] = "quantum_badge"
            ua_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(unknown_auth)
            edge_cases["unknown_auth_method"] = "PASS" if ua_behaviour["flat_features"].shape[0] == len(unknown_auth) else "FAIL"

            unknown_resource = telemetry.copy()
            unknown_resource.loc[0, "resource_accessed"] = "weirdresource"
            ur_behaviour = app.AdaptiveBehaviourContextEngine(window_size=app.CONFIG["WINDOW_SIZE"]).fit_transform(unknown_resource)
            edge_cases["unknown_resource_category"] = "PASS" if ur_behaviour["flat_features"].shape[0] == len(unknown_resource) else "FAIL"

            results["EdgeCases"] = _record_result(
                "EdgeCases",
                edge_start,
                all(value == "PASS" for value in edge_cases.values()),
                details=edge_cases,
            )
        except Exception as exc:
            results["EdgeCases"] = _record_result(start=edge_start, name="EdgeCases", passed=False, error=str(exc))

        warning_text = [str(w.message) for w in caught_warnings]
        for component in results.values():
            if component["warnings"] == []:
                component["warnings"] = warning_text

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "components": results,
        "overall_execution_time_sec": round(time.perf_counter() - overall_start, 4),
        "memory": {"current_bytes": int(current), "peak_bytes": int(peak)},
    }


if __name__ == "__main__":
    report = run_qa()
    print(json.dumps(report, default=str, indent=2))
