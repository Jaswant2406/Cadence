from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from app import TelemetryGenerator


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 25_000
REQUIRED_COLUMNS = [
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
]
OPTIONAL_COLUMNS = ["label"]
ALLOWED_ENTITY_TYPES = set(TelemetryGenerator.ENTITY_TYPES)
LABEL_CANONICAL = {
    "normal": "Normal",
    "benign": "Benign",
    **{label.lower(): label for label in TelemetryGenerator.ATTACK_LABELS},
}


@dataclass
class CsvValidationResult:
    dataframe: Optional[pd.DataFrame]
    summary: Dict[str, Any]
    preview: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]

    @property
    def valid(self) -> bool:
        return self.dataframe is not None and not self.errors


def _issue(
    reason: str,
    *,
    row: Optional[int] = None,
    column: Optional[str] = None,
    value: Any = None,
) -> Dict[str, Any]:
    return {
        "row": row,
        "column": column,
        "value": None if value is None else str(value)[:160],
        "reason": reason,
    }


def validate_csv_bytes(payload: bytes) -> CsvValidationResult:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not payload:
        return CsvValidationResult(None, {}, [], [_issue("The uploaded file is empty.")], [])
    if len(payload) > MAX_FILE_BYTES:
        return CsvValidationResult(
            None,
            {},
            [],
            [_issue("CSV exceeds the 10 MB upload limit.")],
            [],
        )

    try:
        frame = pd.read_csv(BytesIO(payload), encoding="utf-8")
    except UnicodeDecodeError:
        return CsvValidationResult(
            None,
            {},
            [],
            [_issue("CSV must use UTF-8 encoding.")],
            [],
        )
    except Exception as exc:
        return CsvValidationResult(
            None,
            {},
            [],
            [_issue(f"Unable to parse CSV: {exc}")],
            [],
        )

    if frame.empty:
        errors.append(_issue("CSV must contain at least one data row."))
    if len(frame) > MAX_ROWS:
        errors.append(_issue(f"CSV contains {len(frame):,} rows; the limit is {MAX_ROWS:,}."))

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    accepted_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    unexpected = [column for column in frame.columns if column not in accepted_columns]
    if missing:
        errors.append(_issue(f"Missing required columns: {', '.join(missing)}."))
    if unexpected:
        warnings.append(_issue(f"Unexpected columns will be ignored: {', '.join(unexpected)}."))
    if missing or frame.empty:
        return CsvValidationResult(None, {}, [], errors, warnings)

    labels_present = "label" in frame.columns
    selected_columns = REQUIRED_COLUMNS + (["label"] if labels_present else [])
    frame = frame[selected_columns].copy()
    string_columns = [
        column
        for column in selected_columns
        if column not in {"session_duration", "timestamp"}
    ]
    for column in string_columns:
        frame[column] = frame[column].astype("string").str.strip()

    for column in REQUIRED_COLUMNS:
        missing_mask = frame[column].isna()
        if column in string_columns:
            missing_mask = missing_mask | frame[column].eq("")
        for index in frame.index[missing_mask][:50]:
            errors.append(
                _issue(
                    "A value is required.",
                    row=int(index) + 2,
                    column=column,
                    value=frame.at[index, column],
                )
            )

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    for index in frame.index[timestamps.isna()][:50]:
        errors.append(
            _issue(
                "Timestamp is invalid; use an ISO-style date and time.",
                row=int(index) + 2,
                column="timestamp",
                value=frame.at[index, "timestamp"],
            )
        )

    durations = pd.to_numeric(frame["session_duration"], errors="coerce")
    invalid_durations = durations.isna() | (durations < 0)
    for index in frame.index[invalid_durations][:50]:
        errors.append(
            _issue(
                "Session duration must be a non-negative number.",
                row=int(index) + 2,
                column="session_duration",
                value=frame.at[index, "session_duration"],
            )
        )

    for index, value in frame["source_ip"].items():
        try:
            ipaddress.ip_address(str(value))
        except ValueError:
            if len(errors) < 200:
                errors.append(
                    _issue(
                        "Source IP is not a valid IPv4 or IPv6 address.",
                        row=int(index) + 2,
                        column="source_ip",
                        value=value,
                    )
                )

    normalized_types = frame["entity_type"].str.lower()
    invalid_types = ~normalized_types.isin(ALLOWED_ENTITY_TYPES)
    for index in frame.index[invalid_types][:50]:
        errors.append(
            _issue(
                f"Supported values: {', '.join(sorted(ALLOWED_ENTITY_TYPES))}.",
                row=int(index) + 2,
                column="entity_type",
                value=frame.at[index, "entity_type"],
            )
        )

    normalized_labels = None
    if labels_present:
        missing_labels = frame["label"].isna() | frame["label"].eq("")
        if missing_labels.any():
            warnings.append(
                _issue(
                    "The optional label column contains missing values; it will be ignored and evaluation metrics will be omitted."
                )
            )
            frame = frame.drop(columns=["label"])
            labels_present = False
        else:
            normalized_labels = frame["label"].str.lower()

    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        warnings.append(_issue(f"{duplicate_count:,} duplicate rows were found and retained."))

    if errors:
        preview = _preview(frame)
        return CsvValidationResult(None, {}, preview, errors[:200], warnings)

    frame["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["session_duration"] = durations.astype(float)
    frame["entity_type"] = normalized_types
    if labels_present and normalized_labels is not None:
        frame["label"] = normalized_labels.map(LABEL_CANONICAL).fillna(frame["label"])

    sequence_windows = 0
    window_size = 5
    for _, entity_frame in frame.sort_values("timestamp").groupby("entity_id"):
        sequence_windows += max(0, len(entity_frame) - window_size + 1)
    if sequence_windows < 3:
        errors.append(
            _issue("At least three five-event entity windows are required for GRU training.")
        )
        return CsvValidationResult(None, {}, _preview(frame), errors, warnings)

    benign = None
    attack_counts: Dict[str, int] = {}
    if labels_present:
        benign = frame["label"].str.lower().isin(["normal", "benign"])
        attack_counts = frame.loc[~benign, "label"].value_counts().sort_index().astype(int).to_dict()
    else:
        warnings.append(
            _issue(
                "No label column was provided; inference will run and evaluation metrics will be omitted."
            )
        )
    summary = {
        "rows": int(len(frame)),
        "entities": int(frame["entity_id"].nunique()),
        "labels_present": labels_present,
        "normal_rows": int(benign.sum()) if benign is not None else None,
        "anomaly_rows": int((~benign).sum()) if benign is not None else None,
        "anomaly_rate": round(float((~benign).mean()) * 100.0, 2) if benign is not None else None,
        "time_start": str(frame["timestamp"].min()),
        "time_end": str(frame["timestamp"].max()),
        "attack_counts": attack_counts,
        "duplicate_rows": duplicate_count,
    }
    return CsvValidationResult(frame, summary, _preview(frame), errors, warnings)


def _preview(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return frame.head(8).where(pd.notna(frame.head(8)), None).to_dict(orient="records")
