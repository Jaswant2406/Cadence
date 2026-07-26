from __future__ import annotations

import unittest

import pandas as pd

from app import CONFIG, TelemetryGenerator
from cadence_api.csv_validation import ALLOWED_ENTITY_TYPES, REQUIRED_COLUMNS, validate_csv_bytes


def valid_csv(
    rows: int = 8,
    entity_type: str = "user",
    include_label: bool = True,
) -> bytes:
    frame = pd.DataFrame(
        [
            {
                "entity_id": "ENT-0001",
                "entity_type": entity_type,
                "timestamp": f"2026-07-01 {9 + index:02d}:00:00",
                "source_ip": "103.48.1.10",
                "geo_location": "New York, US",
                "resource_accessed": "sso://portal/login",
                "auth_method": "sso",
                "session_duration": 20 + index,
                "command_sequence": "login;open_app;logout",
                "device_fingerprint": "device-a1",
                "label": "Normal",
            }
            for index in range(rows)
        ],
        columns=REQUIRED_COLUMNS + (["label"] if include_label else []),
    )
    return frame.to_csv(index=False).encode("utf-8")


class CsvValidationTests(unittest.TestCase):
    def test_valid_labeled_csv(self) -> None:
        result = validate_csv_bytes(valid_csv())

        self.assertTrue(result.valid)
        self.assertEqual(result.summary["rows"], 8)
        self.assertEqual(result.summary["entities"], 1)
        self.assertEqual(result.summary["normal_rows"], 8)
        self.assertEqual(len(result.preview), 8)

    def test_honeywell_entity_types_are_supported(self) -> None:
        self.assertEqual(ALLOWED_ENTITY_TYPES, {"user", "service_account", "edge_device"})
        for entity_type in sorted(ALLOWED_ENTITY_TYPES):
            with self.subTest(entity_type=entity_type):
                self.assertTrue(validate_csv_bytes(valid_csv(entity_type=entity_type)).valid)

    def test_label_is_optional_for_inference(self) -> None:
        result = validate_csv_bytes(valid_csv(include_label=False))

        self.assertTrue(result.valid, result.errors)
        self.assertFalse(result.summary["labels_present"])
        self.assertNotIn("label", result.dataframe.columns)

    def test_generated_dataset_passes_csv_validation(self) -> None:
        generated = TelemetryGenerator(CONFIG).generate()
        result = validate_csv_bytes(generated.to_csv(index=False).encode("utf-8"))

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(set(generated["entity_type"].unique()), ALLOWED_ENTITY_TYPES)

    def test_missing_required_column_is_blocking(self) -> None:
        frame = pd.read_csv(pd.io.common.BytesIO(valid_csv()))
        payload = frame.drop(columns=["device_fingerprint"]).to_csv(index=False)
        result = validate_csv_bytes(payload.encode("utf-8"))

        self.assertFalse(result.valid)
        self.assertIn("Missing required columns", result.errors[0]["reason"])

    def test_arbitrary_attack_label_does_not_block_inference(self) -> None:
        payload = valid_csv().decode("utf-8").replace("Normal", "Unknown Attack", 1)
        result = validate_csv_bytes(payload.encode("utf-8"))

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.dataframe.iloc[0]["label"], "Unknown Attack")

    def test_insufficient_benign_windows_is_blocking(self) -> None:
        result = validate_csv_bytes(valid_csv(rows=5))

        self.assertFalse(result.valid)
        self.assertTrue(any("three five-event" in issue["reason"] for issue in result.errors))


if __name__ == "__main__":
    unittest.main()
