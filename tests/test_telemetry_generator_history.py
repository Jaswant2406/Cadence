from __future__ import annotations

import unittest
import warnings

import pandas as pd

from app import CONFIG, TelemetryGenerator


class TelemetryGeneratorHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = TelemetryGenerator(CONFIG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.telemetry = cls.generator.generate()
        cls.telemetry["parsed_timestamp"] = pd.to_datetime(cls.telemetry["timestamp"])

    def test_entity_id_has_one_permanent_type(self) -> None:
        type_counts = self.telemetry.groupby("entity_id")["entity_type"].nunique()

        self.assertTrue(type_counts.eq(1).all())
        self.assertEqual(set(self.telemetry["entity_type"]), set(TelemetryGenerator.ENTITY_TYPES))

    def test_every_entity_has_an_established_baseline(self) -> None:
        normal = self.telemetry[self.telemetry["label"] == "Normal"]
        normal_counts = normal.groupby("entity_id").size()

        self.assertEqual(normal_counts.size, len(self.generator._profiles))
        self.assertTrue(
            normal_counts.ge(TelemetryGenerator.MIN_BASELINE_EVENTS_PER_ENTITY).all()
        )

    def test_normal_history_matches_the_stable_entity_profile(self) -> None:
        normal = self.telemetry[self.telemetry["label"] == "Normal"]
        profile_by_id = {profile.entity_id: profile for profile in self.generator._profiles}

        for entity_id, history in normal.groupby("entity_id"):
            profile = profile_by_id[entity_id]
            self.assertEqual(history["geo_location"].nunique(), 1)
            self.assertEqual(history["device_fingerprint"].nunique(), 1)
            self.assertEqual(history["device_fingerprint"].iloc[0], profile.base_device)
            self.assertTrue(set(history["resource_accessed"]).issubset(profile.common_resources))
            self.assertTrue(
                history["session_duration"].between(
                    profile.session_duration_range[0],
                    profile.session_duration_range[1],
                    inclusive="both",
                ).all()
            )

    def test_anomalies_extend_existing_histories(self) -> None:
        normal = self.telemetry[self.telemetry["label"] == "Normal"]
        anomalies = self.telemetry[self.telemetry["label"] != "Normal"]

        self.assertGreaterEqual(len(anomalies) / len(self.telemetry), 0.005)
        self.assertLessEqual(len(anomalies) / len(self.telemetry), 0.03)
        self.assertTrue(set(anomalies["entity_id"]).issubset(set(normal["entity_id"])))
        for entity_id, anomaly_history in anomalies.groupby("entity_id"):
            baseline_history = normal[normal["entity_id"] == entity_id]
            self.assertLess(
                baseline_history["parsed_timestamp"].max(),
                anomaly_history["parsed_timestamp"].min(),
            )


if __name__ == "__main__":
    unittest.main()
