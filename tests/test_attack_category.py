from __future__ import annotations

import unittest

from app import MITREMapper


class AttackCategoryPredictionTests(unittest.TestCase):
    def test_behavior_rules_cover_supported_attack_categories(self) -> None:
        cases = {
            "Brute Force": {"command_sequence": "login_fail;login_fail;login_fail;password_reset_attempt"},
            "Impossible Travel": {"ip_subnet_change": True, "time_since_previous_login": 0.5},
            "Credential Stuffing": {"command_sequence": "login_fail;login_fail;login;open_app"},
            "Lateral Movement": {"command_sequence": "enumerate_hosts;remote_exec;session_clone"},
            "Device Spoofing": {"command_sequence": "login;browser_check;open_app"},
            "Low-and-Slow Exfiltration": {"command_sequence": "query_small_batch;export_csv_chunk;compress_chunk"},
            "Insider Drift": {"command_sequence": "search_directory;open_sensitive_doc;bulk_lookup"},
        }
        mapper = MITREMapper()

        for expected_category, context in cases.items():
            with self.subTest(category=expected_category):
                result = mapper.map_anomaly(80.0, context)
                self.assertEqual(result["predicted_attack_category"], expected_category)
                self.assertGreaterEqual(result["prediction_confidence"], 0.0)
                self.assertLessEqual(result["prediction_confidence"], 100.0)
                self.assertTrue(result["technique_id"])


if __name__ == "__main__":
    unittest.main()
