"""Independent unit tests for the optional network intelligence module."""

from __future__ import annotations

import unittest

from network_intelligence import NetworkIntelligenceEngine


class NetworkIntelligenceEngineTests(unittest.TestCase):
    """Validate standalone offline network intelligence behavior."""

    def setUp(self) -> None:
        self.engine = NetworkIntelligenceEngine()

    def test_residential_network(self) -> None:
        result = self.engine.analyze({"user_id": "u1", "ip_address": "103.48.1.10"})
        self.assertEqual(result["network_type"], "Residential")
        self.assertEqual(result["isp"], "Airtel")
        self.assertEqual(result["asn"], "AS9498")
        self.assertGreaterEqual(result["network_trust_score"], 70.0)

    def test_enterprise_network(self) -> None:
        result = self.engine.analyze({"user_id": "u2", "ip_address": "20.40.22.8"})
        self.assertEqual(result["network_type"], "Enterprise")
        self.assertEqual(result["isp"], "Microsoft Corporate")

    def test_datacenter_network(self) -> None:
        result = self.engine.analyze({"user_id": "u3", "ip_address": "52.95.10.20"})
        self.assertEqual(result["network_type"], "Datacenter")
        self.assertEqual(result["isp"], "AWS")

    def test_vpn_hosting_network(self) -> None:
        result = self.engine.analyze({"user_id": "u4", "ip_address": "45.83.15.2"})
        self.assertEqual(result["network_type"], "VPN/Hosting")
        self.assertEqual(result["isp"], "M247")
        self.assertLessEqual(result["network_trust_score"], 30.0)

    def test_unknown_network(self) -> None:
        result = self.engine.analyze({"user_id": "u5", "ip_address": "8.8.8.8"})
        self.assertEqual(result["network_type"], "Unknown")
        self.assertEqual(result["isp"], "Unknown ISP")
        self.assertEqual(result["asn"], "AS-UNKNOWN")

    def test_asn_change_detection(self) -> None:
        self.engine.analyze({"user_id": "u6", "ip_address": "103.48.1.10"})
        result = self.engine.analyze({"user_id": "u6", "ip_address": "52.95.10.20"})
        self.assertTrue(result["asn_changed"])

    def test_subnet_change_detection(self) -> None:
        self.engine.analyze({"user_id": "u7", "ip_address": "103.48.1.10"})
        result = self.engine.analyze({"user_id": "u7", "ip_address": "103.48.2.10"})
        self.assertTrue(result["subnet_changed"])

    def test_trust_score_generation(self) -> None:
        first = self.engine.analyze({"user_id": "u8", "ip_address": "20.40.22.8"})
        second = self.engine.analyze({"user_id": "u8", "ip_address": "20.40.22.9"})
        self.assertGreaterEqual(second["network_trust_score"], first["network_trust_score"])
        self.assertGreaterEqual(second["historical_match_score"], first["historical_match_score"])

    def test_no_runtime_exceptions_and_dashboard_fields(self) -> None:
        result = self.engine.analyze({"user_id": "u9", "ip_address": "149.28.9.2"})
        self.assertIn("dashboard", result)
        self.assertIn("reasons", result)
        self.assertTrue(isinstance(result["dashboard"]["reasons"], list))


if __name__ == "__main__":
    unittest.main()
