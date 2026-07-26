"""Standalone network intelligence engine for login trust analysis.

This module complements behaviour analytics by supplying additional network
context features such as ASN continuity, subnet familiarity, and network trust.
It is intentionally optional and independent: importing or ignoring it does not
change the previously validated CADENCE anomaly-detection pipeline.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Optional

from .models import NetworkAnalysisResult, NetworkObservation
from .network_database import OfflineNetworkDatabase


class NetworkIntelligenceEngine:
    """Analyze network trustworthiness for login events using offline metadata."""

    DEFAULT_TRUST_RULES: Dict[str, int] = {
        "Residential": 70,
        "Enterprise": 85,
        "Datacenter": 30,
        "VPN/Hosting": 15,
        "Unknown": 40,
        "known_network_bonus": 10,
        "same_asn_bonus": 12,
        "same_subnet_bonus": 10,
        "asn_changed_penalty": 20,
        "subnet_changed_penalty": 12,
        "unknown_network_penalty": 8,
    }

    def __init__(
        self,
        database: Optional[OfflineNetworkDatabase] = None,
        trust_rules: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize the optional network intelligence engine."""
        self.database = database or OfflineNetworkDatabase()
        self.trust_rules = dict(self.DEFAULT_TRUST_RULES)
        if trust_rules:
            self.trust_rules.update(trust_rules)
        self._history: Dict[str, List[NetworkObservation]] = {}

    def analyze(self, login_event: Dict[str, Any]) -> Dict[str, object]:
        """Analyze a login event and return dashboard-ready network trust context.

        Args:
            login_event: Dictionary containing at minimum `ip_address` and
                `user_id`.

        Returns:
            A structured dictionary with ASN, ISP, network-type, continuity
            checks, historical match scoring, trust scoring, and human-readable
            reasons suitable for future dashboard cards.
        """
        user_id = self._require_str(login_event, "user_id")
        ip_address = self._require_str(login_event, "ip_address")
        subnet = self._extract_subnet(ip_address)
        record = self.database.lookup(ip_address)

        asn = record.asn if record else "AS-UNKNOWN"
        isp = record.isp if record else "Unknown ISP"
        network_type = record.network_type if record else "Unknown"

        user_history = self._history.get(user_id, [])
        previous = user_history[-1] if user_history else None

        known_network = any(obs.asn == asn or obs.subnet == subnet for obs in user_history)
        asn_changed = previous is not None and previous.asn != asn
        subnet_changed = previous is not None and previous.subnet != subnet
        historical_match_score = self._historical_match_score(user_history, asn, subnet)
        network_trust_score = self._network_trust_score(
            network_type=network_type,
            known_network=known_network,
            asn_changed=asn_changed,
            subnet_changed=subnet_changed,
        )
        reasons = self._build_reasons(
            network_type=network_type,
            known_network=known_network,
            asn_changed=asn_changed,
            subnet_changed=subnet_changed,
        )

        result = NetworkAnalysisResult(
            asn=asn,
            isp=isp,
            network_type=network_type,
            known_network=known_network,
            asn_changed=asn_changed,
            subnet_changed=subnet_changed,
            historical_match_score=historical_match_score,
            network_trust_score=network_trust_score,
            reasons=reasons,
        )
        self._history.setdefault(user_id, []).append(
            NetworkObservation(
                user_id=user_id,
                ip_address=ip_address,
                subnet=subnet,
                asn=asn,
                isp=isp,
                network_type=network_type,
            )
        )
        return result.to_dict()

    def _historical_match_score(
        self,
        history: List[NetworkObservation],
        asn: str,
        subnet: str,
    ) -> float:
        """Score similarity between the current network and user history."""
        if not history:
            return 50.0
        asn_matches = sum(1 for obs in history if obs.asn == asn)
        subnet_matches = sum(1 for obs in history if obs.subnet == subnet)
        asn_ratio = asn_matches / len(history)
        subnet_ratio = subnet_matches / len(history)
        return round(min(100.0, (asn_ratio * 60.0) + (subnet_ratio * 40.0)), 2)

    def _network_trust_score(
        self,
        network_type: str,
        known_network: bool,
        asn_changed: bool,
        subnet_changed: bool,
    ) -> float:
        """Apply configurable rules to derive a bounded trust score."""
        score = float(self.trust_rules.get(network_type, self.trust_rules["Unknown"]))
        if known_network:
            score += self.trust_rules["known_network_bonus"]
        else:
            score -= self.trust_rules["unknown_network_penalty"]
        if asn_changed:
            score -= self.trust_rules["asn_changed_penalty"]
        else:
            score += self.trust_rules["same_asn_bonus"]
        if subnet_changed:
            score -= self.trust_rules["subnet_changed_penalty"]
        else:
            score += self.trust_rules["same_subnet_bonus"]
        return round(max(0.0, min(100.0, score)), 2)

    def _build_reasons(
        self,
        network_type: str,
        known_network: bool,
        asn_changed: bool,
        subnet_changed: bool,
    ) -> List[str]:
        """Generate simple dashboard-friendly explanations."""
        reasons: List[str] = []
        reasons.append(f"{'Known' if known_network else 'Unknown'} Network")
        reasons.append(f"{'ASN Changed' if asn_changed else 'Same ASN'}")
        reasons.append(f"{'Subnet Changed' if subnet_changed else 'Same Subnet'}")
        reasons.append(f"{network_type} Network")
        return reasons

    def _extract_subnet(self, ip_address: str) -> str:
        """Convert an IP address to a coarse /24 subnet string."""
        ip_obj = ipaddress.ip_address(ip_address)
        return str(ipaddress.ip_network(f"{ip_obj}/24", strict=False))

    def _require_str(self, payload: Dict[str, Any], key: str) -> str:
        """Validate required string fields for incoming login events."""
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"login_event must contain a non-empty string '{key}'.")
        return value.strip()
