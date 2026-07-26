"""Typed models for the optional network intelligence module.

These models support lightweight offline network enrichment that complements
behaviour analytics. They add network-context features such as ASN continuity,
subnet familiarity, and trust scoring without changing the existing anomaly
detection pipeline or any of its validated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class NetworkRecord:
    """Offline network metadata for one CIDR range."""

    cidr: str
    asn: str
    isp: str
    network_type: str


@dataclass(frozen=True)
class NetworkObservation:
    """Historical network observation stored per user."""

    user_id: str
    ip_address: str
    subnet: str
    asn: str
    isp: str
    network_type: str


@dataclass
class NetworkAnalysisResult:
    """Dashboard-ready output for one login event."""

    asn: str
    isp: str
    network_type: str
    known_network: bool
    asn_changed: bool
    subnet_changed: bool
    historical_match_score: float
    network_trust_score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON/dashboard-friendly representation."""
        return {
            "asn": self.asn,
            "isp": self.isp,
            "network_type": self.network_type,
            "known_network": self.known_network,
            "asn_changed": self.asn_changed,
            "subnet_changed": self.subnet_changed,
            "historical_match_score": self.historical_match_score,
            "network_trust_score": self.network_trust_score,
            "reasons": list(self.reasons),
            "dashboard": {
                "network_trust": f"{int(round(self.network_trust_score))}%",
                "isp": self.isp,
                "asn": self.asn,
                "network_type": self.network_type,
                "reasons": list(self.reasons),
            },
        }
