"""Optional offline network intelligence package for CADENCE.

This package is intentionally independent from the existing anomaly-detection
pipeline. It can be imported by future services or dashboards to enrich login
events with network trust context, while leaving the validated backend
unchanged when unused.
"""

from .models import NetworkAnalysisResult, NetworkRecord
from .network_database import OfflineNetworkDatabase
from .network_intelligence_engine import NetworkIntelligenceEngine

__all__ = [
    "NetworkAnalysisResult",
    "NetworkRecord",
    "OfflineNetworkDatabase",
    "NetworkIntelligenceEngine",
]
