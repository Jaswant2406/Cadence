"""Offline sample network intelligence database.

The database intentionally uses a small local CIDR-to-metadata mapping with no
network calls, APIs, or external services. It is sufficient for testing and
future dashboard prototyping without affecting the production CADENCE backend.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable, List, Optional

from .models import NetworkRecord


class OfflineNetworkDatabase:
    """Local CIDR-backed lookup table for sample ASN and ISP intelligence."""

    def __init__(self, records: Optional[Iterable[NetworkRecord]] = None) -> None:
        """Initialize the offline database with sample or caller-provided data."""
        self._records: List[NetworkRecord] = list(records) if records is not None else self._default_records()

    def lookup(self, ip_address: str) -> Optional[NetworkRecord]:
        """Return the first matching record for an IP address, if any."""
        ip_obj = ipaddress.ip_address(ip_address)
        for record in self._records:
            if ip_obj in ipaddress.ip_network(record.cidr, strict=False):
                return record
        return None

    def _default_records(self) -> List[NetworkRecord]:
        """Provide a small offline sample database across common network types."""
        return [
            NetworkRecord(cidr="103.48.0.0/16", asn="AS9498", isp="Airtel", network_type="Residential"),
            NetworkRecord(cidr="49.32.0.0/16", asn="AS55836", isp="Jio", network_type="Residential"),
            NetworkRecord(cidr="117.192.0.0/16", asn="AS9829", isp="BSNL", network_type="Residential"),
            NetworkRecord(cidr="20.40.0.0/16", asn="AS8075", isp="Microsoft Corporate", network_type="Enterprise"),
            NetworkRecord(cidr="15.220.0.0/16", asn="AS65010", isp="Honeywell Corporate", network_type="Enterprise"),
            NetworkRecord(cidr="52.95.0.0/16", asn="AS16509", isp="AWS", network_type="Datacenter"),
            NetworkRecord(cidr="20.120.0.0/16", asn="AS8075", isp="Azure", network_type="Datacenter"),
            NetworkRecord(cidr="34.80.0.0/16", asn="AS15169", isp="Google Cloud", network_type="Datacenter"),
            NetworkRecord(cidr="134.122.0.0/16", asn="AS14061", isp="DigitalOcean", network_type="Datacenter"),
            NetworkRecord(cidr="129.146.0.0/16", asn="AS31898", isp="Oracle Cloud", network_type="Datacenter"),
            NetworkRecord(cidr="45.83.0.0/16", asn="AS9009", isp="M247", network_type="VPN/Hosting"),
            NetworkRecord(cidr="51.68.0.0/16", asn="AS16276", isp="OVH", network_type="VPN/Hosting"),
            NetworkRecord(cidr="149.28.0.0/16", asn="AS20473", isp="Vultr", network_type="VPN/Hosting"),
        ]
