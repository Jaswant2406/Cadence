"""Phase 1 telemetry generation for CADENCE.

This module implements only the synthetic telemetry generation layer.
It does not include Streamlit, machine learning models, or future modules.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


CONFIG: Dict[str, Any] = {
    "N_LOGS": 2500,
    "ATTACK_RATE": 0.025,
    "WINDOW_SIZE": 5,
    "TARGET_FPR": 0.01,
    "SEED": 42,
}


random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["SEED"])


@dataclass(frozen=True)
class EntityProfile:
    """Stable behavior profile used to synthesize realistic entity activity."""

    entity_id: str
    entity_type: str
    department: str
    seniority: str
    home_region: str
    preferred_auth: str
    common_resources: Tuple[str, ...]
    normal_commands: Tuple[str, ...]
    avg_session_duration: int
    session_duration_range: Tuple[int, int]
    base_device: str


class TelemetryGenerator:
    """Generate synthetic enterprise telemetry for anomaly-detection experiments."""

    MIN_ATTACK_RATE: float = 0.005
    MAX_ATTACK_RATE: float = 0.03
    MIN_BASELINE_EVENTS_PER_ENTITY: int = 5
    MIN_ATTACK_SAMPLES_PER_CATEGORY: int = 20
    ATTACK_LABELS: Tuple[str, ...] = (
        "Brute Force",
        "Impossible Travel",
        "Credential Stuffing",
        "Lateral Movement",
        "Device Spoofing",
        "Low-and-Slow Exfiltration",
        "Insider Drift",
    )

    ENTITY_TYPES: Tuple[str, ...] = ("user", "service_account", "edge_device")
    AUTH_METHODS: Tuple[str, ...] = ("password", "mfa_push", "webauthn", "sso", "vpn_cert", "api_token")
    RESOURCES_BY_DEPARTMENT: Dict[str, Tuple[str, ...]] = {
        "engineering": (
            "git://internal/repo/core-api",
            "k8s://cluster/prod-namespace",
            "db://warehouse/feature_store",
            "confluence://engineering/runbooks",
            "s3://analytics-build-artifacts",
        ),
        "finance": (
            "erp://oracle-payables",
            "share://finance/quarterly-close",
            "db://warehouse/revenue_mart",
            "confluence://finance/policies",
            "s3://finance-reports",
        ),
        "it": (
            "ad://corp/directory",
            "vpn://gateway/ops",
            "siem://splunk/auth_logs",
            "cmdb://assets/fleet",
            "rmm://windows/endpoints",
        ),
        "sales": (
            "crm://salesforce/opportunities",
            "share://sales/regional-forecast",
            "confluence://sales/playbooks",
            "email://o365/customer-thread",
            "s3://sales-enablement-assets",
        ),
        "hr": (
            "hris://workday/people",
            "share://hr/policies",
            "confluence://hr/onboarding",
            "email://o365/recruiting",
            "db://warehouse/headcount_mart",
        ),
        "security": (
            "edr://sentinelone/incidents",
            "siem://splunk/detections",
            "ticket://jira/secops",
            "vault://secrets/rotation",
            "confluence://security/playbooks",
        ),
    }
    NORMAL_COMMANDS: Dict[str, Tuple[str, ...]] = {
        "user": ("login", "open_app", "view_record", "edit_document", "logout"),
        "service_account": ("token_auth", "job_start", "read_queue", "write_result", "job_end"),
        "edge_device": ("device_auth", "read_sensor", "publish_telemetry", "health_check", "sleep"),
    }
    REGIONAL_SUBNETS: Dict[str, Tuple[str, str]] = {
        "new_york": ("10.14.0.0/16", "198.51.100.0/24"),
        "london": ("10.22.0.0/16", "203.0.113.0/24"),
        "singapore": ("10.31.0.0/16", "192.0.2.0/24"),
        "frankfurt": ("10.41.0.0/16", "172.20.40.0/24"),
        "sydney": ("10.52.0.0/16", "172.21.50.0/24"),
        "toronto": ("10.63.0.0/16", "172.22.60.0/24"),
    }
    GEO_LOCATIONS: Dict[str, str] = {
        "new_york": "New York, US",
        "london": "London, GB",
        "singapore": "Singapore, SG",
        "frankfurt": "Frankfurt, DE",
        "sydney": "Sydney, AU",
        "toronto": "Toronto, CA",
    }
    SUSPICIOUS_RESOURCES: Tuple[str, ...] = (
        "vault://secrets/global-admin",
        "db://warehouse/pii_export",
        "s3://legal-hold-archives",
        "ad://corp/domain-admins",
        "k8s://cluster/prod-kube-system",
        "share://board/acquisition-plan",
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the generator with validated configuration."""
        self.config = config
        self.seed = int(config.get("SEED", 42))
        self.n_logs = int(config.get("N_LOGS", 2500))
        self.attack_rate = float(config.get("ATTACK_RATE", 0.025))
        if not self.MIN_ATTACK_RATE <= self.attack_rate <= self.MAX_ATTACK_RATE:
            raise ValueError(
                f"ATTACK_RATE must remain between {self.MIN_ATTACK_RATE:.3f} and "
                f"{self.MAX_ATTACK_RATE:.2f}."
            )
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        self._profiles: List[EntityProfile] = []
        self._events_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        self._base_time = datetime(2026, 7, 1, 8, 0, 0)

    def generate(self) -> pd.DataFrame:
        """Generate telemetry logs as a pandas DataFrame.

        Returns:
            A DataFrame with the requested telemetry schema.
        """
        try:
            self._profiles = self._build_profiles()
            total_attacks = self._determine_total_attacks()
            total_normals = self.n_logs - total_attacks

            minimum_baseline_events = len(self._profiles) * self.MIN_BASELINE_EVENTS_PER_ENTITY
            if total_normals < minimum_baseline_events:
                raise ValueError(
                    "N_LOGS and ATTACK_RATE must allow at least "
                    f"{self.MIN_BASELINE_EVENTS_PER_ENTITY} baseline events per entity."
                )
            normal_logs = [
                self._generate_normal_log(profile)
                for profile in self._profiles
                for _ in range(self.MIN_BASELINE_EVENTS_PER_ENTITY)
            ]
            normal_logs.extend(
                self._generate_normal_log()
                for _ in range(total_normals - minimum_baseline_events)
            )
            attack_logs = self._generate_attack_logs(total_attacks)
            logs = normal_logs + attack_logs
            self._rng.shuffle(logs)

            df = pd.DataFrame(logs)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            return df[
                [
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
            ]
        except Exception as exc:
            raise RuntimeError(f"Failed to generate telemetry logs: {exc}") from exc

    def _determine_total_attacks(self) -> int:
        """Determine a feasible attack budget while warning on configuration conflicts."""
        minimum_attacks = len(self.ATTACK_LABELS) * self.MIN_ATTACK_SAMPLES_PER_CATEGORY
        configured_attacks = max(1, int(round(self.n_logs * self.attack_rate)))
        if configured_attacks >= self.n_logs:
            raise ValueError(
                "N_LOGS must be larger than the configured number of attack samples."
            )
        if minimum_attacks > configured_attacks:
            warnings.warn(
                (
                    "Configured ATTACK_RATE allows only "
                    f"{configured_attacks} attack samples, which is below the requested minimum of "
                    f"{self.MIN_ATTACK_SAMPLES_PER_CATEGORY} per attack category "
                    f"({minimum_attacks} total). Adjusting attack distribution to preserve the "
                    "configured attack rate while keeping every attack category represented."
                ),
                RuntimeWarning,
            )
        if configured_attacks < len(self.ATTACK_LABELS):
            warnings.warn(
                (
                    "Configured ATTACK_RATE yields fewer attack samples than attack categories. "
                    "Increasing total attacks to ensure every attack category appears at least once."
                ),
                RuntimeWarning,
            )
            return len(self.ATTACK_LABELS)
        return configured_attacks

    def _build_profiles(self, n_entities: int = 180) -> List[EntityProfile]:
        """Create reusable entity profiles with stable behavioral patterns."""
        profiles: List[EntityProfile] = []
        departments = tuple(self.RESOURCES_BY_DEPARTMENT.keys())
        regions = tuple(self.REGIONAL_SUBNETS.keys())
        type_weights = [0.72, 0.16, 0.12]
        seniority_levels = ("junior", "mid", "senior", "lead")

        for idx in range(n_entities):
            entity_type = self._rng.choices(self.ENTITY_TYPES, weights=type_weights, k=1)[0]
            department = self._rng.choice(departments)
            seniority = self._rng.choice(seniority_levels)
            region = self._rng.choice(regions)
            auth = self._preferred_auth(entity_type)
            resources = self._select_common_resources(department, entity_type)
            commands = self.NORMAL_COMMANDS[entity_type]
            avg_duration = self._average_duration(entity_type)
            duration_range = (max(4, avg_duration - 12), avg_duration + 18)
            device = self._make_device_fingerprint(f"{entity_type}-{department}-{idx}-{region}")
            profiles.append(
                EntityProfile(
                    entity_id=f"ENT-{idx + 1:04d}",
                    entity_type=entity_type,
                    department=department,
                    seniority=seniority,
                    home_region=region,
                    preferred_auth=auth,
                    common_resources=resources,
                    normal_commands=commands,
                    avg_session_duration=avg_duration,
                    session_duration_range=duration_range,
                    base_device=device,
                )
            )
        return profiles

    def _generate_normal_log(self, profile: Optional[EntityProfile] = None) -> Dict[str, Any]:
        """Generate a single normal telemetry event."""
        profile = profile or self._rng.choice(self._profiles)
        timestamp = self._sample_timestamp(profile, suspicious=False)
        source_ip = self._generate_ip(profile.home_region, public_bias=0.18)
        resource = self._rng.choice(profile.common_resources)
        auth_method = self._normal_auth(profile)
        duration = self._bounded_int(
            profile.avg_session_duration,
            spread=18,
            minimum=profile.session_duration_range[0],
            maximum=profile.session_duration_range[1],
        )
        command_sequence = self._normal_command_sequence(profile)

        event = self._build_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method=auth_method,
            session_duration=duration,
            command_sequence=command_sequence,
            device_fingerprint=profile.base_device,
            label="Normal",
        )
        self._events_by_entity.setdefault(profile.entity_id, []).append(event)
        return event

    def _generate_attack_logs(self, total_attacks: int) -> List[Dict[str, Any]]:
        """Generate attack telemetry with balanced class representation."""
        attack_counts = self._build_attack_counts(total_attacks)

        logs: List[Dict[str, Any]] = []
        generators = {
            "Brute Force": self._attack_brute_force,
            "Impossible Travel": self._attack_impossible_travel,
            "Credential Stuffing": self._attack_credential_stuffing,
            "Lateral Movement": self._attack_lateral_movement,
            "Device Spoofing": self._attack_device_spoofing,
            "Low-and-Slow Exfiltration": self._attack_low_and_slow_exfiltration,
            "Insider Drift": self._attack_insider_drift,
        }

        for label, count in attack_counts.items():
            for _ in range(count):
                logs.append(generators[label]())
        return logs

    def _build_attack_counts(self, total_attacks: int) -> Dict[str, int]:
        """Distribute attacks across categories while preserving configured attack volume."""
        if total_attacks < len(self.ATTACK_LABELS):
            raise ValueError("total_attacks must be at least the number of attack categories.")

        base_count = total_attacks // len(self.ATTACK_LABELS)
        attack_counts = {label: base_count for label in self.ATTACK_LABELS}
        remainder = total_attacks - sum(attack_counts.values())
        labels = list(self.ATTACK_LABELS)
        self._rng.shuffle(labels)
        for idx in range(remainder):
            attack_counts[labels[idx % len(labels)]] += 1
        return attack_counts

    def _attack_brute_force(self) -> Dict[str, Any]:
        """Generate repeated failed-auth style activity against a single identity."""
        profile = self._select_attack_profile(("user", "service_account"))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        region = self._rng.choice(tuple(self.REGIONAL_SUBNETS.keys()))
        source_ip = self._generate_ip(region, public_bias=0.95)
        resource = self._rng.choice((f"vpn://gateway/{profile.department}", "ad://corp/directory"))
        duration = self._bounded_int(3, spread=2, minimum=1, maximum=12)
        commands = "login_fail;login_fail;login_fail;password_reset_attempt"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method="password",
            session_duration=duration,
            command_sequence=commands,
            device_fingerprint=self._make_device_fingerprint(f"bruteforce-{source_ip}-{timestamp}"),
            label="Brute Force",
        )

    def _attack_impossible_travel(self) -> Dict[str, Any]:
        """Generate near-consecutive access from a distant region for the same entity."""
        profile = self._select_attack_profile(("user",))
        prior_event = self._latest_event_for_entity(profile.entity_id)
        if prior_event is not None:
            base_time = pd.to_datetime(prior_event["timestamp"]) + timedelta(minutes=self._rng.randint(15, 45))
            source_region = self._infer_region_from_ip(prior_event["source_ip"])
        else:
            base_time = self._sample_timestamp(profile, suspicious=True)
            source_region = profile.home_region

        far_regions = [region for region in self.REGIONAL_SUBNETS if region != source_region]
        destination_region = self._rng.choice(far_regions)
        source_ip = self._generate_ip(destination_region, public_bias=0.85)

        return self._record_attack_event(
            profile=profile,
            timestamp=base_time,
            source_ip=source_ip,
            resource_accessed=self._rng.choice(profile.common_resources),
            auth_method=self._normal_auth(profile),
            session_duration=self._bounded_int(profile.avg_session_duration, spread=10, minimum=5, maximum=120),
            command_sequence="login;open_app;view_record;logout",
            device_fingerprint=profile.base_device,
            label="Impossible Travel",
        )

    def _attack_credential_stuffing(self) -> Dict[str, Any]:
        """Generate credential reuse attempts across exposed services."""
        profile = self._select_attack_profile(("user", "service_account"))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        source_ip = self._generate_ip(self._rng.choice(tuple(self.REGIONAL_SUBNETS.keys())), public_bias=0.98)
        resource = self._rng.choice(
            (
                "sso://okta/login",
                "vpn://gateway/remote",
                "email://o365/inbox",
                "crm://salesforce/login",
            )
        )
        commands = "login_fail;login_fail;login;open_app"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method="password",
            session_duration=self._bounded_int(6, spread=3, minimum=1, maximum=20),
            command_sequence=commands,
            device_fingerprint=self._make_device_fingerprint(f"stuffing-{source_ip}-{profile.entity_id}"),
            label="Credential Stuffing",
        )

    def _attack_lateral_movement(self) -> Dict[str, Any]:
        """Generate pivot behavior into privileged internal resources."""
        profile = self._select_attack_profile(("service_account", "edge_device", "user"))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        source_ip = self._generate_ip(profile.home_region, public_bias=0.10)
        resource = self._rng.choice(
            (
                "ad://corp/domain-admins",
                "rmm://windows/domain-controllers",
                "k8s://cluster/prod-kube-system",
                "vault://secrets/rotation",
            )
        )
        commands = "login;enumerate_hosts;remote_exec;privilege_probe;session_clone"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method=self._rng.choice(("password", "mfa_push", "api_token")),
            session_duration=self._bounded_int(48, spread=18, minimum=12, maximum=180),
            command_sequence=commands,
            device_fingerprint=profile.base_device,
            label="Lateral Movement",
        )

    def _attack_device_spoofing(self) -> Dict[str, Any]:
        """Generate a familiar user logging in from an unfamiliar device signature."""
        profile = self._select_attack_profile(("user", "edge_device"))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        source_ip = self._generate_ip(profile.home_region, public_bias=0.42)
        commands = "login;browser_check;open_app;download_record;logout"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=self._rng.choice(profile.common_resources),
            auth_method=self._normal_auth(profile),
            session_duration=self._bounded_int(profile.avg_session_duration, spread=12, minimum=5, maximum=180),
            command_sequence=commands,
            device_fingerprint=self._make_device_fingerprint(
                f"spoofed-{profile.entity_id}-{timestamp}-{source_ip}"
            ),
            label="Device Spoofing",
        )

    def _attack_low_and_slow_exfiltration(self) -> Dict[str, Any]:
        """Generate long-lived, low-volume data collection activity."""
        profile = self._select_attack_profile(("user", "service_account"))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        source_ip = self._generate_ip(profile.home_region, public_bias=0.20)
        resource = self._rng.choice(
            (
                "db://warehouse/pii_export",
                "share://finance/quarterly-close",
                "s3://legal-hold-archives",
                "db://warehouse/revenue_mart",
            )
        )
        commands = "login;query_small_batch;export_csv_chunk;compress_chunk;logout"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method=self._normal_auth(profile),
            session_duration=self._bounded_int(95, spread=35, minimum=30, maximum=360),
            command_sequence=commands,
            device_fingerprint=profile.base_device,
            label="Low-and-Slow Exfiltration",
        )

    def _attack_insider_drift(self) -> Dict[str, Any]:
        """Generate gradual access drift into unusual resources and commands."""
        profile = self._select_attack_profile(("user",))
        timestamp = self._sample_timestamp(profile, suspicious=True)
        source_ip = self._generate_ip(profile.home_region, public_bias=0.15)
        resource = self._rng.choice(self.SUSPICIOUS_RESOURCES)
        commands = "login;search_directory;open_sensitive_doc;bulk_lookup;logout"

        return self._record_attack_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource,
            auth_method=self._normal_auth(profile),
            session_duration=self._bounded_int(38, spread=14, minimum=8, maximum=150),
            command_sequence=commands,
            device_fingerprint=profile.base_device,
            label="Insider Drift",
        )

    def _record_attack_event(
        self,
        profile: EntityProfile,
        timestamp: datetime,
        source_ip: str,
        resource_accessed: str,
        auth_method: str,
        session_duration: int,
        command_sequence: str,
        device_fingerprint: str,
        label: str,
    ) -> Dict[str, Any]:
        """Build and store an attack event for entity-level continuity."""
        event = self._build_event(
            profile=profile,
            timestamp=timestamp,
            source_ip=source_ip,
            resource_accessed=resource_accessed,
            auth_method=auth_method,
            session_duration=session_duration,
            command_sequence=command_sequence,
            device_fingerprint=device_fingerprint,
            label=label,
        )
        self._events_by_entity.setdefault(profile.entity_id, []).append(event)
        return event

    def _build_event(
        self,
        profile: EntityProfile,
        timestamp: datetime,
        source_ip: str,
        resource_accessed: str,
        auth_method: str,
        session_duration: int,
        command_sequence: str,
        device_fingerprint: str,
        label: str,
    ) -> Dict[str, Any]:
        """Standardize event construction."""
        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": timestamp,
            "source_ip": source_ip,
            "geo_location": self.GEO_LOCATIONS[self._infer_region_from_ip(source_ip)],
            "resource_accessed": resource_accessed,
            "auth_method": auth_method,
            "session_duration": int(session_duration),
            "command_sequence": command_sequence,
            "device_fingerprint": device_fingerprint,
            "label": label,
        }

    def _sample_timestamp(self, profile: EntityProfile, suspicious: bool) -> datetime:
        """Sample timestamps with business-hour bias for normal activity."""
        if suspicious:
            latest_event = self._latest_event_for_entity(profile.entity_id)
            if latest_event is not None:
                latest_timestamp = pd.to_datetime(latest_event["timestamp"]).to_pydatetime()
                return latest_timestamp + timedelta(
                    minutes=self._rng.randint(2, 180),
                    seconds=int(self._np_rng.integers(0, 60)),
                )
        day_offset = int(self._np_rng.integers(0, 30))
        if suspicious:
            hour = int(self._rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))
            minute = int(self._np_rng.integers(0, 60))
        else:
            if profile.entity_type in {"service_account", "edge_device"}:
                hour = int(self._np_rng.integers(0, 24))
            else:
                hour = int(self._rng.choices([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18], k=1)[0])
            minute = int(self._np_rng.integers(0, 60))
        second = int(self._np_rng.integers(0, 60))
        return self._base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

    def _generate_ip(self, region: str, public_bias: float) -> str:
        """Generate an IP address from a regional subnet pool."""
        internal_cidr, public_cidr = self.REGIONAL_SUBNETS[region]
        cidr = public_cidr if self._rng.random() < public_bias else internal_cidr
        network = ipaddress.ip_network(cidr)
        offset = self._rng.randint(10, min(network.num_addresses - 2, 240))
        return str(network.network_address + offset)

    def _infer_region_from_ip(self, source_ip: str) -> str:
        """Infer the originating region from the configured subnet pool."""
        ip_obj = ipaddress.ip_address(source_ip)
        for region, cidrs in self.REGIONAL_SUBNETS.items():
            if any(ip_obj in ipaddress.ip_network(cidr) for cidr in cidrs):
                return region
        return self._rng.choice(tuple(self.REGIONAL_SUBNETS.keys()))

    def _normal_auth(self, profile: EntityProfile) -> str:
        """Sample a typical authentication method for the entity."""
        if profile.entity_type == "service_account":
            return self._rng.choice(("api_token", "vpn_cert"))
        if profile.entity_type == "edge_device":
            return self._rng.choice(("api_token", "vpn_cert"))
        return self._rng.choices(
            (profile.preferred_auth, "sso", "mfa_push", "webauthn", "password"),
            weights=(0.45, 0.20, 0.15, 0.10, 0.10),
            k=1,
        )[0]

    def _normal_command_sequence(self, profile: EntityProfile) -> str:
        """Build a short, realistic command sequence for normal activity."""
        base = list(profile.normal_commands)
        sequence_length = self._rng.randint(3, min(5, len(base)))
        selected = base[:1] + self._rng.sample(base[1:], sequence_length - 1)
        return ";".join(selected)

    def _preferred_auth(self, entity_type: str) -> str:
        """Select a stable primary authentication method by entity type."""
        mapping = {
            "user": self._rng.choice(("sso", "mfa_push", "webauthn", "password")),
            "service_account": self._rng.choice(("api_token", "vpn_cert")),
            "edge_device": self._rng.choice(("api_token", "vpn_cert")),
        }
        return mapping[entity_type]

    def _select_common_resources(self, department: str, entity_type: str) -> Tuple[str, ...]:
        """Assign common resources with entity-type-aware access patterns."""
        resources = list(self.RESOURCES_BY_DEPARTMENT[department])
        if entity_type == "service_account":
            resources.extend(("queue://kafka/etl-events", "db://warehouse/feature_store"))
        if entity_type == "edge_device":
            resources.extend(("mqtt://plant/telemetry", "iot://fleet/device-management"))
        self._rng.shuffle(resources)
        return tuple(resources[:4])

    def _average_duration(self, entity_type: str) -> int:
        """Return baseline session duration in minutes."""
        baseline = {
            "user": 32,
            "service_account": 14,
            "edge_device": 8,
        }
        return baseline[entity_type]

    def _make_device_fingerprint(self, seed_text: str) -> str:
        """Create a stable pseudo-device fingerprint."""
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return f"dfp-{digest[:16]}"

    def _bounded_int(self, center: int, spread: int, minimum: int, maximum: int) -> int:
        """Sample an integer and clamp it into a valid range."""
        value = int(round(self._np_rng.normal(center, max(1, spread / 2))))
        return max(minimum, min(maximum, value))

    def _select_attack_profile(self, preferred_types: Sequence[str]) -> EntityProfile:
        """Select an entity profile for a given attack family."""
        candidates = [profile for profile in self._profiles if profile.entity_type in preferred_types]
        return self._rng.choice(candidates if candidates else self._profiles)

    def _latest_event_for_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest event for an entity if one exists."""
        events = self._events_by_entity.get(entity_id, [])
        if not events:
            return None
        return max(events, key=lambda event: pd.to_datetime(event["timestamp"]))


class AdaptiveBehaviourContextEngine:
    """Build adaptive per-entity behaviour profiles and engineered context features."""

    def __init__(self, window_size: int = CONFIG["WINDOW_SIZE"]) -> None:
        """Initialize the engine and preprocessing components."""
        self.window_size = int(window_size)
        self.entity_profiles: Dict[str, Dict[str, Any]] = {}
        self.cohort_profiles: Dict[str, Dict[str, Any]] = {}
        self._fitted = False
        self._feature_columns: List[str] = []
        self._categorical_columns = ["auth_method", "resource_category"]
        self._numeric_columns = [
            "login_hour",
            "session_duration",
            "new_device_indicator",
            "ip_subnet_change",
            "time_since_previous_login",
            "login_frequency",
        ]
        self.encoder = self._build_encoder()
        self.scaler = StandardScaler()

    def fit_transform(self, telemetry_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit behavioural profiles and transform telemetry into flat and sequential features."""
        prepared_df = self._prepare_dataframe(telemetry_df)
        featured_df = self._engineer_features(prepared_df, update_profiles=True)
        flat_features = self._fit_preprocessors(featured_df)
        sequences, sequence_entity_ids = self._build_sequences(featured_df, flat_features)
        return {
            "flat_features": flat_features,
            "flat_feature_frame": featured_df[self._feature_columns].copy(),
            "sequence_windows": sequences,
            "sequence_entity_ids": sequence_entity_ids,
            "entity_profiles": self.entity_profiles,
        }

    def transform(self, telemetry_df: pd.DataFrame) -> Dict[str, Any]:
        """Transform telemetry using existing profiles and fitted preprocessors."""
        if not self._fitted:
            raise RuntimeError("AdaptiveBehaviourContextEngine must be fitted before transform.")
        prepared_df = self._prepare_dataframe(telemetry_df)
        featured_df = self._engineer_features(prepared_df, update_profiles=False)
        flat_features = self._transform_with_preprocessors(featured_df)
        sequences, sequence_entity_ids = self._build_sequences(featured_df, flat_features)
        return {
            "flat_features": flat_features,
            "flat_feature_frame": featured_df[self._feature_columns].copy(),
            "sequence_windows": sequences,
            "sequence_entity_ids": sequence_entity_ids,
            "entity_profiles": self.entity_profiles,
        }

    def get_entity_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Return the current per-entity behavioural profiles."""
        return self.entity_profiles

    def _prepare_dataframe(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the telemetry input."""
        try:
            required_columns = {
                "entity_id",
                "entity_type",
                "timestamp",
                "source_ip",
                "geo_location",
                "resource_accessed",
                "auth_method",
                "session_duration",
                "device_fingerprint",
            }
            missing_columns = required_columns.difference(telemetry_df.columns)
            if missing_columns:
                raise ValueError(f"Missing required telemetry columns: {sorted(missing_columns)}")

            prepared_df = telemetry_df.copy()
            prepared_df["timestamp"] = pd.to_datetime(prepared_df["timestamp"], errors="coerce")
            if prepared_df["timestamp"].isna().any():
                raise ValueError("One or more timestamps could not be parsed.")
            prepared_df["session_duration"] = pd.to_numeric(
                prepared_df["session_duration"], errors="coerce"
            )
            if prepared_df["session_duration"].isna().any():
                raise ValueError("One or more session_duration values are invalid.")
            prepared_df["auth_method"] = prepared_df["auth_method"].fillna("unknown").astype(str)
            prepared_df["resource_accessed"] = prepared_df["resource_accessed"].fillna("unknown://unknown")
            prepared_df["device_fingerprint"] = prepared_df["device_fingerprint"].fillna("unknown_device")
            prepared_df["entity_type"] = prepared_df["entity_type"].fillna("unknown").astype(str)
            prepared_df["source_ip"] = prepared_df["source_ip"].fillna("0.0.0.0").astype(str)
            prepared_df["geo_location"] = prepared_df["geo_location"].fillna("Unknown").astype(str)
            prepared_df["entity_id"] = prepared_df["entity_id"].astype(str)
            prepared_df = prepared_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
            return prepared_df
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare telemetry data: {exc}") from exc

    def _engineer_features(self, telemetry_df: pd.DataFrame, update_profiles: bool) -> pd.DataFrame:
        """Create flat engineered features while adaptively maintaining profiles."""
        try:
            self.cohort_profiles = self._compute_cohort_profiles(telemetry_df)
            history_by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            feature_rows: List[Dict[str, Any]] = []

            for row in telemetry_df.itertuples(index=False):
                entity_id = str(row.entity_id)
                entity_type = str(row.entity_type)
                history = history_by_entity[entity_id]
                cohort_profile = self.cohort_profiles.get(entity_type, self._default_cohort_profile())
                baseline_profile = (
                    self._profile_from_history(history)
                    if len(history) >= self.window_size
                    else cohort_profile
                )

                login_hour = int(row.timestamp.hour)
                session_duration = float(row.session_duration)
                subnet = self._extract_subnet(str(row.source_ip))
                resource_category = self._resource_category(str(row.resource_accessed))
                time_since_previous = self._time_since_previous_login(history, row.timestamp)
                login_frequency = self._login_frequency(history, row.timestamp)
                new_device_indicator = float(self._is_new_device(history, str(row.device_fingerprint)))
                ip_subnet_change = float(subnet != baseline_profile["common_ip_subnet"])

                feature_row = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "timestamp": row.timestamp,
                    "login_hour": login_hour,
                    "session_duration": session_duration,
                    "new_device_indicator": new_device_indicator,
                    "ip_subnet_change": ip_subnet_change,
                    "auth_method": str(row.auth_method),
                    "resource_category": resource_category,
                    "time_since_previous_login": time_since_previous,
                    "login_frequency": login_frequency,
                    "preferred_login_hour": float(baseline_profile["preferred_login_hour"]),
                    "typical_session_duration": float(baseline_profile["typical_session_duration"]),
                    "common_ip_subnet": baseline_profile["common_ip_subnet"],
                    "frequently_accessed_resources": tuple(
                        baseline_profile["frequently_accessed_resources"]
                    ),
                }
                feature_rows.append(feature_row)

                history.append(
                    {
                        "timestamp": row.timestamp,
                        "login_hour": login_hour,
                        "session_duration": session_duration,
                        "source_ip": str(row.source_ip),
                        "subnet": subnet,
                        "resource_accessed": str(row.resource_accessed),
                        "auth_method": str(row.auth_method),
                        "device_fingerprint": str(row.device_fingerprint),
                        "resource_category": resource_category,
                    }
                )

            featured_df = pd.DataFrame(feature_rows)
            self._feature_columns = self._numeric_columns + self._categorical_columns
            if update_profiles:
                self.entity_profiles = {
                    entity_id: self._profile_from_history(history)
                    for entity_id, history in history_by_entity.items()
                }
            return featured_df
        except Exception as exc:
            raise RuntimeError(f"Failed to engineer adaptive behaviour features: {exc}") from exc

    def _fit_preprocessors(self, featured_df: pd.DataFrame) -> np.ndarray:
        """Fit OneHotEncoder and StandardScaler, then return flat features."""
        try:
            numeric_matrix = self.scaler.fit_transform(featured_df[self._numeric_columns])
            categorical_matrix = self.encoder.fit_transform(featured_df[self._categorical_columns])
            flat_features = np.hstack([numeric_matrix, categorical_matrix])
            self._fitted = True
            return flat_features.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to fit behaviour preprocessors: {exc}") from exc

    def _transform_with_preprocessors(self, featured_df: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessors to engineered features."""
        try:
            numeric_matrix = self.scaler.transform(featured_df[self._numeric_columns])
            categorical_matrix = self.encoder.transform(featured_df[self._categorical_columns])
            flat_features = np.hstack([numeric_matrix, categorical_matrix])
            return flat_features.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to transform behaviour features: {exc}") from exc

    def _build_sequences(
        self,
        featured_df: pd.DataFrame,
        flat_features: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Create sequential windows of fixed length for each entity."""
        try:
            sequences: List[np.ndarray] = []
            sequence_entity_ids: List[str] = []
            for entity_id, entity_frame in featured_df.groupby("entity_id", sort=False):
                entity_indices = entity_frame.index.to_list()
                entity_vectors = flat_features[entity_indices]
                if len(entity_vectors) < self.window_size:
                    continue
                for start_idx in range(0, len(entity_vectors) - self.window_size + 1):
                    window = entity_vectors[start_idx : start_idx + self.window_size]
                    sequences.append(window)
                    sequence_entity_ids.append(entity_id)
            if not sequences:
                return np.empty((0, self.window_size, flat_features.shape[1]), dtype=np.float32), []
            return np.stack(sequences).astype(np.float32), sequence_entity_ids
        except Exception as exc:
            raise RuntimeError(f"Failed to build sequential windows: {exc}") from exc

    def _compute_cohort_profiles(self, telemetry_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Estimate cohort behaviour for cold-start entities grouped by entity type."""
        cohort_profiles: Dict[str, Dict[str, Any]] = {}
        for entity_type, cohort_frame in telemetry_df.groupby("entity_type"):
            login_hours = cohort_frame["timestamp"].dt.hour.tolist()
            durations = cohort_frame["session_duration"].astype(float).tolist()
            subnets = [self._extract_subnet(ip) for ip in cohort_frame["source_ip"].astype(str)]
            resources = cohort_frame["resource_accessed"].astype(str).tolist()
            cohort_profiles[str(entity_type)] = {
                "preferred_login_hour": float(np.median(login_hours)) if login_hours else 12.0,
                "typical_session_duration": float(np.median(durations)) if durations else 30.0,
                "common_ip_subnet": self._most_common(subnets, default="0.0.0.0/24"),
                "frequently_accessed_resources": tuple(self._top_k(resources)),
            }
        return cohort_profiles

    def _profile_from_history(self, history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize an entity's current behavioural profile from its prior observations."""
        if not history:
            return self._default_cohort_profile()
        login_hours = [event["login_hour"] for event in history]
        durations = [event["session_duration"] for event in history]
        subnets = [event["subnet"] for event in history]
        resources = [event["resource_accessed"] for event in history]
        return {
            "preferred_login_hour": float(np.median(login_hours)),
            "typical_session_duration": float(np.median(durations)),
            "common_ip_subnet": self._most_common(subnets, default="0.0.0.0/24"),
            "frequently_accessed_resources": tuple(self._top_k(resources)),
        }

    def _default_cohort_profile(self) -> Dict[str, Any]:
        """Fallback profile when no cohort data is available."""
        return {
            "preferred_login_hour": 12.0,
            "typical_session_duration": 30.0,
            "common_ip_subnet": "0.0.0.0/24",
            "frequently_accessed_resources": ("unknown://unknown",),
        }

    def _time_since_previous_login(
        self,
        history: Sequence[Dict[str, Any]],
        current_timestamp: pd.Timestamp,
    ) -> float:
        """Return hours since the entity's previous login."""
        if not history:
            return float(24.0)
        previous_timestamp = history[-1]["timestamp"]
        delta_seconds = max((current_timestamp - previous_timestamp).total_seconds(), 0.0)
        return float(delta_seconds / 3600.0)

    def _login_frequency(
        self,
        history: Sequence[Dict[str, Any]],
        current_timestamp: pd.Timestamp,
        lookback_hours: int = 24,
    ) -> float:
        """Count prior logins within a recent time window."""
        if not history:
            return 0.0
        threshold = current_timestamp - pd.Timedelta(hours=lookback_hours)
        recent_events = [event for event in history if event["timestamp"] >= threshold]
        return float(len(recent_events))

    def _is_new_device(self, history: Sequence[Dict[str, Any]], device_fingerprint: str) -> bool:
        """Flag whether a device has been seen before for the entity."""
        if not history:
            return True
        known_devices = {event["device_fingerprint"] for event in history}
        return device_fingerprint not in known_devices

    def _extract_subnet(self, source_ip: str) -> str:
        """Normalize an IP address into a coarse /24 subnet representation."""
        try:
            ip_obj = ipaddress.ip_address(source_ip)
            network = ipaddress.ip_network(f"{ip_obj}/24", strict=False)
            return str(network)
        except ValueError:
            return "0.0.0.0/24"

    def _resource_category(self, resource_accessed: str) -> str:
        """Convert a resource URI into a stable coarse category."""
        if "://" not in resource_accessed:
            return "unknown"
        return resource_accessed.split("://", maxsplit=1)[0].strip().lower() or "unknown"

    def _most_common(self, values: Sequence[str], default: str) -> str:
        """Return the most frequent value from a sequence."""
        if not values:
            return default
        return Counter(values).most_common(1)[0][0]

    def _top_k(self, values: Sequence[str], k: int = 3) -> List[str]:
        """Return the most frequent values in descending order."""
        if not values:
            return ["unknown://unknown"]
        return [value for value, _ in Counter(values).most_common(k)]

    def _build_encoder(self) -> OneHotEncoder:
        """Create a version-compatible OneHotEncoder instance."""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


class ConformalGRUAutoencoder(nn.Module):
    """GRU autoencoder with split-conformal calibration for sequence anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        window_size: int = CONFIG["WINDOW_SIZE"],
        hidden_dim: int = 64,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 20,
        calibration_fraction: float = 0.2,
        target_fpr: float = CONFIG["TARGET_FPR"],
        seed: int = CONFIG["SEED"],
        device: Optional[str] = None,
    ) -> None:
        """Initialize the conformal GRU autoencoder."""
        super().__init__()
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.calibration_fraction = float(calibration_fraction)
        self.target_fpr = float(target_fpr)
        self.seed = int(seed)
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_name)

        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.latent_projection = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.window_size * self.input_dim),
        )
        self.loss_fn = nn.MSELoss()
        self.threshold_: Optional[float] = None
        self.calibration_errors_: Optional[np.ndarray] = None
        self.training_history_: List[float] = []
        self.is_fitted_: bool = False
        self.to(self.device)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode input sequences and reconstruct their flattened temporal features."""
        _, hidden_state = self.encoder(sequences)
        latent = self.latent_projection(hidden_state[-1])
        reconstructed = self.decoder(latent)
        return reconstructed.view(-1, self.window_size, self.input_dim)

    def fit(self, sequences: np.ndarray, labels: Optional[Sequence[Any]] = None) -> "ConformalGRUAutoencoder":
        """Train on benign sequences only and calibrate an adaptive conformal threshold."""
        try:
            self._set_deterministic_state()
            benign_sequences = self._select_benign_sequences(sequences, labels)
            if benign_sequences.shape[0] < 3:
                raise ValueError("At least 3 benign sequences are required for train/calibration split.")

            train_sequences, calibration_sequences = self._split_train_calibration(benign_sequences)
            train_loader = self._build_dataloader(train_sequences, shuffle=True)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

            self.train()
            self.training_history_ = []
            for _ in range(self.epochs):
                epoch_loss = 0.0
                sample_count = 0
                for (batch_sequences,) in train_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    optimizer.zero_grad()
                    reconstructed = self.forward(batch_sequences)
                    loss = self.loss_fn(reconstructed, batch_sequences)
                    loss.backward()
                    optimizer.step()
                    batch_size = batch_sequences.size(0)
                    epoch_loss += float(loss.item()) * batch_size
                    sample_count += batch_size
                self.training_history_.append(epoch_loss / max(sample_count, 1))

            calibration_errors = self.compute_reconstruction_error(calibration_sequences)
            self.calibration_errors_ = calibration_errors
            self.threshold_ = float(np.quantile(calibration_errors, 1.0 - self.target_fpr, method="higher"))
            self.is_fitted_ = True
            return self
        except TypeError:
            try:
                self._set_deterministic_state()
                benign_sequences = self._select_benign_sequences(sequences, labels)
                if benign_sequences.shape[0] < 3:
                    raise ValueError("At least 3 benign sequences are required for train/calibration split.")

                train_sequences, calibration_sequences = self._split_train_calibration(benign_sequences)
                train_loader = self._build_dataloader(train_sequences, shuffle=True)
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

                self.train()
                self.training_history_ = []
                for _ in range(self.epochs):
                    epoch_loss = 0.0
                    sample_count = 0
                    for (batch_sequences,) in train_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        optimizer.zero_grad()
                        reconstructed = self.forward(batch_sequences)
                        loss = self.loss_fn(reconstructed, batch_sequences)
                        loss.backward()
                        optimizer.step()
                        batch_size = batch_sequences.size(0)
                        epoch_loss += float(loss.item()) * batch_size
                        sample_count += batch_size
                    self.training_history_.append(epoch_loss / max(sample_count, 1))

                calibration_errors = self.compute_reconstruction_error(calibration_sequences)
                self.calibration_errors_ = calibration_errors
                self.threshold_ = float(np.quantile(calibration_errors, 1.0 - self.target_fpr))
                self.is_fitted_ = True
                return self
            except Exception as exc:
                raise RuntimeError(f"Failed to train and calibrate ConformalGRUAutoencoder: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to train and calibrate ConformalGRUAutoencoder: {exc}") from exc

    def predict(self, sequences: np.ndarray) -> Dict[str, np.ndarray]:
        """Return reconstruction error, risk score, and anomaly prediction for each sequence."""
        if not self.is_fitted_ or self.threshold_ is None:
            raise RuntimeError("ConformalGRUAutoencoder must be fitted before prediction.")
        reconstruction_error = self.compute_reconstruction_error(sequences)
        risk_score = self._risk_score(reconstruction_error)
        anomaly_prediction = reconstruction_error > self.threshold_
        return {
            "reconstruction_error": reconstruction_error,
            "risk_score": risk_score,
            "anomaly_prediction": anomaly_prediction,
        }

    def fit_predict(
        self,
        train_sequences: np.ndarray,
        inference_sequences: np.ndarray,
        train_labels: Optional[Sequence[Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Fit on benign data and immediately score inference sequences."""
        self.fit(train_sequences, train_labels)
        return self.predict(inference_sequences)

    def compute_reconstruction_error(self, sequences: np.ndarray) -> np.ndarray:
        """Compute mean squared reconstruction error per input sequence."""
        validated = self._validate_sequences(sequences)
        dataloader = self._build_dataloader(validated, shuffle=False)
        self.eval()
        errors: List[np.ndarray] = []
        with torch.no_grad():
            for (batch_sequences,) in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                reconstructed = self.forward(batch_sequences)
                batch_errors = torch.mean((reconstructed - batch_sequences) ** 2, dim=(1, 2))
                errors.append(batch_errors.detach().cpu().numpy())
        if not errors:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(errors).astype(np.float32)

    def _select_benign_sequences(
        self,
        sequences: np.ndarray,
        labels: Optional[Sequence[Any]],
    ) -> np.ndarray:
        """Filter input to benign-only sequences for training and calibration."""
        validated = self._validate_sequences(sequences)
        if labels is None:
            return validated
        label_array = np.asarray(labels)
        if label_array.shape[0] != validated.shape[0]:
            raise ValueError("labels must align one-to-one with sequences.")
        benign_mask = np.isin(
            np.char.lower(label_array.astype(str)),
            np.array(["normal", "benign"]),
        )
        benign_sequences = validated[benign_mask]
        if benign_sequences.size == 0:
            raise ValueError("No benign sequences were available for training.")
        return benign_sequences

    def _split_train_calibration(self, benign_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a split-conformal benign train/calibration partition."""
        indices = np.arange(benign_sequences.shape[0])
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        shuffled = benign_sequences[indices]
        calibration_size = max(1, int(round(shuffled.shape[0] * self.calibration_fraction)))
        calibration_size = min(calibration_size, shuffled.shape[0] - 1)
        split_index = shuffled.shape[0] - calibration_size
        train_sequences = shuffled[:split_index]
        calibration_sequences = shuffled[split_index:]
        if train_sequences.shape[0] == 0 or calibration_sequences.shape[0] == 0:
            raise ValueError("Unable to create non-empty train and calibration splits.")
        return train_sequences, calibration_sequences

    def _build_dataloader(self, sequences: np.ndarray, shuffle: bool) -> DataLoader:
        """Create a DataLoader for batched GRU training or inference."""
        tensor_sequences = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(tensor_sequences)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _validate_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Validate and normalize input sequence arrays."""
        array = np.asarray(sequences, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(
                "sequences must have shape (n_samples, window_size, input_dim)."
            )
        if array.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window_size={self.window_size}, received {array.shape[1]}."
            )
        if array.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {array.shape[2]}."
            )
        return array

    def _risk_score(self, reconstruction_error: np.ndarray) -> np.ndarray:
        """Map reconstruction error to a bounded 0-100 risk score."""
        if self.threshold_ is None or self.threshold_ <= 0:
            return np.zeros_like(reconstruction_error, dtype=np.float32)
        scaled = 100.0 * (reconstruction_error / self.threshold_)
        clipped = np.clip(scaled, 0.0, 100.0)
        return clipped.astype(np.float32)

    def _set_deterministic_state(self) -> None:
        """Reset random state before training to keep calibration reproducible."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class BaselineIsolationForest:
    """Isolation Forest baseline operating on engineered flat behaviour features."""

    def __init__(
        self,
        contamination: float = CONFIG["ATTACK_RATE"],
        n_estimators: int = 200,
        random_state: int = CONFIG["SEED"],
    ) -> None:
        """Initialize the baseline anomaly detector."""
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.is_fitted_: bool = False

    def fit(self, features: np.ndarray) -> "BaselineIsolationForest":
        """Train IsolationForest on engineered feature vectors."""
        try:
            validated_features = self._validate_features(features)
            self.model.fit(validated_features)
            self.is_fitted_ = True
            return self
        except Exception as exc:
            raise RuntimeError(f"Failed to train BaselineIsolationForest: {exc}") from exc

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Return binary predictions and anomaly scores for inference."""
        if not self.is_fitted_:
            raise RuntimeError("BaselineIsolationForest must be fitted before prediction.")
        validated_features = self._validate_features(features)
        raw_predictions = self.model.predict(validated_features)
        anomaly_scores = -self.model.score_samples(validated_features)
        predictions = raw_predictions == -1
        return {
            "predictions": predictions,
            "anomaly_scores": anomaly_scores.astype(np.float32),
        }

    def fit_predict(self, train_features: np.ndarray, inference_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit the baseline model and immediately run inference."""
        self.fit(train_features)
        return self.predict(inference_features)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return IsolationForest anomaly scores for inference workflows."""
        if not self.is_fitted_:
            raise RuntimeError("BaselineIsolationForest must be fitted before scoring.")
        validated_features = self._validate_features(features)
        return (-self.model.score_samples(validated_features)).astype(np.float32)

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate and normalize flat feature inputs."""
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("features must have shape (n_samples, n_features).")
        if array.shape[0] == 0:
            raise ValueError("features must contain at least one sample.")
        return array


class MITREMapper:
    """Map anomaly signals into selected MITRE ATT&CK techniques and SOC actions."""

    CATEGORY_CATALOG: Dict[str, Dict[str, str]] = {
        "Brute Force": {
            "technique_id": "T1110",
            "technique_name": "Brute Force",
            "soc_recommendation": (
                "Review authentication failures, enforce MFA, rate-limit login attempts, "
                "and reset potentially exposed credentials."
            ),
        },
        "Impossible Travel": {
            "technique_id": "T1078",
            "technique_name": "Valid Accounts",
            "soc_recommendation": (
                "Validate account ownership, review recent successful logins, rotate credentials, "
                "and verify device and geolocation legitimacy."
            ),
        },
        "Credential Stuffing": {
            "technique_id": "T1110.004",
            "technique_name": "Credential Stuffing",
            "soc_recommendation": (
                "Block repeated credential reuse attempts, rotate exposed credentials, "
                "enforce MFA, and review successful sessions following failures."
            ),
        },
        "Lateral Movement": {
            "technique_id": "T1021",
            "technique_name": "Remote Services",
            "soc_recommendation": (
                "Inspect lateral access paths, isolate impacted hosts, review privileged sessions, "
                "and restrict remote service exposure."
            ),
        },
        "Device Spoofing": {
            "technique_id": "T1036",
            "technique_name": "Masquerading",
            "soc_recommendation": (
                "Validate device identity, revoke suspicious sessions, compare endpoint "
                "attestation, and review recent activity from the affected entity."
            ),
        },
        "Low-and-Slow Exfiltration": {
            "technique_id": "T1041",
            "technique_name": "Exfiltration Over C2 Channel",
            "soc_recommendation": (
                "Review long-running transfers, restrict access to sensitive stores, "
                "preserve flow evidence, and isolate the originating entity if confirmed."
            ),
        },
        "Insider Drift": {
            "technique_id": "T1078",
            "technique_name": "Valid Accounts",
            "soc_recommendation": (
                "Review access against the entity's role and historical baseline, validate "
                "business need, and contain access if the behavioral drift is unauthorized."
            ),
        },
    }

    def map_anomaly(
        self,
        anomaly_score: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict an attack category and map it to a MITRE technique."""
        try:
            context = context or {}
            category = self._predict_attack_category(context=context)
            technique = self.CATEGORY_CATALOG[category]
            conformal_score = float(context.get("conformal_score", anomaly_score))
            isolation_forest_score = float(
                context.get("isolation_forest_score", anomaly_score)
            )
            model_agreement = bool(
                context.get(
                    "model_agreement",
                    context.get("gru_prediction", False)
                    and context.get("if_prediction", False),
                )
            )
            confidence = self._calculate_prediction_confidence(
                anomaly_score=anomaly_score,
                conformal_score=conformal_score,
                isolation_forest_score=isolation_forest_score,
                model_agreement=model_agreement,
            )
            risk_score = self._calculate_risk_score(
                anomaly_score=anomaly_score,
                model_confidence=confidence,
                conformal_score=conformal_score,
                isolation_forest_score=isolation_forest_score,
                model_agreement=model_agreement,
            )
            severity = self._severity_from_score(risk_score, model_agreement)
            return {
                "predicted_attack_category": category,
                "prediction_confidence": confidence,
                "technique_id": technique["technique_id"],
                "technique_name": technique["technique_name"],
                "combined_risk_score": risk_score,
                "alert_score": risk_score,
                "severity": severity,
                "soc_recommendation": technique["soc_recommendation"],
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to map anomaly to MITRE ATT&CK: {exc}") from exc

    def map_batch(
        self,
        anomaly_scores: Sequence[float],
        contexts: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Map multiple anomalies into MITRE technique records."""
        try:
            score_list = list(anomaly_scores)
            context_list = list(contexts) if contexts is not None else [None] * len(score_list)
            if len(score_list) != len(context_list):
                raise ValueError("anomaly_scores and contexts must have the same length.")
            return [
                self.map_anomaly(anomaly_score=score, context=context)
                for score, context in zip(score_list, context_list)
            ]
        except Exception as exc:
            raise RuntimeError(f"Failed to batch map anomalies: {exc}") from exc

    def _predict_attack_category(self, context: Dict[str, Any]) -> str:
        """Infer an attack category strictly from behavioral context."""
        auth_method = str(context.get("auth_method", "")).lower()
        resource_category = str(context.get("classification_resource_categories", context.get("resource_category", ""))).lower()
        command_sequence = str(context.get("classification_command_sequence", context.get("command_sequence", ""))).lower()
        resource_accessed = str(context.get("classification_resources", context.get("resource_accessed", ""))).lower()
        ip_subnet_change = bool(context.get("ip_subnet_change", False))
        new_device_indicator = bool(context.get("new_device_indicator", False))
        login_frequency = float(context.get("login_frequency", 0.0))
        time_since_previous_login = float(context.get("time_since_previous_login", 24.0))

        if "password_reset_attempt" in command_sequence or command_sequence.count("login_fail") >= 3:
            return "Brute Force"
        if "login_fail" in command_sequence and "open_app" in command_sequence:
            return "Credential Stuffing"
        if any(marker in command_sequence for marker in ("remote_exec", "enumerate_hosts", "session_clone", "privilege_probe")) or "domain-admins" in resource_accessed:
            return "Lateral Movement"
        if any(marker in command_sequence for marker in ("query_small_batch", "export_csv_chunk", "compress_chunk")) or "pii_export" in resource_accessed:
            return "Low-and-Slow Exfiltration"
        if any(marker in command_sequence for marker in ("open_sensitive_doc", "bulk_lookup", "search_directory")):
            return "Insider Drift"
        if "browser_check" in command_sequence or new_device_indicator:
            return "Device Spoofing"
        if ip_subnet_change and time_since_previous_login <= 1.0:
            return "Impossible Travel"
        if resource_category in {"ad", "rmm", "k8s", "vault", "vpn"}:
            return "Lateral Movement"
        if login_frequency >= 8.0 or auth_method == "password" and time_since_previous_login < 0.25:
            return "Brute Force"
        return "Insider Drift"

    def _calculate_prediction_confidence(
        self,
        anomaly_score: float,
        conformal_score: float,
        isolation_forest_score: float,
        model_agreement: bool,
    ) -> float:
        """Estimate prediction confidence from model strength and consistency."""
        anomaly = min(max(float(anomaly_score), 0.0), 100.0)
        conformal = min(max(float(conformal_score), 0.0), 100.0)
        isolation_forest = min(max(float(isolation_forest_score), 0.0), 100.0)
        consistency = 100.0 - abs(conformal - isolation_forest)
        agreement_bonus = 10.0 if model_agreement else 0.0
        confidence = (
            0.40 * anomaly
            + 0.20 * conformal
            + 0.20 * isolation_forest
            + 0.20 * consistency
            + agreement_bonus
        )
        return round(min(max(confidence, 0.0), 99.0), 1)

    def _calculate_risk_score(
        self,
        anomaly_score: float,
        model_confidence: float,
        conformal_score: float,
        isolation_forest_score: float,
        model_agreement: bool,
    ) -> float:
        """Combine all detector evidence into the dashboard's 0-100 risk score."""
        anomaly = min(max(float(anomaly_score), 0.0), 100.0)
        confidence = min(max(float(model_confidence), 0.0), 100.0)
        conformal = min(max(float(conformal_score), 0.0), 100.0)
        isolation_forest = min(max(float(isolation_forest_score), 0.0), 100.0)
        agreement = 100.0 if model_agreement else 0.0
        score = (
            0.30 * anomaly
            + 0.20 * confidence
            + 0.20 * conformal
            + 0.20 * isolation_forest
            + 0.10 * agreement
        )
        return round(min(max(score, 0.0), 100.0), 1)

    def _severity_from_score(self, risk_score: float, model_agreement: bool = False) -> str:
        """Derive severity consistently from dynamic risk and model agreement."""
        score = float(risk_score)
        if model_agreement and score >= 80.0:
            return "Critical"
        if score >= 65.0:
            return "High"
        if score >= 58.0:
            return "Medium"
        return "Low"


class EvaluationEngine:
    """Compare anomaly detectors using classification and ranking metrics."""

    def evaluate_models(
        self,
        y_true: Sequence[Any],
        gru_results: Dict[str, np.ndarray],
        isolation_forest_results: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate ConformalGRUAutoencoder and BaselineIsolationForest side by side."""
        try:
            ground_truth = self._validate_labels(y_true)
            gru_metrics = self._evaluate_single_model(
                y_true=ground_truth,
                predictions=gru_results.get("anomaly_prediction"),
                scores=gru_results.get("risk_score", gru_results.get("reconstruction_error")),
            )
            isolation_forest_metrics = self._evaluate_single_model(
                y_true=ground_truth,
                predictions=isolation_forest_results.get("predictions"),
                scores=isolation_forest_results.get("anomaly_scores"),
            )
            return {
                "ConformalGRUAutoencoder": gru_metrics,
                "BaselineIsolationForest": isolation_forest_metrics,
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to evaluate anomaly detectors: {exc}") from exc

    def _evaluate_single_model(
        self,
        y_true: np.ndarray,
        predictions: Optional[np.ndarray],
        scores: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute metrics for one detector from actual predictions."""
        if predictions is None:
            raise ValueError("Model predictions are required for evaluation.")
        y_pred = self._validate_predictions(predictions, expected_length=y_true.shape[0])
        y_score = self._validate_scores(scores, expected_length=y_true.shape[0])

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_score)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "false_positive_rate": false_positive_rate,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    def _validate_labels(self, y_true: Sequence[Any]) -> np.ndarray:
        """Normalize ground-truth labels into a binary anomaly vector."""
        label_array = np.asarray(list(y_true))
        if label_array.ndim != 1 or label_array.shape[0] == 0:
            raise ValueError("y_true must be a non-empty one-dimensional sequence.")
        if label_array.dtype.kind in {"b", "i", "u", "f"}:
            return (label_array.astype(float) > 0).astype(int)
        normalized = np.char.lower(label_array.astype(str))
        benign_values = np.array(["normal", "benign", "0", "false"])
        return (~np.isin(normalized, benign_values)).astype(int)

    def _validate_predictions(self, predictions: Any, expected_length: int) -> np.ndarray:
        """Validate model prediction vectors."""
        pred_array = np.asarray(predictions)
        if pred_array.ndim != 1 or pred_array.shape[0] != expected_length:
            raise ValueError("Prediction vector length must match y_true.")
        if pred_array.dtype.kind in {"b", "i", "u", "f"}:
            return (pred_array.astype(float) > 0).astype(int)
        normalized = np.char.lower(pred_array.astype(str))
        positive_values = np.array(["1", "true", "anomaly", "attack"])
        return np.isin(normalized, positive_values).astype(int)

    def _validate_scores(self, scores: Optional[Any], expected_length: int) -> np.ndarray:
        """Validate anomaly score vectors used for PR-AUC."""
        if scores is None:
            raise ValueError("Model scores are required for PR-AUC evaluation.")
        score_array = np.asarray(scores, dtype=np.float32)
        if score_array.ndim != 1 or score_array.shape[0] != expected_length:
            raise ValueError("Score vector length must match y_true.")
        return score_array


class BackendEngine:
    """Orchestrate the complete backend CADENCE workflow without any UI dependencies."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        shap_top_k: int = 10,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[Any] = None,
    ) -> None:
        """Initialize the backend engine and its component services."""
        self.config = dict(CONFIG)
        if config:
            self.config.update(config)
        self.shap_top_k = int(shap_top_k)
        self.logger = logger or self._build_logger()
        self.progress_callback = progress_callback

    def run_pipeline(self, telemetry_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the end-to-end CADENCE workflow and return structured outputs."""
        try:
            self.logger.info("Starting CADENCE backend pipeline.")
            self._set_reproducible_seed()

            if telemetry_data is None:
                telemetry_generator = TelemetryGenerator(self.config)
                telemetry_data = self._execute_stage(
                    "telemetry_generation",
                    telemetry_generator.generate,
                )
            else:
                telemetry_data = self._execute_stage(
                    "telemetry_ingestion",
                    lambda: telemetry_data.copy(deep=True),
                )
            self.logger.info("Generated telemetry records: %s", len(telemetry_data))

            context_engine = AdaptiveBehaviourContextEngine(
                window_size=int(self.config["WINDOW_SIZE"])
            )
            behaviour_outputs = self._execute_stage(
                "behaviour_profiling",
                context_engine.fit_transform,
                telemetry_data,
            )
            prepared_telemetry = self._execute_stage(
                "telemetry_preparation",
                context_engine._prepare_dataframe,
                telemetry_data,
            )
            engineered_features = self._execute_stage(
                "feature_assembly",
                self._assemble_engineered_features,
                prepared_telemetry,
                behaviour_outputs,
            )
            self.logger.info(
                "Engineered flat features: %s rows, sequence windows: %s",
                len(engineered_features),
                behaviour_outputs["sequence_windows"].shape[0],
            )

            sequence_metadata = self._execute_stage(
                "sequence_metadata",
                self._build_sequence_metadata,
                prepared_telemetry,
            )
            sequence_windows = behaviour_outputs["sequence_windows"]
            if sequence_windows.shape[0] != len(sequence_metadata):
                raise ValueError("Sequence metadata and sequence windows are misaligned.")
            if sequence_windows.shape[0] < 3:
                raise ValueError("At least 3 sequences are required for GRU training.")

            gru_model = ConformalGRUAutoencoder(
                input_dim=int(sequence_windows.shape[2]),
                window_size=int(self.config["WINDOW_SIZE"]),
                target_fpr=float(self.config["TARGET_FPR"]),
                seed=int(self.config["SEED"]),
                epochs=5,
            )
            self._execute_stage("gru_training", gru_model.fit, sequence_windows)
            gru_predictions = self._execute_stage("gru_inference", gru_model.predict, sequence_windows)
            self.logger.info("GRU conformal threshold calibrated at %.6f", gru_model.threshold_)

            isolation_forest = BaselineIsolationForest(
                contamination=float(self.config["ATTACK_RATE"]),
                random_state=int(self.config["SEED"]),
            )
            flat_feature_matrix = behaviour_outputs["flat_features"]
            self._execute_stage("if_training", isolation_forest.fit, flat_feature_matrix)
            isolation_forest_predictions = self._execute_stage(
                "if_inference",
                isolation_forest.predict,
                flat_feature_matrix,
            )
            windowed_if_predictions = self._execute_stage(
                "if_window_aggregation",
                self._aggregate_isolation_forest_windows,
                isolation_forest_predictions,
                sequence_metadata,
            )

            risk_scores = {
                "gru_risk_score": gru_predictions["risk_score"],
                "isolation_forest_risk_score": windowed_if_predictions["risk_score"],
            }

            mitre_mapper = MITREMapper()
            incident_contexts = self._execute_stage(
                "incident_context_building",
                self._build_incident_contexts,
                engineered_features,
                sequence_metadata,
                gru_predictions,
                windowed_if_predictions,
            )
            mitre_results = self._execute_stage(
                "mitre_mapping",
                self._map_incidents_to_mitre,
                mitre_mapper,
                incident_contexts,
            )

            shap_results = self._execute_stage(
                "shap_explanations",
                self._generate_shap_results,
                isolation_forest,
                flat_feature_matrix,
                engineered_features,
                context_engine,
                isolation_forest_predictions,
            )

            evaluation_metrics: Dict[str, Dict[str, Any]] = {}
            if "sequence_label" in sequence_metadata and sequence_metadata["sequence_label"].notna().all():
                evaluation_engine = EvaluationEngine()
                evaluation_metrics = self._execute_stage(
                    "evaluation",
                    evaluation_engine.evaluate_models,
                    sequence_metadata["sequence_label"].to_numpy(dtype=int),
                    gru_predictions,
                    windowed_if_predictions,
                )
            self.logger.info("CADENCE backend pipeline completed successfully.")

            return {
                "telemetry_data": telemetry_data,
                "engineered_features": engineered_features,
                "gru_predictions": {
                    **gru_predictions,
                    "sequence_metadata": sequence_metadata.copy(),
                },
                "isolation_forest_predictions": {
                    "event_level": {
                        **isolation_forest_predictions,
                        "engineered_features": engineered_features.copy(),
                    },
                    "sequence_level": windowed_if_predictions,
                },
                "risk_scores": risk_scores,
                "adaptive_threshold": float(gru_model.threshold_ or 0.0),
                "mitre_results": mitre_results,
                "shap_results": shap_results,
                "evaluation_metrics": evaluation_metrics,
            }
        except Exception as exc:
            self.logger.exception("CADENCE backend pipeline failed.")
            raise RuntimeError(f"CADENCE backend pipeline failed: {exc}") from exc

    def _execute_stage(self, stage_name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute one pipeline stage with logging and wrapped exception handling."""
        self.logger.info("Running stage: %s", stage_name)
        if self.progress_callback is not None:
            self.progress_callback(stage_name, "running")
        try:
            result = func(*args, **kwargs)
            self.logger.info("Completed stage: %s", stage_name)
            if self.progress_callback is not None:
                self.progress_callback(stage_name, "completed")
            return result
        except Exception as exc:
            self.logger.exception("Pipeline stage failed: %s", stage_name)
            if self.progress_callback is not None:
                self.progress_callback(stage_name, "failed")
            raise RuntimeError(f"Stage '{stage_name}' failed: {exc}") from exc

    def _assemble_engineered_features(
        self,
        prepared_telemetry: pd.DataFrame,
        behaviour_outputs: Dict[str, Any],
    ) -> pd.DataFrame:
        """Combine engineered context features with aligned telemetry metadata."""
        engineered_features = behaviour_outputs["flat_feature_frame"].copy()
        engineered_features.insert(0, "entity_id", prepared_telemetry["entity_id"].to_numpy())
        engineered_features.insert(1, "entity_type", prepared_telemetry["entity_type"].to_numpy())
        engineered_features.insert(2, "timestamp", prepared_telemetry["timestamp"].to_numpy())
        engineered_features["source_ip"] = prepared_telemetry["source_ip"].to_numpy()
        engineered_features["geo_location"] = prepared_telemetry["geo_location"].to_numpy()
        engineered_features["resource_accessed"] = prepared_telemetry["resource_accessed"].to_numpy()
        engineered_features["device_fingerprint"] = prepared_telemetry["device_fingerprint"].to_numpy()
        engineered_features["command_sequence"] = prepared_telemetry["command_sequence"].to_numpy()
        if "label" in prepared_telemetry:
            engineered_features["label"] = prepared_telemetry["label"].to_numpy()
        return engineered_features

    def _map_incidents_to_mitre(
        self,
        mitre_mapper: MITREMapper,
        incident_contexts: Sequence[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Map incident contexts to MITRE ATT&CK records."""
        return pd.DataFrame(
            [
                {
                    **incident,
                    **mitre_mapper.map_anomaly(
                        anomaly_score=float(incident["anomaly_score"]),
                        context=incident,
                    ),
                }
                for incident in incident_contexts
            ]
        )

    def _build_sequence_metadata(self, prepared_telemetry: pd.DataFrame) -> pd.DataFrame:
        """Create sequence metadata and optional evaluation-only ground truth."""
        window_size = int(self.config["WINDOW_SIZE"])
        rows: List[Dict[str, Any]] = []
        for entity_id, entity_frame in prepared_telemetry.groupby("entity_id", sort=False):
            if len(entity_frame) < window_size:
                continue
            for start_idx in range(0, len(entity_frame) - window_size + 1):
                window_frame = entity_frame.iloc[start_idx : start_idx + window_size]
                sequence_label: Optional[int] = None
                if "label" in window_frame:
                    labels = window_frame["label"].astype(str).str.lower()
                    sequence_label = int((~labels.isin(["normal", "benign"])).any())
                rows.append(
                    {
                        "entity_id": entity_id,
                        "window_start": window_frame.iloc[0]["timestamp"],
                        "window_end": window_frame.iloc[-1]["timestamp"],
                        "sequence_label": sequence_label,
                        "event_indices": window_frame.index.to_list(),
                        "last_event_index": int(window_frame.index[-1]),
                    }
                )
        return pd.DataFrame(rows)

    def _aggregate_isolation_forest_windows(
        self,
        event_predictions: Dict[str, np.ndarray],
        sequence_metadata: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Aggregate flat-event Isolation Forest outputs to sequence granularity."""
        event_pred = np.asarray(event_predictions["predictions"], dtype=bool)
        event_scores = np.asarray(event_predictions["anomaly_scores"], dtype=np.float32)
        sequence_predictions: List[bool] = []
        sequence_scores: List[float] = []
        for event_indices in sequence_metadata["event_indices"]:
            indices = np.asarray(event_indices, dtype=int)
            sequence_predictions.append(bool(event_pred[indices].any()))
            sequence_scores.append(float(np.max(event_scores[indices])))
        scores = np.asarray(sequence_scores, dtype=np.float32)
        max_score = float(np.max(scores)) if scores.size else 0.0
        risk_score = (100.0 * scores / max_score).astype(np.float32) if max_score > 0 else np.zeros_like(scores)
        return {
            "predictions": np.asarray(sequence_predictions, dtype=bool),
            "anomaly_scores": scores,
            "risk_score": risk_score,
        }

    def _build_incident_contexts(
        self,
        engineered_features: pd.DataFrame,
        sequence_metadata: pd.DataFrame,
        gru_predictions: Dict[str, np.ndarray],
        if_predictions: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Build incident contexts from anomalous sequences for MITRE mapping."""
        incidents: List[Dict[str, Any]] = []
        gru_raw_scores = np.asarray(gru_predictions["reconstruction_error"], dtype=float)
        if_raw_scores = np.asarray(if_predictions["anomaly_scores"], dtype=float)

        def normalize_scores(scores: np.ndarray) -> np.ndarray:
            score_min = float(np.min(scores)) if scores.size else 0.0
            score_max = float(np.max(scores)) if scores.size else 0.0
            if score_max <= score_min:
                return np.zeros_like(scores, dtype=float)
            return 100.0 * (scores - score_min) / (score_max - score_min)

        normalized_anomaly_scores = (
            normalize_scores(gru_raw_scores) + normalize_scores(if_raw_scores)
        ) / 2.0
        for idx, metadata_row in sequence_metadata.iterrows():
            gru_flag = bool(gru_predictions["anomaly_prediction"][idx])
            if_flag = bool(if_predictions["predictions"][idx])
            if not (gru_flag or if_flag):
                continue
            event_indices = [int(value) for value in metadata_row["event_indices"]]
            window_rows = engineered_features.iloc[event_indices]
            event_row = engineered_features.iloc[int(metadata_row["last_event_index"])]
            incidents.append(
                {
                    "entity_id": metadata_row["entity_id"],
                    "window_start": metadata_row["window_start"],
                    "window_end": metadata_row["window_end"],
                    "resource_accessed": event_row["resource_accessed"],
                    "resource_category": event_row["resource_category"],
                    "auth_method": event_row["auth_method"],
                    "command_sequence": event_row["command_sequence"],
                    "classification_command_sequence": ";".join(window_rows["command_sequence"].astype(str)),
                    "classification_resources": ";".join(window_rows["resource_accessed"].astype(str)),
                    "classification_resource_categories": ";".join(window_rows["resource_category"].astype(str)),
                    "ip_subnet_change": bool(event_row["ip_subnet_change"]),
                    "new_device_indicator": bool(window_rows["new_device_indicator"].max()),
                    "time_since_previous_login": float(window_rows["time_since_previous_login"].min()),
                    "login_frequency": float(window_rows["login_frequency"].max()),
                    "gru_prediction": gru_flag,
                    "if_prediction": if_flag,
                    "gru_risk_score": float(gru_predictions["risk_score"][idx]),
                    "if_risk_score": float(if_predictions["risk_score"][idx]),
                    "anomaly_score": float(normalized_anomaly_scores[idx]),
                    "conformal_score": float(gru_predictions["risk_score"][idx]),
                    "isolation_forest_score": float(if_predictions["risk_score"][idx]),
                    "model_agreement": bool(gru_flag and if_flag),
                }
            )
        return incidents

    def _generate_shap_results(
        self,
        model: BaselineIsolationForest,
        feature_matrix: np.ndarray,
        engineered_features: pd.DataFrame,
        context_engine: AdaptiveBehaviourContextEngine,
        event_predictions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for the highest-scoring Isolation Forest incidents."""
        try:
            import shap  # type: ignore

            anomaly_scores = np.asarray(event_predictions["anomaly_scores"], dtype=np.float32)
            top_indices = np.argsort(anomaly_scores)[-self.shap_top_k :][::-1]
            if top_indices.size == 0:
                return {"available": True, "explanations": pd.DataFrame()}

            feature_names = self._flat_feature_names(context_engine)
            selected_features = feature_matrix[top_indices]
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(selected_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_frame = pd.DataFrame(shap_values, columns=feature_names)
            shap_frame.insert(0, "incident_index", top_indices.astype(int))
            shap_frame.insert(1, "entity_id", engineered_features.iloc[top_indices]["entity_id"].to_numpy())
            shap_frame.insert(2, "anomaly_score", anomaly_scores[top_indices])
            return {"available": True, "explanations": shap_frame}
        except Exception as exc:
            self.logger.warning("SHAP explanation generation unavailable: %s", exc)
            return {
                "available": False,
                "reason": str(exc),
                "explanations": pd.DataFrame(),
            }

    def _flat_feature_names(self, context_engine: AdaptiveBehaviourContextEngine) -> List[str]:
        """Return the fitted flat feature names used by the engineered feature matrix."""
        categorical_feature_names = list(
            context_engine.encoder.get_feature_names_out(context_engine._categorical_columns)
        )
        return list(context_engine._numeric_columns) + categorical_feature_names

    def _set_reproducible_seed(self) -> None:
        """Reset all supported random seeds before pipeline execution."""
        seed = int(self.config["SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_logger(self) -> logging.Logger:
        """Create a default logger for pipeline execution."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger


class BackendValidationSuite:
    """Run backend validation checks for every CADENCE component and pipeline stage."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the validation suite with reproducible configuration."""
        self.config = dict(CONFIG)
        if config:
            self.config.update(config)
        self.logger = logger or self._build_logger()
        self._artifact_cache: Optional[Dict[str, Any]] = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all independent backend tests and return a structured summary."""
        tests = {
            "TelemetryGenerator": self._run_test("TelemetryGenerator", self._test_telemetry_generator),
            "AdaptiveBehaviourContextEngine": self._run_test(
                "AdaptiveBehaviourContextEngine",
                self._test_adaptive_behaviour_context_engine,
            ),
            "ConformalGRUAutoencoder": self._run_test(
                "ConformalGRUAutoencoder",
                self._test_conformal_gru_autoencoder,
            ),
            "BaselineIsolationForest": self._run_test(
                "BaselineIsolationForest",
                self._test_baseline_isolation_forest,
            ),
            "MITREMapper": self._run_test("MITREMapper", self._test_mitre_mapper),
            "EvaluationEngine": self._run_test("EvaluationEngine", self._test_evaluation_engine),
            "BackendEngine": self._run_test("BackendEngine", self._test_backend_engine),
        }
        passed = all(result["passed"] for result in tests.values())
        return {
            "passed": passed,
            "tests": tests,
        }

    def _run_test(self, name: str, test_func: Any) -> Dict[str, Any]:
        """Execute one validation test with logging and error capture."""
        self.logger.info("Starting validation test: %s", name)
        try:
            result = test_func()
            self.logger.info("Validation test passed: %s", name)
            return {"passed": True, "details": result}
        except Exception as exc:
            self.logger.exception("Validation test failed: %s", name)
            return {"passed": False, "error": str(exc)}

    def _test_telemetry_generator(self) -> Dict[str, Any]:
        """Validate telemetry generation outputs and reproducibility."""
        self._set_reproducible_seed()
        generator_one = TelemetryGenerator(self.config)
        telemetry_one = generator_one.generate()
        self._set_reproducible_seed()
        generator_two = TelemetryGenerator(self.config)
        telemetry_two = generator_two.generate()

        required_columns = {
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
        }
        if not required_columns.issubset(telemetry_one.columns):
            raise AssertionError("Telemetry output schema is incomplete.")
        if len(telemetry_one) != int(self.config["N_LOGS"]):
            raise AssertionError("TelemetryGenerator did not return the configured number of logs.")
        if not telemetry_one.equals(telemetry_two):
            raise AssertionError("TelemetryGenerator output is not reproducible for the configured seed.")

        return {
            "rows": int(len(telemetry_one)),
            "attack_rows": int((telemetry_one["label"].astype(str).str.lower() != "normal").sum()),
        }

    def _test_adaptive_behaviour_context_engine(self) -> Dict[str, Any]:
        """Validate behavioural profiling and feature engineering outputs."""
        artifacts = self._get_base_artifacts()
        behaviour_outputs = artifacts["behaviour_outputs"]
        telemetry = artifacts["telemetry_data"]
        if behaviour_outputs["flat_features"].shape[0] != len(telemetry):
            raise AssertionError("Flat feature count does not match telemetry row count.")
        if behaviour_outputs["sequence_windows"].ndim != 3:
            raise AssertionError("Sequence windows must be a 3D tensor.")
        if behaviour_outputs["sequence_windows"].shape[1] != int(self.config["WINDOW_SIZE"]):
            raise AssertionError("Sequence windows do not match configured window size.")
        return {
            "flat_feature_shape": tuple(behaviour_outputs["flat_features"].shape),
            "sequence_window_shape": tuple(behaviour_outputs["sequence_windows"].shape),
        }

    def _test_conformal_gru_autoencoder(self) -> Dict[str, Any]:
        """Validate GRU training, calibration thresholding, and bounded risk scores."""
        artifacts = self._get_base_artifacts()
        sequence_windows = artifacts["sequence_windows"]
        sequence_labels = artifacts["sequence_labels"]
        benign_sequences = sequence_windows[sequence_labels == 0]
        if benign_sequences.shape[0] < 3:
            raise AssertionError("Insufficient benign sequences for conformal validation.")

        self._set_reproducible_seed()
        model = ConformalGRUAutoencoder(
            input_dim=int(sequence_windows.shape[2]),
            window_size=int(self.config["WINDOW_SIZE"]),
            target_fpr=float(self.config["TARGET_FPR"]),
            seed=int(self.config["SEED"]),
        )
        model.fit(benign_sequences)
        predictions = model.predict(sequence_windows)

        if model.threshold_ is None or model.calibration_errors_ is None:
            raise AssertionError("Conformal calibration threshold was not computed.")
        expected_threshold = self._higher_quantile(
            model.calibration_errors_,
            1.0 - float(self.config["TARGET_FPR"]),
        )
        if not np.isclose(model.threshold_, expected_threshold):
            raise AssertionError("Conformal threshold does not match the empirical quantile.")
        if not np.all((predictions["risk_score"] >= 0.0) & (predictions["risk_score"] <= 100.0)):
            raise AssertionError("GRU risk scores must remain within [0, 100].")

        return {
            "threshold": float(model.threshold_),
            "mean_reconstruction_error": float(np.mean(predictions["reconstruction_error"])),
        }

    def _test_baseline_isolation_forest(self) -> Dict[str, Any]:
        """Validate Isolation Forest training, scoring, and inference outputs."""
        artifacts = self._get_base_artifacts()
        flat_features = artifacts["behaviour_outputs"]["flat_features"]
        benign_event_mask = artifacts["prepared_telemetry"]["label"].astype(str).str.lower().isin(
            ["normal", "benign"]
        ).to_numpy()
        benign_features = flat_features[benign_event_mask]

        self._set_reproducible_seed()
        model = BaselineIsolationForest(
            contamination=float(self.config["ATTACK_RATE"]),
            random_state=int(self.config["SEED"]),
        )
        model.fit(benign_features)
        predictions = model.predict(flat_features)
        if predictions["predictions"].shape[0] != flat_features.shape[0]:
            raise AssertionError("Isolation Forest prediction length is invalid.")
        if not np.all(np.isfinite(predictions["anomaly_scores"])):
            raise AssertionError("Isolation Forest anomaly scores must be finite.")

        return {
            "anomalies_flagged": int(predictions["predictions"].sum()),
            "mean_anomaly_score": float(np.mean(predictions["anomaly_scores"])),
        }

    def _test_mitre_mapper(self) -> Dict[str, Any]:
        """Validate MITRE ATT&CK mappings for representative incident contexts."""
        mapper = MITREMapper()
        brute_force = mapper.map_anomaly(
            90.0,
            {
                "command_sequence": "login_fail;login_fail;password_reset_attempt",
                "auth_method": "password",
                "time_since_previous_login": 0.1,
                "login_frequency": 10.0,
            },
        )
        remote_services = mapper.map_anomaly(
            80.0,
            {
                "resource_category": "vpn",
                "command_sequence": "login;enumerate_hosts;remote_exec",
            },
        )
        valid_accounts = mapper.map_anomaly(
            75.0,
            {
                "ip_subnet_change": True,
                "new_device_indicator": True,
                "resource_category": "sso",
            },
        )
        if brute_force["technique_id"] != "T1110":
            raise AssertionError("Brute-force mapping failed.")
        if remote_services["technique_id"] != "T1021":
            raise AssertionError("Remote services mapping failed.")
        if valid_accounts["technique_id"] != "T1078":
            raise AssertionError("Valid accounts mapping failed.")
        return {
            "mapped_techniques": [
                brute_force["technique_id"],
                remote_services["technique_id"],
                valid_accounts["technique_id"],
            ]
        }

    def _test_evaluation_engine(self) -> Dict[str, Any]:
        """Validate metric calculations from actual predictions."""
        evaluator = EvaluationEngine()
        y_true = np.array([0, 0, 1, 1], dtype=int)
        gru_results = {
            "anomaly_prediction": np.array([False, True, True, False]),
            "risk_score": np.array([0.1, 0.6, 0.9, 0.2], dtype=np.float32),
        }
        if_results = {
            "predictions": np.array([False, False, True, True]),
            "anomaly_scores": np.array([0.05, 0.1, 0.8, 0.7], dtype=np.float32),
        }
        metrics = evaluator.evaluate_models(y_true, gru_results, if_results)
        gru_metrics = metrics["ConformalGRUAutoencoder"]
        if_metrics = metrics["BaselineIsolationForest"]
        if not np.isclose(gru_metrics["precision"], 0.5):
            raise AssertionError("GRU precision should be computed from actual predictions.")
        if not np.isclose(gru_metrics["recall"], 0.5):
            raise AssertionError("GRU recall should be computed from actual predictions.")
        if if_metrics["confusion_matrix"] != {"tn": 2, "fp": 0, "fn": 0, "tp": 2}:
            raise AssertionError("Isolation Forest confusion matrix is incorrect.")
        return metrics

    def _test_backend_engine(self) -> Dict[str, Any]:
        """Validate end-to-end pipeline execution, seed reproducibility, and training isolation."""
        fit_capture: Dict[str, Any] = {"gru_labels_arg": "unset", "if_fit_calls": 0}
        original_gru_fit = ConformalGRUAutoencoder.fit
        original_if_fit = BaselineIsolationForest.fit

        def wrapped_gru_fit(
            model_self: ConformalGRUAutoencoder,
            sequences: np.ndarray,
            labels: Optional[Sequence[Any]] = None,
        ) -> ConformalGRUAutoencoder:
            fit_capture["gru_labels_arg"] = labels
            return original_gru_fit(model_self, sequences, labels)

        def wrapped_if_fit(
            model_self: BaselineIsolationForest,
            features: np.ndarray,
        ) -> BaselineIsolationForest:
            fit_capture["if_fit_calls"] = int(fit_capture["if_fit_calls"]) + 1
            return original_if_fit(model_self, features)

        ConformalGRUAutoencoder.fit = wrapped_gru_fit
        BaselineIsolationForest.fit = wrapped_if_fit
        try:
            engine_one = BackendEngine(config=self.config, logger=self.logger)
            output_one = engine_one.run_pipeline()
            engine_two = BackendEngine(config=self.config, logger=self.logger)
            output_two = engine_two.run_pipeline()
        finally:
            ConformalGRUAutoencoder.fit = original_gru_fit
            BaselineIsolationForest.fit = original_if_fit

        if fit_capture["gru_labels_arg"] is not None:
            raise AssertionError("Labels were passed into GRU training.")
        if int(fit_capture["if_fit_calls"]) < 1:
            raise AssertionError("Isolation Forest training was not executed.")
        if not output_one["telemetry_data"].equals(output_two["telemetry_data"]):
            raise AssertionError("Pipeline telemetry output is not reproducible.")
        if not np.isclose(output_one["adaptive_threshold"], output_two["adaptive_threshold"]):
            raise AssertionError("Adaptive threshold is not reproducible.")
        if not np.all(
            (output_one["risk_scores"]["gru_risk_score"] >= 0.0)
            & (output_one["risk_scores"]["gru_risk_score"] <= 100.0)
        ):
            raise AssertionError("GRU pipeline risk scores must remain within [0, 100].")
        if not np.all(
            (output_one["risk_scores"]["isolation_forest_risk_score"] >= 0.0)
            & (output_one["risk_scores"]["isolation_forest_risk_score"] <= 100.0)
        ):
            raise AssertionError("Isolation Forest pipeline risk scores must remain within [0, 100].")
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
        if not required_keys.issubset(output_one.keys()):
            raise AssertionError("Pipeline output is missing required keys.")
        return {
            "adaptive_threshold": float(output_one["adaptive_threshold"]),
            "mitre_incidents": int(len(output_one["mitre_results"])),
            "gru_sequences": int(len(output_one["gru_predictions"]["risk_score"])),
        }

    def _get_base_artifacts(self) -> Dict[str, Any]:
        """Create and cache shared backend artifacts to avoid redundant computation."""
        if self._artifact_cache is not None:
            return self._artifact_cache

        self._set_reproducible_seed()
        telemetry_generator = TelemetryGenerator(self.config)
        telemetry_data = telemetry_generator.generate()
        context_engine = AdaptiveBehaviourContextEngine(window_size=int(self.config["WINDOW_SIZE"]))
        behaviour_outputs = context_engine.fit_transform(telemetry_data)
        prepared_telemetry = context_engine._prepare_dataframe(telemetry_data)
        backend_engine = BackendEngine(config=self.config, logger=self.logger)
        engineered_features = backend_engine._assemble_engineered_features(
            prepared_telemetry,
            behaviour_outputs,
        )
        sequence_metadata = backend_engine._build_sequence_metadata(prepared_telemetry)
        sequence_labels = sequence_metadata["sequence_label"].to_numpy(dtype=int)
        self._artifact_cache = {
            "telemetry_data": telemetry_data,
            "behaviour_outputs": behaviour_outputs,
            "prepared_telemetry": prepared_telemetry,
            "engineered_features": engineered_features,
            "sequence_metadata": sequence_metadata,
            "sequence_windows": behaviour_outputs["sequence_windows"],
            "sequence_labels": sequence_labels,
        }
        return self._artifact_cache

    def _higher_quantile(self, values: np.ndarray, quantile: float) -> float:
        """Compute a version-compatible empirical higher quantile."""
        try:
            return float(np.quantile(values, quantile, method="higher"))
        except TypeError:
            return float(np.quantile(values, quantile, interpolation="higher"))

    def _set_reproducible_seed(self) -> None:
        """Reset reproducible seeds across all supported libraries."""
        seed = int(self.config["SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_logger(self) -> logging.Logger:
        """Create a dedicated logger for backend validation."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger


def test_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the complete backend validation suite and print a concise summary."""
    suite = BackendValidationSuite(config=config)
    summary = suite.run_all_tests()
    print("\nCADENCE Backend Validation Summary")
    print("=" * 36)
    print(f"Overall status: {'PASSED' if summary['passed'] else 'FAILED'}")
    for test_name, result in summary["tests"].items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"- {test_name}: {status}")
        if result["passed"]:
            print(f"  details: {result['details']}")
        else:
            print(f"  error: {result['error']}")
    return summary


def main() -> None:
    """Generate logs and print a compact summary for local execution."""
    try:
        generator = TelemetryGenerator(CONFIG)
        telemetry_df = generator.generate()
        print(telemetry_df.head(10).to_string(index=False))
        print("\nLabel distribution:")
        print(telemetry_df["label"].value_counts().to_string())
        print(f"\nTotal logs: {len(telemetry_df)}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
