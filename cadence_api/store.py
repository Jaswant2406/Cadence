from __future__ import annotations

import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from app import BackendEngine, CONFIG

from .views import build_snapshot


PIPELINE_STAGES = [
    "telemetry_generation",
    "telemetry_ingestion",
    "behaviour_profiling",
    "telemetry_preparation",
    "feature_assembly",
    "sequence_metadata",
    "gru_training",
    "gru_inference",
    "if_training",
    "if_inference",
    "if_window_aggregation",
    "incident_context_building",
    "mitre_mapping",
    "shap_explanations",
    "evaluation",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImportArtifact:
    import_id: str
    filename: str
    dataframe: Optional[pd.DataFrame]
    response: Dict[str, Any]


@dataclass
class RunArtifact:
    run_id: str
    source: str
    filename: Optional[str]
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    error: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None

    def public(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "source": self.source,
            "filename": self.filename,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class ApplicationStore:
    def __init__(self) -> None:
        self.imports: Dict[str, ImportArtifact] = {}
        self.runs: Dict[str, RunArtifact] = {}
        self.latest_run_id: Optional[str] = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cadence-run")

    def add_import(
        self,
        filename: str,
        dataframe: Optional[pd.DataFrame],
        response: Dict[str, Any],
    ) -> ImportArtifact:
        import_id = uuid.uuid4().hex[:12]
        artifact = ImportArtifact(import_id, filename, dataframe, response)
        with self._lock:
            self.imports[import_id] = artifact
        return artifact

    def start_run(self, source: str, import_id: Optional[str] = None) -> RunArtifact:
        dataframe: Optional[pd.DataFrame] = None
        filename: Optional[str] = None
        if source == "csv":
            if not import_id or import_id not in self.imports:
                raise KeyError("Validated CSV import was not found.")
            imported = self.imports[import_id]
            if imported.dataframe is None:
                raise ValueError("CSV import contains validation errors.")
            dataframe = imported.dataframe.copy(deep=True)
            filename = imported.filename

        run_id = uuid.uuid4().hex[:12]
        run = RunArtifact(run_id=run_id, source=source, filename=filename)
        with self._lock:
            self.runs[run_id] = run
            self.latest_run_id = run_id
        self._executor.submit(self._execute, run_id, dataframe)
        return run

    def get_run(self, run_id: str) -> RunArtifact:
        with self._lock:
            if run_id not in self.runs:
                raise KeyError("Run was not found.")
            return self.runs[run_id]

    def latest(self) -> Optional[RunArtifact]:
        with self._lock:
            return self.runs.get(self.latest_run_id) if self.latest_run_id else None

    def _execute(self, run_id: str, dataframe: Optional[pd.DataFrame]) -> None:
        run = self.get_run(run_id)
        with self._lock:
            run.status = "running"
            run.stage = "starting"
            run.progress = 2

        completed_stages: set[str] = set()

        def progress(stage: str, status: str) -> None:
            with self._lock:
                run.stage = stage
                if status == "completed":
                    completed_stages.add(stage)
                run.progress = min(
                    95,
                    max(3, int(math.ceil(len(completed_stages) / 14 * 92))),
                )

        try:
            engine = BackendEngine(config=CONFIG, progress_callback=progress)
            result = engine.run_pipeline(telemetry_data=dataframe)
            snapshot = build_snapshot(result, run_id=run_id)
            with self._lock:
                run.snapshot = snapshot
                run.status = "completed"
                run.stage = "completed"
                run.progress = 100
                run.completed_at = now_iso()
        except Exception as exc:
            with self._lock:
                run.status = "failed"
                run.stage = "failed"
                run.error = str(exc)
                run.completed_at = now_iso()


store = ApplicationStore()

