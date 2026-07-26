from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    row: Optional[int] = None
    column: Optional[str] = None
    value: Optional[str] = None
    reason: str


class ImportResponse(BaseModel):
    import_id: str
    filename: str
    valid: bool
    summary: Dict[str, Any] = Field(default_factory=dict)
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)


class RunCreate(BaseModel):
    source: Literal["generated", "csv"] = "generated"
    import_id: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    source: Literal["generated", "csv"]
    filename: Optional[str] = None
    stage: str = "queued"
    progress: int = 0
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    page: int
    page_size: int
    total: int
    pages: int

