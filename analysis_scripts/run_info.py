from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunInfo:
    """Describes a single benchmark output directory."""

    path: Path
    name: str
    job_id: int
    timestamp: datetime
    model: str
    quantization: str
    mode: str

    @property
    def summary_path(self) -> Path:
        return self.path / "benchmark_summary.json"

    @property
    def session_root(self) -> Path:
        return self.path / self.model / self.mode / "runs"
