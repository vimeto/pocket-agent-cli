from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .benchmark_loader import iter_problem_records
from .run_finder import find_runs, filter_runs
from .run_info import RunInfo


@dataclass
class AttemptRecord:
    job_id: int
    model: str
    quantization: str
    mode: str
    session_id: str
    problem_id: int
    run_id: int
    attempt_index: Optional[int]
    success: bool
    duration_seconds: float
    energy_joules: Optional[float]
    avg_power_watts: Optional[float]
    max_power_watts: Optional[float]
    min_power_watts: Optional[float]
    energy_samples: Optional[int]
    thinking_tokens: Optional[int]
    actionable_tokens: Optional[int]
    thinking_ratio: Optional[float]
    tool_call_count: int
    submission_found: Optional[bool]
    submission_via_tool: Optional[bool]
    iteration_count: Optional[int]
    context_length_used: Optional[int]
    temperature: Optional[float]

    def to_row(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "quantization": self.quantization,
            "mode": self.mode,
            "session_id": self.session_id,
            "problem_id": self.problem_id,
            "run_id": self.run_id,
            "attempt_index": self.attempt_index,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "energy_joules": self.energy_joules,
            "avg_power_watts": self.avg_power_watts,
            "max_power_watts": self.max_power_watts,
            "min_power_watts": self.min_power_watts,
            "energy_samples": self.energy_samples,
            "thinking_tokens": self.thinking_tokens,
            "actionable_tokens": self.actionable_tokens,
            "thinking_ratio": self.thinking_ratio,
            "tool_call_count": self.tool_call_count,
            "submission_found": self.submission_found,
            "submission_via_tool": self.submission_via_tool,
            "iteration_count": self.iteration_count,
            "context_length_used": self.context_length_used,
            "temperature": self.temperature,
        }


def load_runs(data_root: Path, min_job_id: int | None = None, max_age_hours: float | None = None) -> List[RunInfo]:
    runs = find_runs(data_root)
    return filter_runs(runs, min_job_id=min_job_id, max_age_hours=max_age_hours)


def load_attempts(runs: Iterable[RunInfo]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    for run in runs:
        for payload in iter_problem_records(run):
            metrics = payload.get("metrics", {})
            energy_summary = metrics.get("energy_summary") or {}
            thinking_stats = metrics.get("thinking_stats") or {}

            tool_calls = payload.get("tool_calls") or []
            attempt_index = _safe_int_from_filename(payload.get("source_path"), payload.get("problem_id"), payload.get("run_id"))

            record = AttemptRecord(
                job_id=run.job_id,
                model=run.model,
                quantization=run.quantization,
                mode=run.mode,
                session_id=payload.get("session_id", ""),
                problem_id=int(payload.get("problem_id")),
                run_id=int(payload.get("run_id", 0)),
                attempt_index=attempt_index,
                success=bool(payload.get("success", False)),
                duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
                energy_joules=_safe_float(energy_summary.get("total_energy_joules")),
                avg_power_watts=_safe_float(energy_summary.get("avg_power_watts")),
                max_power_watts=_safe_float(energy_summary.get("max_power_watts")),
                min_power_watts=_safe_float(energy_summary.get("min_power_watts")),
                energy_samples=_safe_int(energy_summary.get("samples")),
                thinking_tokens=_safe_int(thinking_stats.get("thinking_tokens")),
                actionable_tokens=_safe_int(thinking_stats.get("regular_tokens")),
                thinking_ratio=_safe_float(thinking_stats.get("thinking_ratio")),
                tool_call_count=len(tool_calls),
                submission_found=_safe_bool(metrics.get("submission_found")),
                submission_via_tool=_safe_bool(metrics.get("submission_via_tool")),
                iteration_count=_safe_int(metrics.get("iteration_count")),
                context_length_used=_safe_int(payload.get("context_length_used")),
                temperature=_safe_float(payload.get("temperature")),
            )
            records.append(record.to_row())

    return records


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    return None


def _safe_int_from_filename(source_path: Optional[str], problem_id: int | None, run_id: int | None) -> Optional[int]:
    """Best-effort extraction of attempt index from the filename pattern."""
    if source_path and problem_id is not None:
        try:
            path = Path(source_path)
            # Example: bench_model_full_tool_175xxxxx/problem_13_run_4.json
            name = path.name
            if name.startswith("problem_") and "_run_" in name:
                return int(name.split("_run_")[-1].split(".")[0])
        except Exception:
            pass
    if run_id is not None:
        return int(run_id)
    return None
