from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .run_info import RunInfo


@dataclass(frozen=True)
class SessionRecord:
    job_id: int
    model: str
    quantization: str
    mode: str
    session_id: str
    run_path: Path
    job_timestamp: Optional[str]
    total_problems: int
    passed_problems: int
    pass_rate: Optional[float]
    pass_at_k: Dict[str, float]
    total_duration_seconds: Optional[float]
    system_duration_seconds: Optional[float]
    energy_joules: Optional[float]
    avg_power_watts: Optional[float]
    max_power_watts: Optional[float]
    min_power_watts: Optional[float]
    energy_samples: Optional[int]
    cpu_avg_percent: Optional[float]
    cpu_max_percent: Optional[float]
    memory_avg_percent: Optional[float]
    memory_max_percent: Optional[float]
    inter_token_avg_ms: Optional[float]
    inter_token_p95_ms: Optional[float]
    inter_token_p99_ms: Optional[float]
    context_avg_tokens: Optional[float]
    context_max_tokens: Optional[int]
    warm_start_avg_s: Optional[float]
    warm_start_count: Optional[int]
    num_samples: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "quantization": self.quantization,
            "mode": self.mode,
            "session_id": self.session_id,
            "run_path": str(self.run_path),
            "job_timestamp": self.job_timestamp,
            "total_problems": self.total_problems,
            "passed_problems": self.passed_problems,
            "pass_rate": self.pass_rate,
            **{k: self.pass_at_k.get(k) for k in sorted(self.pass_at_k)},
            "total_duration_seconds": self.total_duration_seconds,
            "system_duration_seconds": self.system_duration_seconds,
            "energy_joules": self.energy_joules,
            "avg_power_watts": self.avg_power_watts,
            "max_power_watts": self.max_power_watts,
            "min_power_watts": self.min_power_watts,
            "energy_samples": self.energy_samples,
            "cpu_avg_percent": self.cpu_avg_percent,
            "cpu_max_percent": self.cpu_max_percent,
            "memory_avg_percent": self.memory_avg_percent,
            "memory_max_percent": self.memory_max_percent,
            "inter_token_avg_ms": self.inter_token_avg_ms,
            "inter_token_p95_ms": self.inter_token_p95_ms,
            "inter_token_p99_ms": self.inter_token_p99_ms,
            "context_avg_tokens": self.context_avg_tokens,
            "context_max_tokens": self.context_max_tokens,
            "warm_start_avg_s": self.warm_start_avg_s,
            "warm_start_count": self.warm_start_count,
            "num_samples": self.num_samples,
        }


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as fh:
        return json.load(fh)


def extract_session_records(run: RunInfo) -> List[SessionRecord]:
    summary = _load_json(run.summary_path)
    config = summary.get("config", {})
    sessions = summary.get("sessions", [])

    records: List[SessionRecord] = []
    for session in sessions:
        aggregate = session.get("aggregate_stats", {})
        system_metrics = aggregate.get("system_metrics", {}) or {}
        inter_token = aggregate.get("inter_token_latency", {}) or {}
        warm_start = aggregate.get("warm_start", {}) or {}
        context_length = aggregate.get("context_length", {}) or {}
        pass_at_k = aggregate.get("pass_at_k", {}) or {}

        records.append(
            SessionRecord(
                job_id=run.job_id,
                model=run.model,
                quantization=run.quantization,
                mode=run.mode,
                session_id=session.get("session_id", ""),
                run_path=run.path,
                job_timestamp=summary.get("timestamp"),
                total_problems=int(aggregate.get("total_problems", 0) or 0),
                passed_problems=int(aggregate.get("passed_problems", 0) or 0),
                pass_rate=_safe_float(aggregate.get("pass_rate")),
                pass_at_k={k: _safe_float(v) for k, v in pass_at_k.items()},
                total_duration_seconds=_safe_float(aggregate.get("total_duration_seconds")),
                system_duration_seconds=_safe_float(system_metrics.get("duration_seconds")),
                energy_joules=_safe_float(system_metrics.get("total_energy_joules")),
                avg_power_watts=_safe_float(system_metrics.get("avg_power_watts")),
                max_power_watts=_safe_float(system_metrics.get("max_power_watts")),
                min_power_watts=_safe_float(system_metrics.get("min_power_watts")),
                energy_samples=_safe_int(system_metrics.get("samples")),
                cpu_avg_percent=_safe_float(system_metrics.get("cpu_avg_percent")),
                cpu_max_percent=_safe_float(system_metrics.get("cpu_max_percent")),
                memory_avg_percent=_safe_float(system_metrics.get("memory_avg_percent")),
                memory_max_percent=_safe_float(system_metrics.get("memory_max_percent")),
                inter_token_avg_ms=_safe_float(inter_token.get("avg_ms")),
                inter_token_p95_ms=_safe_float(inter_token.get("p95_ms")),
                inter_token_p99_ms=_safe_float(inter_token.get("p99_ms")),
                context_avg_tokens=_safe_float(context_length.get("avg")),
                context_max_tokens=_safe_int(context_length.get("max")),
                warm_start_avg_s=_safe_float(warm_start.get("avg_duration_s")),
                warm_start_count=_safe_int(warm_start.get("count")),
                num_samples=_safe_int(config.get("num_samples")),
            )
        )

    return records


def iter_problem_records(run: RunInfo) -> Iterator[Dict[str, object]]:
    if not run.session_root.exists():
        return iter(())

    def _iter() -> Iterator[Dict[str, object]]:
        for session_dir in run.session_root.iterdir():
            if not session_dir.is_dir():
                continue
            for path in session_dir.glob("problem_*_run_*.json"):
                payload = _load_json(path)
                payload["job_id"] = run.job_id
                payload["model"] = run.model
                payload["quantization"] = run.quantization
                payload["mode"] = run.mode
                payload["session_id"] = session_dir.name
                payload["run_path"] = str(run.path)
                payload["source_path"] = str(path)
                yield payload

    return _iter()


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
