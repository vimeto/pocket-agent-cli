from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .benchmark_loader import SessionRecord


@dataclass
class AggregatedRow:
    model: str
    quantization: str
    mode: str
    job_ids: Tuple[int, ...]
    total_problems: int
    passed_problems: int
    pass_rate: Optional[float]
    pass_at_k: Dict[str, Optional[float]]
    total_duration_seconds: Optional[float]
    energy_joules: Optional[float]
    avg_power_watts: Optional[float]
    max_power_watts: Optional[float]
    min_power_watts: Optional[float]
    cpu_avg_percent: Optional[float]
    cpu_max_percent: Optional[float]
    memory_avg_percent: Optional[float]
    memory_max_percent: Optional[float]
    inter_token_avg_ms: Optional[float]
    inter_token_p95_ms: Optional[float]
    context_avg_tokens: Optional[float]
    context_max_tokens: Optional[float]
    warm_start_avg_s: Optional[float]
    warm_start_count: int

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.model,
            "quantization": self.quantization,
            "mode": self.mode,
            "job_ids": ",".join(str(j) for j in self.job_ids),
            "total_problems": self.total_problems,
            "passed_problems": self.passed_problems,
            "pass_rate": self.pass_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "energy_joules": self.energy_joules,
            "avg_power_watts": self.avg_power_watts,
            "max_power_watts": self.max_power_watts,
            "min_power_watts": self.min_power_watts,
            "cpu_avg_percent": self.cpu_avg_percent,
            "cpu_max_percent": self.cpu_max_percent,
            "memory_avg_percent": self.memory_avg_percent,
            "memory_max_percent": self.memory_max_percent,
            "inter_token_avg_ms": self.inter_token_avg_ms,
            "inter_token_p95_ms": self.inter_token_p95_ms,
            "context_avg_tokens": self.context_avg_tokens,
            "context_max_tokens": self.context_max_tokens,
            "warm_start_avg_s": self.warm_start_avg_s,
            "warm_start_count": self.warm_start_count,
        }
        for key, value in sorted(self.pass_at_k.items()):
            payload[key] = value
        return payload


def aggregate_sessions_by_config(records: Iterable[SessionRecord]) -> List[AggregatedRow]:
    buckets: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for record in records:
        key = (record.model, record.quantization, record.mode)
        bucket = buckets.setdefault(
            key,
            {
                "job_ids": set(),
                "total_problems": 0,
                "passed_problems": 0,
                "duration_sum": 0.0,
                "duration_count": 0,
                "system_duration": 0.0,
                "energy_sum": 0.0,
                "power_weighted": 0.0,
                "max_power": None,
                "min_power": None,
                "cpu_weighted": 0.0,
                "cpu_max": None,
                "cpu_weight": 0.0,
                "mem_weighted": 0.0,
                "mem_weight": 0.0,
                "mem_max": None,
                "pass_at_k": defaultdict(float),
                "inter_token_weighted": 0.0,
                "inter_token_weight": 0,
                "inter_token_p95_max": None,
                "context_weighted": 0.0,
                "context_weight": 0,
                "context_max": None,
                "warm_start_weighted": 0.0,
                "warm_start_count": 0,
            },
        )

        bucket["job_ids"].add(record.job_id)
        bucket["total_problems"] += record.total_problems
        bucket["passed_problems"] += record.passed_problems

        if record.total_duration_seconds:
            bucket["duration_sum"] += record.total_duration_seconds
            bucket["duration_count"] += 1

        if record.energy_joules:
            bucket["energy_sum"] += record.energy_joules

        duration = record.system_duration_seconds or record.total_duration_seconds or 0.0
        if record.avg_power_watts and duration:
            bucket["power_weighted"] += record.avg_power_watts * duration
            bucket["system_duration"] += duration
        elif duration:
            bucket["system_duration"] += duration

        bucket["max_power"] = _safe_max(bucket.get("max_power"), record.max_power_watts)
        bucket["min_power"] = _safe_min(bucket.get("min_power"), record.min_power_watts)

        if record.cpu_avg_percent and duration:
            bucket["cpu_weighted"] += record.cpu_avg_percent * duration
            bucket["cpu_weight"] += duration
        bucket["cpu_max"] = _safe_max(bucket.get("cpu_max"), record.cpu_max_percent)

        if record.memory_avg_percent and duration:
            bucket["mem_weighted"] += record.memory_avg_percent * duration
            bucket["mem_weight"] += duration
        bucket["mem_max"] = _safe_max(bucket.get("mem_max"), record.memory_max_percent)

        if record.total_problems:
            for key_k, value in record.pass_at_k.items():
                if value is None:
                    continue
                bucket["pass_at_k"][key_k] += value * record.total_problems

        if record.inter_token_avg_ms and record.total_problems:
            bucket["inter_token_weighted"] += record.inter_token_avg_ms * record.total_problems
            bucket["inter_token_weight"] += record.total_problems

        bucket["inter_token_p95_max"] = _safe_max(bucket.get("inter_token_p95_max"), record.inter_token_p95_ms)

        if record.context_avg_tokens and record.total_problems:
            bucket["context_weighted"] += record.context_avg_tokens * record.total_problems
            bucket["context_weight"] += record.total_problems
        bucket["context_max"] = _safe_max(bucket.get("context_max"), record.context_max_tokens)

        if record.warm_start_avg_s and record.warm_start_count:
            bucket["warm_start_weighted"] += record.warm_start_avg_s * record.warm_start_count
            bucket["warm_start_count"] += record.warm_start_count

    aggregated: List[AggregatedRow] = []
    for (model, quant, mode), bucket in sorted(buckets.items()):
        total_problems = bucket["total_problems"]
        passed_problems = bucket["passed_problems"]
        pass_rate = passed_problems / total_problems if total_problems else None

        pass_at_k = {}
        for key_k, weighted_sum in bucket["pass_at_k"].items():
            pass_at_k[key_k] = weighted_sum / total_problems if total_problems else None

        avg_duration = (
            bucket["duration_sum"] / bucket["duration_count"] if bucket["duration_count"] else None
        )
        avg_power = (
            bucket["power_weighted"] / bucket["system_duration"] if bucket["system_duration"] else None
        )
        cpu_avg = (
            bucket["cpu_weighted"] / bucket["cpu_weight"] if bucket["cpu_weight"] else None
        )
        mem_avg = (
            bucket["mem_weighted"] / bucket["mem_weight"] if bucket["mem_weight"] else None
        )
        inter_token_avg = (
            bucket["inter_token_weighted"] / bucket["inter_token_weight"] if bucket["inter_token_weight"] else None
        )
        context_avg = (
            bucket["context_weighted"] / bucket["context_weight"] if bucket["context_weight"] else None
        )
        warm_start_avg = (
            bucket["warm_start_weighted"] / bucket["warm_start_count"] if bucket["warm_start_count"] else None
        )

        aggregated.append(
            AggregatedRow(
                model=model,
                quantization=quant,
                mode=mode,
                job_ids=tuple(sorted(bucket["job_ids"])),
                total_problems=total_problems,
                passed_problems=passed_problems,
                pass_rate=pass_rate,
                pass_at_k=pass_at_k,
                total_duration_seconds=avg_duration,
                energy_joules=bucket["energy_sum"] if bucket["energy_sum"] else None,
                avg_power_watts=avg_power,
                max_power_watts=bucket["max_power"],
                min_power_watts=bucket["min_power"],
                cpu_avg_percent=cpu_avg,
                cpu_max_percent=bucket["cpu_max"],
                memory_avg_percent=mem_avg,
                memory_max_percent=bucket["mem_max"],
                inter_token_avg_ms=inter_token_avg,
                inter_token_p95_ms=bucket["inter_token_p95_max"],
                context_avg_tokens=context_avg,
                context_max_tokens=bucket["context_max"],
                warm_start_avg_s=warm_start_avg,
                warm_start_count=bucket["warm_start_count"],
            )
        )

    return aggregated


def summarize_by_job(
    records: Iterable[SessionRecord],
    telemetry_by_job: Dict[int, Dict[str, Optional[float]]],
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[SessionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.job_id].append(record)

    job_rows: List[Dict[str, object]] = []
    for job_id, job_records in sorted(grouped.items()):
        total_problems = sum(r.total_problems for r in job_records)
        passed_problems = sum(r.passed_problems for r in job_records)
        duration = sum((r.total_duration_seconds or 0.0) for r in job_records)
        energy = sum((r.energy_joules or 0.0) for r in job_records)
        pass_rate = passed_problems / total_problems if total_problems else None
        modes = sorted({r.mode for r in job_records})
        models = sorted({r.model for r in job_records})
        quants = sorted({r.quantization for r in job_records})

        row: Dict[str, object] = {
            "job_id": job_id,
            "models": ",".join(models),
            "quantizations": ",".join(quants),
            "modes": ",".join(modes),
            "total_problems": total_problems,
            "passed_problems": passed_problems,
            "pass_rate": pass_rate,
            "total_duration_seconds": duration if duration else None,
            "energy_joules": energy if energy else None,
        }
        row.update(telemetry_by_job.get(job_id, {}))
        job_rows.append(row)

    return job_rows


def _safe_max(current: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _safe_min(current: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)
