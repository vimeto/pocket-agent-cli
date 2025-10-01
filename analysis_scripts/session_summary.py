from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .benchmark_loader import SessionRecord


def summarise_sessions(records: Iterable[SessionRecord]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Aggregate session-level statistics weighted by number of problems."""
    buckets: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: Dict[Tuple[str, str, str], float] = defaultdict(float)

    for rec in records:
        key = (rec.model, rec.quantization, rec.mode)
        weight = rec.total_problems or 0
        if weight == 0:
            continue
        weights[key] += weight

        # Weighted aggregates
        if rec.total_duration_seconds is not None:
            buckets[key]["avg_duration_seconds"] += rec.total_duration_seconds
        if rec.inter_token_avg_ms is not None:
            buckets[key]["inter_token_avg_ms_sum"] += rec.inter_token_avg_ms * weight
        if rec.inter_token_p95_ms is not None:
            buckets[key]["inter_token_p95_ms_sum"] += rec.inter_token_p95_ms * weight
        if rec.context_avg_tokens is not None:
            buckets[key]["context_avg_tokens_sum"] += rec.context_avg_tokens * weight
        if rec.context_max_tokens is not None:
            buckets[key]["context_max_tokens_sum"] += rec.context_max_tokens * weight
        if rec.warm_start_avg_s is not None:
            buckets[key]["warm_start_avg_s_sum"] += rec.warm_start_avg_s * weight
        if rec.energy_joules is not None:
            buckets[key]["energy_joules_sum"] += rec.energy_joules
        if rec.avg_power_watts is not None:
            buckets[key]["avg_power_watts_sum"] += rec.avg_power_watts * weight
        if rec.max_power_watts is not None:
            buckets[key]["max_power_watts_sum"] += rec.max_power_watts * weight
        if rec.min_power_watts is not None:
            buckets[key]["min_power_watts_sum"] += rec.min_power_watts * weight
        if rec.total_problems is not None:
            buckets[key]["total_problems_sum"] += rec.total_problems
        if rec.passed_problems is not None:
            buckets[key]["passed_problems_sum"] += rec.passed_problems
        if rec.total_duration_seconds is not None:
            buckets[key]["total_duration_seconds_sum"] += rec.total_duration_seconds

    summary: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for key, values in buckets.items():
        weight = weights.get(key, 0.0)
        if weight == 0:
            continue
        result: Dict[str, float] = {}
        # Weighted means
        if "inter_token_avg_ms_sum" in values:
            result["inter_token_avg_ms"] = values["inter_token_avg_ms_sum"] / weight
        if "inter_token_p95_ms_sum" in values:
            result["inter_token_p95_ms"] = values["inter_token_p95_ms_sum"] / weight
        if "context_avg_tokens_sum" in values:
            result["context_avg_tokens"] = values["context_avg_tokens_sum"] / weight
        if "context_max_tokens_sum" in values:
            result["context_max_tokens"] = values["context_max_tokens_sum"] / weight
        if "warm_start_avg_s_sum" in values:
            result["warm_start_avg_s"] = values["warm_start_avg_s_sum"] / weight
        if "avg_power_watts_sum" in values:
            result["avg_power_watts_weighted"] = values["avg_power_watts_sum"] / weight
        if "max_power_watts_sum" in values:
            result["max_power_watts_weighted"] = values["max_power_watts_sum"] / weight
        if "min_power_watts_sum" in values:
            result["min_power_watts_weighted"] = values["min_power_watts_sum"] / weight
        if "total_problems_sum" in values:
            result["total_problems"] = values["total_problems_sum"]
        if "passed_problems_sum" in values:
            result["passed_problems"] = values["passed_problems_sum"]
        if "energy_joules_sum" in values:
            result["total_energy_joules"] = values["energy_joules_sum"]
        if "total_duration_seconds_sum" in values:
            result["total_duration_seconds"] = values["total_duration_seconds_sum"]

        summary[key] = result

    return summary
