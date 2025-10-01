from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ConfigKey:
    model: str
    quantization: str
    mode: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.model, self.quantization, self.mode)


@dataclass
class ConfigSummary:
    config: ConfigKey
    total_attempts: int
    successes: int
    tasks_total: int
    tasks_solved: int
    pass_at_k: Dict[int, float]
    pass_rate: float
    pass_rate_ci: Tuple[float, float]
    avg_duration: Optional[float]
    median_duration: Optional[float]
    p95_duration: Optional[float]
    avg_energy: Optional[float]
    total_energy: Optional[float]
    energy_per_success: Optional[float]
    avg_power: Optional[float]
    avg_tool_calls: Optional[float]
    avg_thinking_tokens: Optional[float]
    avg_actionable_tokens: Optional[float]
    thinking_ratio_mean: Optional[float]
    avg_total_tokens: Optional[float]
    tokens_per_second_mean: Optional[float]
    energy_per_token_mean: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        data = {
            "model": self.config.model,
            "quantization": self.config.quantization,
            "mode": self.config.mode,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "tasks_total": self.tasks_total,
            "tasks_solved": self.tasks_solved,
            "pass_rate": self.pass_rate,
            "pass_rate_ci_low": self.pass_rate_ci[0],
            "pass_rate_ci_high": self.pass_rate_ci[1],
            "avg_duration_seconds": self.avg_duration,
            "median_duration_seconds": self.median_duration,
            "p95_duration_seconds": self.p95_duration,
            "avg_energy_joules": self.avg_energy,
            "total_energy_joules": self.total_energy,
            "energy_per_success_joules": self.energy_per_success,
            "avg_power_watts": self.avg_power,
            "avg_tool_calls": self.avg_tool_calls,
            "avg_thinking_tokens": self.avg_thinking_tokens,
            "avg_actionable_tokens": self.avg_actionable_tokens,
            "thinking_ratio_mean": self.thinking_ratio_mean,
        }
        for k, v in self.pass_at_k.items():
            data[f"pass_at_{k}"] = v
        data["avg_total_tokens"] = self.avg_total_tokens
        data["tokens_per_second_mean"] = self.tokens_per_second_mean
        data["energy_per_token_mean"] = self.energy_per_token_mean
        return data


def summarise_attempts(
    attempts: Iterable[Dict[str, object]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> List[ConfigSummary]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in attempts:
        key = (row["model"], row["quantization"], row["mode"])
        grouped[key].append(row)

    summaries: List[ConfigSummary] = []
    for (model, quant, mode), rows in grouped.items():
        total_attempts = len(rows)
        successes = sum(1 for r in rows if r.get("success"))

        # Distinct tasks solved
        problems: Dict[int, List[bool]] = defaultdict(list)
        durations: List[float] = []
        energies: List[float] = []
        powers: List[float] = []
        tool_calls: List[int] = []
        thinking_tokens: List[int] = []
        actionable_tokens: List[int] = []
        thinking_ratios: List[float] = []
        total_tokens: List[int] = []
        tokens_per_second: List[float] = []
        energy_per_token: List[float] = []

        for r in rows:
            problem_id = int(r.get("problem_id"))
            problems[problem_id].append(bool(r.get("success")))
            duration = _safe_float(r.get("duration_seconds"))
            if duration is not None:
                durations.append(duration)
            energy = _safe_float(r.get("energy_joules"))
            if energy is not None:
                energies.append(energy)
            power = _safe_float(r.get("avg_power_watts"))
            if power is not None:
                powers.append(power)
            tc = r.get("tool_call_count")
            if isinstance(tc, (int, float)):
                tool_calls.append(float(tc))
            tt = _safe_float(r.get("thinking_tokens"))
            if tt is not None:
                thinking_tokens.append(tt)
            at = _safe_float(r.get("actionable_tokens"))
            if at is not None:
                actionable_tokens.append(at)
            tr = _safe_float(r.get("thinking_ratio"))
            if tr is not None:
                thinking_ratios.append(tr)
            if tt is not None and at is not None:
                total = tt + at
                total_tokens.append(total)
                if duration and duration > 0:
                    tokens_per_second.append(total / duration)
                energy = _safe_float(r.get("energy_joules"))
                if energy is not None and total > 0:
                    energy_per_token.append(energy / total)

        tasks_total = len(problems)
        tasks_solved = sum(1 for successes_list in problems.values() if any(successes_list))

        pass_at_k = {
            k: _average_pass_at_k(successes_list=problems, k=k)
            for k in ks
        }

        pass_rate = successes / total_attempts if total_attempts else 0.0
        ci = _wilson_interval(successes, total_attempts) if total_attempts else (0.0, 0.0)

        avg_duration = _mean(durations)
        median_duration = _median(durations)
        p95_duration = _percentile(durations, 95)
        avg_energy = _mean(energies)
        total_energy = sum(energies) if energies else None
        energy_per_success = (total_energy / tasks_solved) if total_energy is not None and tasks_solved else None
        avg_power = _mean(powers)
        avg_tool_calls = _mean(tool_calls)
        avg_thinking = _mean(thinking_tokens)
        avg_actionable = _mean(actionable_tokens)
        thinking_ratio_mean = _mean(thinking_ratios)
        avg_total_tokens = _mean(total_tokens)
        tokens_per_second_mean = _mean(tokens_per_second)
        energy_per_token_mean = _mean(energy_per_token)

        summaries.append(
            ConfigSummary(
                config=ConfigKey(model, quant, mode),
                total_attempts=total_attempts,
                successes=successes,
                tasks_total=tasks_total,
                tasks_solved=tasks_solved,
                pass_at_k=pass_at_k,
                pass_rate=pass_rate,
                pass_rate_ci=ci,
                avg_duration=avg_duration,
                median_duration=median_duration,
                p95_duration=p95_duration,
                avg_energy=avg_energy,
                total_energy=total_energy,
                energy_per_success=energy_per_success,
                avg_power=avg_power,
                avg_tool_calls=avg_tool_calls,
                avg_thinking_tokens=avg_thinking,
                avg_actionable_tokens=avg_actionable,
                thinking_ratio_mean=thinking_ratio_mean,
                avg_total_tokens=avg_total_tokens,
                tokens_per_second_mean=tokens_per_second_mean,
                energy_per_token_mean=energy_per_token_mean,
            )
        )

    return summaries


def _average_pass_at_k(successes_list: Dict[int, List[bool]], k: int) -> float:
    values: List[float] = []
    for attempts in successes_list.values():
        n = len(attempts)
        c = sum(1 for a in attempts if a)
        values.append(_pass_at_k_single(n, c, k))
    return _mean(values)


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    if n == 0 or k == 0:
        return 0.0
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    comb = math.comb
    numerator = comb(n - c, k)
    denominator = comb(n, k)
    return 1.0 - numerator / denominator


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    from math import sqrt

    z = 1.959963984540054  # 95% confidence
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
