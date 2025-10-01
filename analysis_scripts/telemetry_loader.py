from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

Numeric = Optional[float]


def load_gpu_metrics(path: Path) -> Dict[str, Numeric]:
    records: List[Dict[str, Numeric]] = []
    if not path.exists():
        return {}

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {key.strip(): value for key, value in row.items()}
            try:
                records.append(
                    {
                        "gpu_util": _parse_numeric(clean.get("utilization.gpu [%]")),
                        "mem_util": _parse_numeric(clean.get("utilization.memory [%]")),
                        "mem_used_mib": _parse_numeric(clean.get("memory.used [MiB]")),
                        "mem_free_mib": _parse_numeric(clean.get("memory.free [MiB]")),
                        "power_w": _parse_numeric(clean.get("power.draw [W]")),
                        "temperature": _parse_numeric(clean.get("temperature.gpu")),
                    }
                )
            except ValueError:
                continue

    if not records:
        return {}

    def _avg(key: str) -> Numeric:
        values = [row[key] for row in records if row[key] is not None]
        return mean(values) if values else None

    def _max(key: str) -> Numeric:
        values = [row[key] for row in records if row[key] is not None]
        return max(values) if values else None

    return {
        "gpu_util_avg": _avg("gpu_util"),
        "gpu_util_max": _max("gpu_util"),
        "gpu_mem_util_avg": _avg("mem_util"),
        "gpu_mem_util_max": _max("mem_util"),
        "gpu_mem_used_avg_mib": _avg("mem_used_mib"),
        "gpu_mem_used_max_mib": _max("mem_used_mib"),
        "gpu_power_avg_w": _avg("power_w"),
        "gpu_power_max_w": _max("power_w"),
        "gpu_temperature_avg_c": _avg("temperature"),
        "gpu_samples": len(records),
    }


def load_cpu_metrics(path: Path) -> Dict[str, Numeric]:
    if not path.exists():
        return {}

    cpu_values: List[float] = []
    mem_values: List[float] = []

    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                continue
            cpu_percent = _parse_cpu_percent(row[1:-1])
            mem_percent = _parse_numeric(row[-1]) if row[-1] else None
            if cpu_percent is not None:
                cpu_values.append(cpu_percent)
            if mem_percent is not None:
                mem_values.append(mem_percent)

    if not cpu_values and not mem_values:
        return {}

    payload: Dict[str, Numeric] = {}
    if cpu_values:
        payload["cpu_util_avg_percent"] = mean(cpu_values)
        payload["cpu_util_max_percent"] = max(cpu_values)
        payload["cpu_samples"] = len(cpu_values)
    if mem_values:
        payload["cpu_mem_avg_percent"] = mean(mem_values)
        payload["cpu_mem_max_percent"] = max(mem_values)
    return payload


def load_job_telemetry(log_root: Path, job_id: int) -> Dict[str, Numeric]:
    telemetry: Dict[str, Numeric] = {}
    gpu_metrics = load_gpu_metrics(log_root / f"gpu_monitor_{job_id}.csv")
    cpu_metrics = load_cpu_metrics(log_root / f"cpu_monitor_{job_id}.csv")
    telemetry.update(gpu_metrics)
    telemetry.update(cpu_metrics)
    return telemetry


def _parse_numeric(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    for suffix in ["%", "MiB", "W"]:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    value = value.replace(" %", "").replace(" MiB", "").replace(" W", "")
    value = value.replace("%", "").replace("MiB", "").replace("W", "")
    value = value.strip()
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_cpu_percent(parts: Iterable[str]) -> Optional[float]:
    tokens = [token for token in parts if token is not None]
    if not tokens:
        return None
    joined = ".".join(token.strip().replace("%", "") for token in tokens)
    joined = joined.strip()
    if not joined:
        return None
    try:
        return float(joined)
    except ValueError:
        return None
