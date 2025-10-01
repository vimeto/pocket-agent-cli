from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

from .run_info import RunInfo

def _parse_run_dir(path: Path) -> RunInfo | None:
    if not path.name.startswith("bench_"):
        return None

    summary_path = path / "benchmark_summary.json"
    if not summary_path.exists():
        return None

    try:
        base, job_part = path.name.rsplit("_job", 1)
        job_id = int(job_part)
    except ValueError:
        return None

    data = json.loads(summary_path.read_text())
    config = data.get("config", {})
    model = config.get("model_name")
    quant = config.get("model_version")
    mode = config.get("mode")

    if not all([model, quant, mode]):
        return None

    timestamp = _parse_timestamp(data.get("timestamp"))
    if timestamp is None:
        timestamp = _parse_timestamp_from_name(base)

    return RunInfo(
        path=path,
        name=path.name,
        job_id=job_id,
        timestamp=timestamp or datetime.fromtimestamp(0),
        model=model,
        quantization=str(quant),
        mode=mode,
    )


def _parse_timestamp(raw: object) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            try:
                return datetime.strptime(raw, "%Y%m%d_%H%M%S")
            except ValueError:
                return None
    return None


def _parse_timestamp_from_name(base: str) -> datetime | None:
    tokens = base.split("_")
    if len(tokens) < 3:
        return None
    # Timestamp occupies the last two tokens: YYYYMMDD and HHMMSS
    ts_tokens = tokens[-2:]
    ts_candidate = "_".join(ts_tokens)
    try:
        return datetime.strptime(ts_candidate, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def find_runs(root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        run = _parse_run_dir(candidate)
        if not run:
            continue
        if not run.summary_path.exists():
            continue
        runs.append(run)
    runs.sort(key=lambda r: (r.job_id, r.timestamp, r.mode, r.name))
    return runs


def filter_runs(
    runs: Iterable[RunInfo],
    min_job_id: int | None = None,
    max_age_hours: float | None = None,
) -> List[RunInfo]:
    filtered = []
    for run in runs:
        if min_job_id is not None and run.job_id < min_job_id:
            continue
        filtered.append(run)

    if max_age_hours is not None and filtered:
        latest_ts = max(run.timestamp for run in filtered)
        threshold = latest_ts - timedelta(hours=max_age_hours)
        filtered = [run for run in filtered if run.timestamp >= threshold]

    filtered.sort(key=lambda r: (r.job_id, r.timestamp, r.mode, r.name))
    return filtered


def group_runs_by_job(runs: Iterable[RunInfo]) -> Dict[int, List[RunInfo]]:
    grouped: Dict[int, List[RunInfo]] = defaultdict(list)
    for run in runs:
        grouped[run.job_id].append(run)
    for job_runs in grouped.values():
        job_runs.sort(key=lambda r: (r.timestamp, r.mode, r.name))
    return dict(sorted(grouped.items()))
