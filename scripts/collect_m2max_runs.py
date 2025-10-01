#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


def copy_recent_runs(source_root: Path, target_root: Path, cutoff_date: dt.date) -> int:
    copied = 0
    for bench_file in source_root.glob("**/bench_*.json"):
        try:
            data = json.loads(bench_file.read_text())
        except Exception as exc:
            print(f"[WARN] Failed to parse {bench_file}: {exc}")
            continue
        start_time = data.get("start_time")
        if not start_time:
            continue
        try:
            start_dt = dt.datetime.fromisoformat(start_time)
        except ValueError:
            continue
        if start_dt.date() != cutoff_date:
            continue

        rel_path = bench_file.relative_to(source_root)
        dest_file = target_root / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bench_file, dest_file)
        copied += 1

        run_dir = bench_file.parent / "runs" / bench_file.stem
        if run_dir.exists():
            dest_run_dir = target_root / run_dir.relative_to(source_root)
            if dest_run_dir.exists():
                shutil.rmtree(dest_run_dir)
            shutil.copytree(run_dir, dest_run_dir)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy today's benchmark runs into the workspace")
    parser.add_argument("source", type=Path, help="Source benchmark root (e.g. ~/.pocket-agent-cli/results/benchmarks)")
    parser.add_argument("target", type=Path, help="Destination directory inside the repo")
    parser.add_argument("--date", type=str, help="ISO date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    cutoff = dt.date.fromisoformat(args.date) if args.date else dt.date.today()

    copied = copy_recent_runs(args.source.expanduser(), args.target, cutoff)
    print(f"Copied {copied} benchmark summary files for {cutoff}.")


if __name__ == "__main__":
    main()
