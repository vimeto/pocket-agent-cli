from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

from .aggregator import aggregate_sessions_by_config, summarize_by_job
from .benchmark_loader import SessionRecord, extract_session_records
from .run_finder import filter_runs, find_runs
from .telemetry_loader import load_job_telemetry


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark and telemetry data")
    parser.add_argument(
        "--data-root",
        default="analysis_data/data/results",
        type=Path,
        help="Path to the extracted benchmark results root",
    )
    parser.add_argument(
        "--log-root",
        default="analysis_data/data/logs",
        type=Path,
        help="Path to the extracted telemetry logs root",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("analysis_scripts/output"),
        type=Path,
        help="Where to write summary tables and figures",
    )
    parser.add_argument(
        "--min-job-id",
        default=5128610,
        type=int,
        help="Ignore runs that were produced by smaller job identifiers",
    )
    parser.add_argument(
        "--max-age-hours",
        default=30.0,
        type=float,
        help="Keep runs within this many hours of the freshest run (set to 0 or negative to disable)",
    )

    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(Path(args.data_root))
    if args.max_age_hours and args.max_age_hours > 0:
        runs = filter_runs(runs, min_job_id=args.min_job_id, max_age_hours=args.max_age_hours)
    else:
        runs = filter_runs(runs, min_job_id=args.min_job_id, max_age_hours=None)

    if not runs:
        print("No runs matched the provided filters.")
        return

    session_records: List[SessionRecord] = []
    for run in runs:
        session_records.extend(extract_session_records(run))

    telemetry_by_job: Dict[int, Dict[str, float | int | None]] = {}
    for job_id in {run.job_id for run in runs}:
        telemetry_by_job[job_id] = load_job_telemetry(Path(args.log_root), job_id)

    config_rows = aggregate_sessions_by_config(session_records)
    job_rows = summarize_by_job(session_records, telemetry_by_job)

    _write_json(output_dir / "per_config_metrics.json", [row.to_dict() for row in config_rows])
    _write_csv(output_dir / "per_config_metrics.csv", [row.to_dict() for row in config_rows])
    _write_json(output_dir / "per_job_metrics.json", job_rows)
    _write_csv(output_dir / "per_job_metrics.csv", job_rows)

    _plot_pass_rates(output_dir / "pass_rates.pdf", config_rows)
    _plot_mode_metrics(output_dir / "mode_energy_power.pdf", config_rows)
    _write_mode_summary(output_dir / "mode_summary.csv", config_rows)

    print(f"Wrote {len(config_rows)} per-configuration rows and {len(job_rows)} per-job rows to {output_dir}.")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: object) -> None:
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _plot_pass_rates(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return

    entries = [row.to_dict() for row in rows]
    _inject_assumed_pass_rates(entries)
    groups = sorted({(entry["model"], entry["quantization"]) for entry in entries})
    modes = ["base", "tool_submission", "full_tool"]

    mode_colors = {
        "base": "#1f77b4",
        "tool_submission": "#ff7f0e",
        "full_tool": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = range(len(groups))
    width = 0.2

    for idx, mode in enumerate(modes):
        offsets = [x + (idx - 1) * width for x in x_positions]
        values = []
        alphas = []
        for group in groups:
            match = next(
                (entry for entry in entries if (entry["model"], entry["quantization"]) == group and entry["mode"] == mode),
                None,
            )
            values.append(match.get("pass_rate") if match else 0.0)
            alphas.append(0.5 if match and match.get("assumed_zero") else 0.9)
        for offset, value, alpha in zip(offsets, values, alphas):
            ax.bar(offset, value, width=width, label=None, color=mode_colors.get(mode, "#555555"), alpha=alpha)
        ax.bar([], [], width=width, label=mode.replace("_", " "), color=mode_colors.get(mode, "#555555"))

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([f"{model}\n{quant}" for model, quant in groups], rotation=0)
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1)
    ax.set_title("Pass@1 success rate across configurations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _inject_assumed_pass_rates(entries: List[Dict[str, object]]) -> None:
    assumed_configs = [
        ("deepseek-r1-distill-qwen-1.5b", "Q4_K_M", "tool_submission"),
        ("deepseek-r1-distill-qwen-1.5b", "F16", "tool_submission"),
    ]

    existing = {(entry["model"], entry["quantization"], entry["mode"]) for entry in entries}

    for model, quant, mode in assumed_configs:
        if (model, quant, mode) in existing:
            continue

        reference = next((entry for entry in entries if entry["model"] == model and entry["quantization"] == quant and entry["mode"] == "base"), None)
        total_problems = reference.get("total_problems") if reference else None

        entries.append(
            {
                "model": model,
                "quantization": quant,
                "mode": mode,
                "pass_rate": 0.0,
                "total_problems": total_problems,
                "energy_joules": None,
                "avg_power_watts": None,
                "assumed_zero": True,
            }
        )


def _plot_mode_metrics(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return

    entries = [row.to_dict() for row in rows]
    groups = sorted({(entry["model"], entry["quantization"]) for entry in entries})
    modes = ["base", "tool_submission", "full_tool"]

    energy_colors = {
        "base": "#1f77b4",
        "tool_submission": "#ff7f0e",
        "full_tool": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    x_positions = range(len(groups))
    width = 0.2

    # Energy per problem
    for idx, mode in enumerate(modes):
        offsets = [x + (idx - 1) * width for x in x_positions]
        values = []
        for group in groups:
            entry = next(
                (item for item in entries if (item["model"], item["quantization"]) == group and item["mode"] == mode),
                None,
            )
            if entry and entry.get("energy_joules") and entry.get("total_problems"):
                denom = entry["total_problems"] or 1
                values.append(entry["energy_joules"] / denom)
            else:
                values.append(0.0)
        axes[0].bar(offsets, values, width=width, color=energy_colors.get(mode, "#555555"), label=mode.replace("_", " "))

    axes[0].set_ylabel("Energy per problem (J)")
    axes[0].set_title("Energy expenditure by mode")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Average power
    for idx, mode in enumerate(modes):
        offsets = [x + (idx - 1) * width for x in x_positions]
        values = []
        for group in groups:
            entry = next(
                (item for item in entries if (item["model"], item["quantization"]) == group and item["mode"] == mode),
                None,
            )
            if entry and entry.get("avg_power_watts") is not None:
                values.append(entry["avg_power_watts"])
            else:
                values.append(0.0)
        axes[1].bar(offsets, values, width=width, color=energy_colors.get(mode, "#555555"))

    axes[1].set_ylabel("Average power (W)")
    axes[1].set_title("Average system power by mode")
    axes[1].set_xticks(list(x_positions))
    axes[1].set_xticklabels([f"{model}\n{quant}" for model, quant in groups], rotation=0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)


def _write_mode_summary(path: Path, rows: Sequence[object]) -> None:
    entries = [row.to_dict() for row in rows]
    entries.sort(key=lambda e: (e["model"], e["quantization"], e["mode"]))

    headers = [
        "model",
        "quantization",
        "mode",
        "total_problems",
        "passed_problems",
        "pass_rate",
        "energy_joules",
        "energy_per_problem",
        "avg_power_watts",
        "max_power_watts",
        "min_power_watts",
    ]

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for entry in entries:
            total_problems = entry.get("total_problems") or 0
            energy = entry.get("energy_joules")
            energy_per_problem = energy / total_problems if energy and total_problems else ""
            writer.writerow(
                [
                    entry.get("model"),
                    entry.get("quantization"),
                    entry.get("mode"),
                    total_problems if total_problems else "",
                    entry.get("passed_problems", ""),
                    entry.get("pass_rate", ""),
                    energy if energy else "",
                    energy_per_problem,
                    entry.get("avg_power_watts", ""),
                    entry.get("max_power_watts", ""),
                    entry.get("min_power_watts", ""),
                ]
            )
if __name__ == "__main__":
    main()
