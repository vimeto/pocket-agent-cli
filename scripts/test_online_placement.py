"""Consistency check: 3-arch trace vs same-rubric sweep sources.

For each arm in the 3-arch trace we compute the marginal pass rate over
the problems in the trace, and compare against the pass rate reported by
the sweep that uses the same subprocess-plus-returncode scoring rubric:

    local arms     -> data/results/mlx_sweep/20260403_091508/<model>_<mode>.jsonl
    cloud/hybrid   -> data/results/full_cloud_sweep/sglang_20260402_162457/<model>_<mode>.jsonl

Restricted to the 50-problem intersection of trace and sweep.

Two checks, with different strictness:

1. LOCAL arms must match MLX sweep pass rate within 1pp.
   The 3-arch-experiment local architecture is produced by copying MLX
   sweep rows verbatim (scripts/run_3arch_experiment.py:197-226), so
   any deviation here means the trace or the lookup is broken. This is
   the only strict check; its FAIL returns exit code 1.

2. CLOUD/HYBRID arms are compared to full_cloud_sweep informationally.
   The 3-arch cloud/hybrid architectures wrap SGLang with a simulated
   network + architecture-aware retry layer (see
   pocket_agent_cli/network/deployment_architectures.py), so agreement
   with cloud_sweep is expected for tool-light / large-model arms
   (typically within 5pp) but genuinely diverges for small-model
   full_tool arms where cloud_sweep's simpler harness avoids
   architecture-induced timeouts. Deltas are reported for Day-2
   awareness, but do NOT fail the test.

We deliberately do NOT cross-validate against
analysis_scripts/output_full/problem_metrics.csv: that CSV uses a
different tool-executor sandbox (benchmark_service.py) and an early-
stop aggregation that makes row means an inconsistent estimator of
Pass@1. See the provenance note in online_placement.py.

Run directly:  uv run python scripts/test_online_placement.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from online_placement import arm_of, list_arms, load_trace

MLX_SWEEP_DIR = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/"
    "data/results/mlx_sweep/20260403_091508"
)
CLOUD_SWEEP_DIR = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/"
    "data/results/full_cloud_sweep/sglang_20260402_162457"
)
LOCAL_TOLERANCE_PP = 1.0   # strict: local arms must match MLX sweep
CLOUD_NOISE_PP = 5.0       # informational threshold for cloud/hybrid deltas


def sweep_path_for(model: str, mode: str, architecture: str) -> Path:
    """Pick the same-rubric sweep file for a given arm."""
    base_dir = MLX_SWEEP_DIR if architecture == "local" else CLOUD_SWEEP_DIR
    return base_dir / f"{model}_{mode}.jsonl"


def sweep_pass_rate(
    path: Path,
    problem_filter: set[str],
) -> tuple[int, int]:
    """(passes, total) from the sweep JSONL, restricted to problem_filter."""
    passes = 0
    total = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row["problem_id"])
            if pid not in problem_filter:
                continue
            total += 1
            if bool(row.get("passed", False)):
                passes += 1
    return passes, total


def main() -> int:
    df = load_trace()
    problems_in_trace = {str(p) for p in df["problem_id"].unique()}
    arms = list_arms(df)

    df = df.copy()
    df["arm"] = df.apply(arm_of, axis=1)
    trace_rates: dict[tuple[str, str, str], float] = {
        arm: g["passed"].mean() for arm, g in df.groupby("arm")
    }

    print(
        f"{'arm':<70} {'trace':>8} {'sweep':>8} {'delta_pp':>10} {'verdict':>12}"
    )
    print("-" * 112)
    local_failures: list[str] = []
    cloud_notable: list[str] = []
    skipped: list[str] = []
    for arm in arms:
        model, mode, architecture = arm
        sweep_file = sweep_path_for(model, mode, architecture)
        if not sweep_file.exists():
            skipped.append(f"{arm} -- no sweep file at {sweep_file}")
            continue
        passes, total = sweep_pass_rate(sweep_file, problems_in_trace)
        if total == 0:
            skipped.append(f"{arm} -- sweep had 0 rows in trace problem set")
            continue
        sweep_rate = passes / total
        trace_rate = trace_rates[arm]
        delta_pp = abs(trace_rate - sweep_rate) * 100.0
        is_local = architecture == "local"
        if is_local:
            ok = delta_pp <= LOCAL_TOLERANCE_PP
            verdict = "ok" if ok else "FAIL"
            if not ok:
                local_failures.append(
                    f"{arm}: trace={trace_rate:.3f} sweep={sweep_rate:.3f} "
                    f"delta={delta_pp:.2f}pp"
                )
        else:
            if delta_pp <= CLOUD_NOISE_PP:
                verdict = "ok"
            else:
                verdict = "note"
                cloud_notable.append(
                    f"{arm}: trace={trace_rate:.3f} sweep={sweep_rate:.3f} "
                    f"delta={delta_pp:.2f}pp"
                )
        print(
            f"{str(arm):<70} {trace_rate:>8.3f} {sweep_rate:>8.3f} "
            f"{delta_pp:>10.2f} {verdict:>12}"
        )

    print()
    if skipped:
        print(f"skipped {len(skipped)} arms (no sweep source):")
        for s in skipped:
            print(f"  - {s}")
    if cloud_notable:
        print(
            f"note: {len(cloud_notable)} cloud/hybrid arms differ from "
            f"cloud_sweep by > {CLOUD_NOISE_PP}pp --"
        )
        print(
            "      expected for small-model full_tool arms where the 3-arch "
            "layer adds timeouts cloud_sweep does not model."
        )
        for n in cloud_notable:
            print(f"  - {n}")
    if local_failures:
        print(
            f"FAIL: {len(local_failures)} local arms diverge from MLX sweep "
            f"by > {LOCAL_TOLERANCE_PP}pp (this is a real bug, not experimental noise)"
        )
        for f_ in local_failures:
            print(f"  - {f_}")
        return 1
    print(
        f"OK: all local arms agree with MLX sweep within "
        f"{LOCAL_TOLERANCE_PP}pp (strict rubric-copy check)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
