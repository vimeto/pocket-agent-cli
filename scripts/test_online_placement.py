"""Consistency check: trace marginal pass rate vs canonical Pass@1 CSV.

For each arm in the 3-arch trace we compute the marginal pass rate over
all problems, and compare against the pass rate reported in
analysis_scripts/output_full/problem_metrics.csv for the matching
(model, mode) combination. This documents that our counterfactual
lookup is consistent (within 2pp) with the canonical Pass@1 source
used elsewhere in the paper.

Note: the trace runs over 50 problems, the canonical CSV is over 500.
We restrict the CSV comparison to the 50-problem intersection before
checking the tolerance.

Run directly:  python scripts/test_online_placement.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from online_placement import arm_of, list_arms, load_trace

CSV_PATH = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/"
    "analysis_scripts/output_full/problem_metrics.csv"
)
TOLERANCE_PP = 2.0  # percentage points


def load_csv_pass_rates(
    csv_path: Path,
    problem_filter: set[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Map (model, mode) -> (passes, total) from the canonical CSV,
    restricted to problems in problem_filter."""
    csv.field_size_limit(sys.maxsize)
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["problem_id"] not in problem_filter:
                continue
            key = (row["model"], row["mode"])
            agg[key][1] += 1
            if row["success"].strip().lower() == "true":
                agg[key][0] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def main() -> int:
    df = load_trace()
    problems_in_trace = set(df["problem_id"].unique())
    arms = list_arms(df)

    # Trace pass rate per arm.
    trace_rates: dict[tuple[str, str, str], float] = {}
    df = df.copy()
    df["arm"] = df.apply(arm_of, axis=1)
    for arm, g in df.groupby("arm"):
        trace_rates[arm] = g["passed"].mean()

    csv_rates = load_csv_pass_rates(CSV_PATH, problems_in_trace)

    print(f"{'arm':<70} {'trace':>8} {'csv':>8} {'delta_pp':>10} {'verdict':>8}")
    print("-" * 108)
    failures: list[str] = []
    skipped: list[str] = []
    for arm in arms:
        model, mode, _arch = arm
        csv_key = (model, mode)
        if csv_key not in csv_rates or csv_rates[csv_key][1] == 0:
            skipped.append(f"{arm} -- no matching CSV rows for (model={model}, mode={mode})")
            continue
        passes, total = csv_rates[csv_key]
        csv_rate = passes / total
        trace_rate = trace_rates[arm]
        delta_pp = abs(trace_rate - csv_rate) * 100.0
        ok = delta_pp <= TOLERANCE_PP
        verdict = "ok" if ok else "FAIL"
        print(
            f"{str(arm):<70} {trace_rate:>8.3f} {csv_rate:>8.3f} "
            f"{delta_pp:>10.2f} {verdict:>8}"
        )
        if not ok:
            failures.append(f"{arm}: trace={trace_rate:.3f} csv={csv_rate:.3f} "
                            f"delta={delta_pp:.2f}pp")

    print()
    if skipped:
        print(f"skipped {len(skipped)} arms (no CSV match):")
        for s in skipped:
            print(f"  - {s}")
    if failures:
        print(f"FAIL: {len(failures)} arms exceed {TOLERANCE_PP}pp tolerance")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print(f"OK: all {len(arms) - len(skipped)} comparable arms agree within "
          f"{TOLERANCE_PP}pp of canonical Pass@1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
