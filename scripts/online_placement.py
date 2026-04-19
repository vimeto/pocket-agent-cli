"""Online placement scaffolding for counterfactual EXP3 replay.

MobiHoc 2026 sprint, Day 1 PM. This module loads the 3-architecture
full-factorial experiment trace and builds the data infrastructure needed
to run EXP3 replay on it tomorrow (Day 2). No bandit loop is implemented
here on purpose: Day 1 is only about clean trace -> arm/reward plumbing
so that the Day 2 EXP3 loop (Auer et al. 2002) can be added on top
without touching trace parsing or reward shaping.

Arm definition: (engine_or_model, thinking_budget, architecture).
The 3-arch trace does not carry an explicit thinking_budget field, so
we use the `mode` column (base / full_tool) as the budget proxy --
`base` suppresses explicit reasoning scaffolding, `full_tool` enables
the tool-use + reasoning prompt. Day 3 ablations can redefine the arm
tuple here without touching the reward code.

Reward: r = -alpha*energy_J - beta*latency_s + gamma*passed,
with alpha, beta, gamma exposed via RewardConfig for ablation.

Scoring provenance (important for Day 2 claims):
The `passed` field in the 3-arch trace is produced by the
subprocess-plus-tempfile test harness (local arm rows copied from
scripts/run_mlx_sweep.py:95-114; cloud/hybrid arm rows produced via
scripts/run_benchmarks_sglang.py:277-298). Both use an identical
subprocess+returncode rubric. This is the same scoring used by
full_cloud_sweep/ and mlx_sweep/, and therefore by the paper's
§5 Pass@1 headline numbers (which quote full_cloud_sweep/ summary.json).

It is NOT the same as analysis_scripts/output_full/problem_metrics.csv,
which uses a different tool-executor sandbox (pocket_agent_cli/tools/
tool_executor.py) and is subject to an early-stop aggregation quirk
(pocket_agent_cli/benchmarks/benchmark_service.py:1391). A naive row
mean over `problem_metrics.csv.success` therefore under-reports Pass@1
by a large margin; for CSV-derived per-model Pass@1 use the Wilson
estimator output at analysis_scripts/output_full/model_mode_summary.csv
instead.

Bottom line: the EXP3 reward here is consistent with the paper's §5
headline numbers. Do NOT cross-validate the trace against
problem_metrics.csv row means.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

TRACE_PATH = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/data/results/"
    "3arch_experiment/20260405_183229/3arch_results_merged.jsonl"
)

# Arm tuple: (engine_or_model, thinking_budget, architecture).
# In this trace, `mode` stands in for thinking_budget.
Arm = tuple[str, str, str]


@dataclass
class RewardConfig:
    """Weights for r = -alpha*energy_J - beta*latency_s + gamma*passed."""

    alpha: float = 1.0   # energy weight (J)
    beta: float = 1.0    # latency weight (s)
    gamma: float = 10.0  # pass bonus (one pass = 10 reward units)


def load_trace(path: Path = TRACE_PATH) -> pd.DataFrame:
    """Read the JSONL trace into a DataFrame."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Normalise the columns we depend on.
    df["passed"] = df["passed"].astype(bool)
    df["total_time_s"] = df["total_time_s"].astype(float)
    df["problem_id"] = df["problem_id"].astype(str)
    # Cloud / hybrid trials log no on-device energy; treat missing as 0 J
    # of device energy (matches the paper's offload accounting).
    df["energy_j"] = pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0)
    return df


def arm_of(row) -> Arm:
    """Extract the arm tuple from a trace row (Series or dict)."""
    return (row["model"], row["mode"], row["architecture"])


def list_arms(df: pd.DataFrame) -> list[Arm]:
    """Return all unique arms present in the trace, deterministically sorted."""
    arms = {arm_of(r) for _, r in df.iterrows()}
    return sorted(arms)


def reward(row, cfg: RewardConfig) -> float:
    """Scalar reward for a single trial row."""
    return (
        -cfg.alpha * float(row["energy_j"])
        - cfg.beta * float(row["total_time_s"])
        + cfg.gamma * (1.0 if bool(row["passed"]) else 0.0)
    )


def build_lookup(df: pd.DataFrame) -> dict[tuple[str, Arm], dict]:
    """Map (problem_id, arm) -> trial row (as dict).

    If multiple rows exist for the same (problem_id, arm) we keep the
    first; the trace in the wild has no duplicates, but we guard anyway.
    """
    out: dict[tuple[str, Arm], dict] = {}
    for _, row in df.iterrows():
        key = (row["problem_id"], arm_of(row))
        if key not in out:
            out[key] = row.to_dict()
    return out


def lookup(
    table: dict[tuple[str, Arm], dict],
    problem_id: str,
    arm: Arm,
) -> Optional[dict]:
    """Counterfactual lookup: return the observed trial row or None."""
    return table.get((problem_id, arm))


def coverage_stats(
    df: pd.DataFrame,
    table: dict[tuple[str, Arm], dict],
) -> dict:
    """Summary stats over the counterfactual grid."""
    problems = sorted(df["problem_id"].unique())
    arms = list_arms(df)
    total_cells = len(problems) * len(arms)
    filled = sum(
        1
        for pid in problems
        for arm in arms
        if (pid, arm) in table
    )
    return {
        "num_problems": len(problems),
        "K_arms": len(arms),
        "total_cells": total_cells,
        "filled_cells": filled,
        "coverage": filled / total_cells if total_cells else 0.0,
    }


def reward_per_arm(
    df: pd.DataFrame,
    cfg: RewardConfig,
) -> pd.DataFrame:
    """mean / min / max reward per arm, across all problems in the trace."""
    df = df.copy()
    df["reward"] = df.apply(lambda r: reward(r, cfg), axis=1)
    df["arm"] = df.apply(arm_of, axis=1)
    agg = (
        df.groupby("arm")["reward"]
        .agg(["mean", "min", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    return agg


def _sanity_random_replay(
    df: pd.DataFrame,
    table: dict[tuple[str, Arm], dict],
    cfg: RewardConfig,
    n: int = 100,
    seed: int = 0,
) -> dict:
    """Pull n random (problem, arm) pairs; confirm lookup + reward work."""
    rng = random.Random(seed)
    problems = sorted(df["problem_id"].unique())
    arms = list_arms(df)
    hits = 0
    rewards = []
    for _ in range(n):
        pid = rng.choice(problems)
        arm = rng.choice(arms)
        row = lookup(table, pid, arm)
        if row is not None:
            hits += 1
            rewards.append(reward(row, cfg))
    return {
        "n_trials": n,
        "hits": hits,
        "miss": n - hits,
        "mean_reward_on_hits": (sum(rewards) / len(rewards)) if rewards else float("nan"),
    }


def main() -> None:
    cfg = RewardConfig()
    df = load_trace()
    table = build_lookup(df)

    stats = coverage_stats(df, table)
    print("=== coverage ===")
    print(f"problems:    {stats['num_problems']}")
    print(f"K arms:      {stats['K_arms']}")
    print(f"cells:       {stats['filled_cells']} / {stats['total_cells']}")
    print(f"coverage:    {stats['coverage']:.1%}")

    print()
    print("=== reward per arm ===")
    agg = reward_per_arm(df, cfg)
    # Full listing -- K is small.
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(agg)

    print()
    print("=== top-5 arms by mean reward ===")
    print(agg.head(5))

    print()
    print("=== sanity: 100 random (problem, arm) lookups ===")
    sanity = _sanity_random_replay(df, table, cfg, n=100, seed=0)
    for k, v in sanity.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
