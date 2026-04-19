"""EXP3 counterfactual replay on the 3-arch full-factorial trace.

MobiHoc 2026 sprint, Day 2. Builds on the Day-1 scaffolding in
scripts/online_placement.py: trace loading, arm definition, and
reward function live there; this module only adds the bandit loop,
the two oracles, and the replay driver.

Algorithm: Auer et al. 2002 EXP3.  At each step t in [1, T] we
sample one problem_id uniformly with replacement, draw arm a_t from
the EXP3 distribution p_t, and observe the (normalised) reward of
that arm on that problem from the counterfactual lookup.  Because
the 3-arch trace is fully factorial (100% coverage across the 50
problems x 24 arms grid, verified by Day-1 test), the replay is
honest: every (problem, arm) cell is present, so no rejection
sampling is needed (cf. Li et al. 2011 which uses rejection).

Reward: a bounded convex combination in [0, 1] by construction,
    r = w_pass * passed
      + w_energy * (1 - min(energy_J / E_cap, 1))
      + w_latency * (1 - min(latency_s / L_cap, 1))
with defaults (w_pass, w_energy, w_latency) = (0.6, 0.3, 0.1) and
caps (E_cap, L_cap) = (200 J, 60 s).  This design avoids the
pathology of min-max normalisation over the raw signed reward,
where on-device energy outliers (up to ~2500 J per trial for a 4B
model) compress the [best, good] gap into rounding noise and
destroy the EXP3 learning signal.  The raw reward in
online_placement.py is retained for Day-1 scaffolding / ablations
and is not used by this replay driver.

Cell values are averaged over the 7 replicates per (problem, arm)
present in the trace -- the Day-1 lookup kept only the first
replicate, which left 6/7 of the data on the table.

Two oracles:
 * Stationary hindsight:  best single arm had we committed to one
   for the entire T-step sequence.  Standard EXP3 regret reference.
 * Clairvoyant per-problem: best arm per problem.  Upper bound on
   any non-context-using policy; not reachable by any learner that
   sees only the chosen arm's reward.

Output: JSON of per-seed trajectories at
  data/results/exp3_replay/exp3_results.json
Figure: research/figs/new/fig_online_placement.pdf
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path

import numpy as np

from online_placement import (
    Arm,
    arm_of,
    list_arms,
    load_trace,
)

RESULTS_PATH = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/"
    "data/results/exp3_replay/exp3_results.json"
)
FIGURE_PATH = Path(
    "/Users/vilhelmtoivonen/code/phd/pocket-agent/cli/"
    "research/figs/new/fig_online_placement.pdf"
)


@dataclass
class BoundedRewardConfig:
    """Bounded reward in [0, 1]. See module docstring."""

    w_pass: float = 0.6
    w_energy: float = 0.3
    w_latency: float = 0.1
    energy_cap_J: float = 200.0
    latency_cap_s: float = 60.0


@dataclass
class ReplayConfig:
    T: int = 1000
    n_seeds: int = 100
    reward: BoundedRewardConfig = None

    def __post_init__(self):
        if self.reward is None:
            self.reward = BoundedRewardConfig()


def bounded_reward(row, cfg: BoundedRewardConfig) -> float:
    p = 1.0 if bool(row["passed"]) else 0.0
    e = 1.0 - min(float(row["energy_j"]) / cfg.energy_cap_J, 1.0)
    l = 1.0 - min(float(row["total_time_s"]) / cfg.latency_cap_s, 1.0)
    return cfg.w_pass * p + cfg.w_energy * e + cfg.w_latency * l


def build_reward_grid(
    df, cfg: BoundedRewardConfig
) -> dict[tuple[str, Arm], float]:
    """Map (problem_id, arm) -> mean bounded reward across 7 replicates."""
    grouped: dict[tuple[str, Arm], list[float]] = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["problem_id"], arm_of(row))
        grouped[key].append(bounded_reward(row, cfg))
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def run_episode(
    problems: list[str],
    arms: list[Arm],
    grid: dict[tuple[str, Arm], float],
    T: int,
    seed: int,
) -> dict:
    """One episode: EXP3 + both oracles on the same problem sequence.

    Reward is already in [0, 1] via bounded_reward, so no rescaling.
    """
    rng_p = random.Random(seed)
    rng_a = random.Random(seed + 10**6)

    K = len(arms)
    # Auer 2002 eq. 3.2: gamma = min(1, sqrt(K ln K / ((e-1) T)))
    gamma = min(1.0, sqrt(K * log(K) / ((exp(1) - 1) * T))) if K > 1 else 1.0
    w = [1.0] * K

    problem_seq = [rng_p.choice(problems) for _ in range(T)]

    exp3_traj = np.empty(T)
    clair_traj = np.empty(T)
    hindsight_totals = np.zeros(K)

    exp3_cum = 0.0
    clair_cum = 0.0

    for t, pid in enumerate(problem_seq):
        Z = sum(w)
        p = [(1 - gamma) * wi / Z + gamma / K for wi in w]

        u = rng_a.random()
        acc = 0.0
        a_t = K - 1
        for i, pi in enumerate(p):
            acc += pi
            if u < acc:
                a_t = i
                break

        r = grid[(pid, arms[a_t])]
        # EXP3 weight update with importance-weighted reward.
        w[a_t] *= exp(gamma * (r / p[a_t]) / K)

        exp3_cum += r
        exp3_traj[t] = exp3_cum

        best_r = max(grid[(pid, a)] for a in arms)
        clair_cum += best_r
        clair_traj[t] = clair_cum

        for i, a in enumerate(arms):
            hindsight_totals[i] += grid[(pid, a)]

    best_stat_idx = int(np.argmax(hindsight_totals))
    stat_traj = np.cumsum(
        [grid[(pid, arms[best_stat_idx])] for pid in problem_seq]
    )

    return {
        "exp3": exp3_traj.tolist(),
        "stationary": stat_traj.tolist(),
        "clairvoyant": clair_traj.tolist(),
        "best_stationary_arm": list(arms[best_stat_idx]),
    }


def run_all(cfg: ReplayConfig) -> dict:
    df = load_trace()
    problems = sorted(df["problem_id"].unique().tolist())
    arms = list_arms(df)
    K = len(arms)
    grid = build_reward_grid(df, cfg.reward)

    vals = list(grid.values())
    print(f"T={cfg.T}  K={K}  n_seeds={cfg.n_seeds}")
    print(f"problems={len(problems)}  grid_cells={len(grid)}")
    print(f"cell-mean reward range: [{min(vals):.3f}, {max(vals):.3f}]")

    episodes = []
    for s in range(cfg.n_seeds):
        ep = run_episode(problems, arms, grid, cfg.T, s)
        episodes.append(ep)
        if (s + 1) % 25 == 0:
            print(f"  seed {s + 1}/{cfg.n_seeds}")

    return {
        "T": cfg.T,
        "K": K,
        "n_seeds": cfg.n_seeds,
        "reward_weights": {
            "w_pass": cfg.reward.w_pass,
            "w_energy": cfg.reward.w_energy,
            "w_latency": cfg.reward.w_latency,
            "energy_cap_J": cfg.reward.energy_cap_J,
            "latency_cap_s": cfg.reward.latency_cap_s,
        },
        "cell_reward_min": float(min(vals)),
        "cell_reward_max": float(max(vals)),
        "arms": [list(a) for a in arms],
        "episodes": episodes,
    }


def regret_curves(results: dict) -> dict:
    """Aggregate regret-over-t with 5th/95th percentile bands."""
    T = results["T"]
    exp3 = np.array([ep["exp3"] for ep in results["episodes"]])
    stat = np.array([ep["stationary"] for ep in results["episodes"]])
    clair = np.array([ep["clairvoyant"] for ep in results["episodes"]])

    regret_stat = stat - exp3
    regret_clair = clair - exp3

    def band(a: np.ndarray) -> dict:
        return {
            "mean": a.mean(axis=0).tolist(),
            "p05": np.quantile(a, 0.05, axis=0).tolist(),
            "p95": np.quantile(a, 0.95, axis=0).tolist(),
        }

    return {
        "t": list(range(1, T + 1)),
        "regret_vs_stationary": band(regret_stat),
        "regret_vs_clairvoyant": band(regret_clair),
    }


def make_figure(results: dict, out_path: Path) -> None:
    """Two-panel figure: regret-over-t (log-log) and final-regret per seed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = regret_curves(results)
    t = np.array(curves["t"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)

    ax = axes[0]
    for key, color, label in [
        ("regret_vs_stationary", "#1f77b4", "EXP3 vs stationary hindsight"),
        ("regret_vs_clairvoyant", "#d62728", "EXP3 vs clairvoyant oracle"),
    ]:
        band = curves[key]
        ax.plot(t, band["mean"], color=color, lw=1.2, label=label)
        ax.fill_between(t, band["p05"], band["p95"], color=color, alpha=0.18, lw=0)
    ax.set_xlabel("decision step $t$")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"EXP3 replay, T={results['T']}, K={results['K']}")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3, lw=0.5)

    ax = axes[1]
    final_regret_stat = np.array(
        [ep["stationary"][-1] - ep["exp3"][-1] for ep in results["episodes"]]
    )
    final_regret_clair = np.array(
        [ep["clairvoyant"][-1] - ep["exp3"][-1] for ep in results["episodes"]]
    )
    positions = [1, 2]
    parts = ax.violinplot(
        [final_regret_stat, final_regret_clair],
        positions=positions,
        showmeans=True,
        widths=0.6,
    )
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#d62728"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(["vs stationary", "vs clairvoyant"], fontsize=8)
    ax.set_ylabel(f"cumulative regret at t={results['T']}")
    ax.set_title(f"seed distribution (n={results['n_seeds']})")
    ax.grid(True, alpha=0.3, lw=0.5, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def save_results(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--w-pass", type=float, default=0.6)
    parser.add_argument("--w-energy", type=float, default=0.3)
    parser.add_argument("--w-latency", type=float, default=0.1)
    parser.add_argument("--energy-cap-J", type=float, default=200.0)
    parser.add_argument("--latency-cap-s", type=float, default=60.0)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    cfg = ReplayConfig(
        T=args.T,
        n_seeds=args.n_seeds,
        reward=BoundedRewardConfig(
            w_pass=args.w_pass,
            w_energy=args.w_energy,
            w_latency=args.w_latency,
            energy_cap_J=args.energy_cap_J,
            latency_cap_s=args.latency_cap_s,
        ),
    )
    results = run_all(cfg)
    save_results(results, RESULTS_PATH)

    curves = regret_curves(results)
    print()
    print("=== final regret at t=T ===")
    for key in ("regret_vs_stationary", "regret_vs_clairvoyant"):
        b = curves[key]
        print(
            f"{key:<26}  mean={b['mean'][-1]:>10.2f}  "
            f"[{b['p05'][-1]:.2f}, {b['p95'][-1]:.2f}]"
        )

    from collections import Counter
    best_arms = Counter(
        tuple(ep["best_stationary_arm"]) for ep in results["episodes"]
    )
    print()
    print("=== best stationary arm across seeds ===")
    for arm, n in best_arms.most_common(5):
        print(f"  {n:>3}/100  {arm}")

    if not args.no_figure:
        make_figure(results, FIGURE_PATH)


if __name__ == "__main__":
    main()
