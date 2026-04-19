"""Compute 95% cluster-bootstrap CIs for every headline claim in paper.tex.

MobiHoc 2026 sprint, Day 2 AM. Loads the canonical per-problem JSONL
sources used by `scripts/test_online_placement.py` and the 3-arch
placement policy data, attaches a cluster-bootstrap CI to each
headline number, and writes:

    research/figures/headline_cis.tex   -- LaTeX \\newcommand macros
    research/figures/headline_cis.json  -- provenance audit sidecar

Run directly:  uv run python analysis_scripts/compute_headline_cis.py

Nothing here touches `online_placement.py`, `test_online_placement.py`,
`exp3_replay.py`, or `placement_policy.py` -- those remain the source
of truth for the point estimates. This script re-derives the point
estimates from their underlying data and adds a CI. If a fresh point
estimate disagrees with the paper, that's flagged in the audit JSON
(`paper_claim` vs `point`) rather than silently overwritten.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_scripts.bootstrap_stats import (  # noqa: E402
    cluster_bootstrap_ci,
    diff_ci,
    pass_rate_ci,
)

# ── Paths ───────────────────────────────────────────────────────────────────

REPO = Path("/Users/vilhelmtoivonen/code/phd/pocket-agent/cli")
MBPP_SWEEP_DIR = REPO / "data/results/full_cloud_sweep/sglang_20260402_162457"
MBPP_SWEEP_DIR_171922 = REPO / "data/results/full_cloud_sweep/sglang_20260402_171922"
HUMANEVAL_DIRS = [
    REPO / "data/results/humaneval_cloud/sglang_20260405_230217",
    REPO / "data/results/humaneval_cloud/sglang_20260405_230509",
    REPO / "data/results/humaneval_cloud/sglang_20260405_232059",
]
COMPRESSION_L0L3_DIR = (
    REPO / "data/results/compression_experiment/20260407_105657"
)
COMPRESSION_L4L5_DIR = (
    REPO / "data/results/compression_experiment/20260407_114649"
)
EARLY_EXIT_DIRS = {
    "qwen-3-4b": REPO / "data/results/early_exit/early_exit_20260405_011122",
    "qwen-3-0.6b": REPO / "data/results/early_exit/early_exit_20260405_140052",
    "deepseek-r1-distill-qwen-1.5b": REPO
    / "data/results/early_exit/early_exit_20260405_184631",
    "qwen-3.5-4b": REPO / "data/results/early_exit/early_exit_20260407_070631",
}
THREEARCH_TRACE = (
    REPO / "data/results/3arch_experiment/20260405_183229/3arch_results_merged.jsonl"
)

OUT_TEX = REPO / "research/figures/headline_cis.tex"
OUT_JSON = REPO / "research/figures/headline_cis.json"

N_RESAMPLES = 10_000
SEED = 0


# ── Loaders (same rubric as scripts/test_online_placement.py) ────────────────


def load_sweep_jsonl(path: Path) -> dict[str, list[bool]]:
    """Return {problem_id: [passed]} from a same-rubric sweep JSONL."""
    rows: dict[str, list[bool]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["problem_id"])].append(bool(r.get("passed", False)))
    return dict(rows)


def load_humaneval_sweep(model: str, mode: str) -> dict[str, list[bool]]:
    """HumanEval is split across three timestamped dirs -- probe them all."""
    for d in HUMANEVAL_DIRS:
        p = d / f"{model}_{mode}.jsonl"
        if p.exists():
            return load_sweep_jsonl(p)
    raise FileNotFoundError(f"HumanEval sweep missing for {model}_{mode}")


def pick_mbpp_sweep(model: str) -> Path:
    """sglang_20260402_162457 has empty gemma-3n files; fall back to 171922."""
    primary = MBPP_SWEEP_DIR
    if model == "gemma-3n-e2b-it":
        return MBPP_SWEEP_DIR_171922
    return primary


def load_mbpp_sweep(model: str, mode: str) -> dict[str, list[bool]]:
    d = pick_mbpp_sweep(model)
    return load_sweep_jsonl(d / f"{model}_{mode}.jsonl")


# ── Claim registry ──────────────────────────────────────────────────────────


@dataclass
class Claim:
    macro: str  # LaTeX command stem, without leading backslash
    label: str  # Human-readable short description
    source: str  # Which file / transformation produced the data
    paper_claim: str  # The exact claim as it appears in paper.tex
    unit: str = "%"  # "%" or "pp" (percentage points)
    point: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n: int = 0
    clusters: int = 0
    notes: str = ""


CLAIMS: list[Claim] = []


def format_pct(x: float, unit: str) -> str:
    if unit == "%":
        return f"{x * 100:.1f}\\%"
    if unit == "pp":
        return f"{x * 100:.1f}pp"
    return f"{x:.3f}"


def format_ci(lo: float, hi: float, unit: str) -> str:
    if unit in ("%", "pp"):
        return f"[{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"[{lo:.3f}, {hi:.3f}]"


def add_pass_rate_claim(
    macro: str,
    label: str,
    source: str,
    paper_claim: str,
    rows_by_pid: dict[str, list[bool]],
) -> None:
    point, lo, hi = pass_rate_ci(
        rows_by_pid, n_resamples=N_RESAMPLES, seed=SEED
    )
    n = sum(len(v) for v in rows_by_pid.values())
    CLAIMS.append(
        Claim(
            macro=macro,
            label=label,
            source=source,
            paper_claim=paper_claim,
            unit="%",
            point=point,
            lo=lo,
            hi=hi,
            n=n,
            clusters=len(rows_by_pid),
        )
    )
    print(
        f"  [{macro}] {label}: {point * 100:.1f}% "
        f"[{lo * 100:.1f}, {hi * 100:.1f}]  (n={n}, clusters={len(rows_by_pid)})"
    )


def add_diff_claim(
    macro: str,
    label: str,
    source: str,
    paper_claim: str,
    a_by_pid: dict[str, list[bool]],
    b_by_pid: dict[str, list[bool]],
) -> None:
    """CI for pass(A) - pass(B), paired over shared problem_ids."""
    a_f = {k: [float(x) for x in v] for k, v in a_by_pid.items()}
    b_f = {k: [float(x) for x in v] for k, v in b_by_pid.items()}
    point, lo, hi = diff_ci(
        a_f, b_f, n_resamples=N_RESAMPLES, seed=SEED
    )
    shared = set(a_by_pid) & set(b_by_pid)
    n = sum(len(a_by_pid[k]) for k in shared) + sum(
        len(b_by_pid[k]) for k in shared
    )
    CLAIMS.append(
        Claim(
            macro=macro,
            label=label,
            source=source,
            paper_claim=paper_claim,
            unit="pp",
            point=point,
            lo=lo,
            hi=hi,
            n=n,
            clusters=len(shared),
        )
    )
    print(
        f"  [{macro}] {label}: {point * 100:.2f}pp "
        f"[{lo * 100:.2f}, {hi * 100:.2f}]  (clusters={len(shared)})"
    )


# ── MBPP Pass@1 per model+mode (base mode, §5 L321) ─────────────────────────


def compute_mbpp_base_passes() -> None:
    """Per-model MBPP Pass@1 from cloud FP16 sweep across all three modes."""
    print("\n== MBPP Pass@1 (cloud FP16, 500 problems) ==")
    # (model, mode, macro, label, paper_claim)
    entries = [
        ("qwen-3-4b", "base", "passQwenThreeFourBBaseMBPP",
         "Qwen 3 (4B) MBPP Base", "Base 68\\%"),
        ("qwen-3-4b", "tool_submission", "passQwenThreeFourBToolSubMBPP",
         "Qwen 3 (4B) MBPP Tool-Sub", "Tool-Submission 79\\%"),
        ("qwen-3-4b", "full_tool", "passQwenThreeFourBFullToolMBPP",
         "Qwen 3 (4B) MBPP Full-Tool", "Full-Tool 77\\%"),
        ("llama-3.2-3b-instruct", "base", "passLlamaBaseMBPP",
         "Llama 3.2 MBPP Base", "Llama 3.2 drops from 50\\%"),
        ("llama-3.2-3b-instruct", "full_tool", "passLlamaFullToolMBPP",
         "Llama 3.2 MBPP Full-Tool", "to 43\\% (Full-Tool)"),
        ("deepseek-r1-distill-qwen-1.5b", "base", "passDeepSeekBaseMBPP",
         "DeepSeek R1 MBPP Base", "DeepSeek R1 from 42\\%"),
        ("deepseek-r1-distill-qwen-1.5b", "full_tool", "passDeepSeekFullToolMBPP",
         "DeepSeek R1 MBPP Full-Tool", "DeepSeek R1 ... to 33\\%"),
        ("gemma-3n-e2b-it", "base", "passGemmaBaseMBPP",
         "Gemma 3n MBPP Base", "Tool-Submission matches Base at 61\\%"),
        ("gemma-3n-e2b-it", "tool_submission", "passGemmaToolSubMBPP",
         "Gemma 3n MBPP Tool-Sub", "Tool-Submission matches Base at 61\\%"),
        ("gemma-3n-e2b-it", "full_tool", "passGemmaFullToolMBPP",
         "Gemma 3n MBPP Full-Tool", "Full-Tool drops sharply to 35\\%"),
        ("qwen-3-0.6b", "base", "passQwenZeroSixBBaseMBPP",
         "Qwen 3 (0.6B) MBPP Base", "Qwen 3 (0.6B) degrades from 42\\%"),
        ("qwen-3-0.6b", "full_tool", "passQwenZeroSixBFullToolMBPP",
         "Qwen 3 (0.6B) MBPP Full-Tool", "to 34\\% in both tool modes"),
        ("qwen-3-0.6b", "tool_submission", "passQwenZeroSixBToolSubMBPP",
         "Qwen 3 (0.6B) MBPP Tool-Sub", "to 34\\% in both tool modes"),
    ]
    for model, mode, macro, label, paper_claim in entries:
        try:
            rows = load_mbpp_sweep(model, mode)
        except FileNotFoundError:
            print(f"  SKIP {macro}: sweep file missing")
            continue
        add_pass_rate_claim(
            macro=macro,
            label=label,
            source=f"data/results/full_cloud_sweep/"
            f"{pick_mbpp_sweep(model).name}/{model}_{mode}.jsonl",
            paper_claim=paper_claim,
            rows_by_pid=rows,
        )


# ── HumanEval Pass@1 per model+mode (§5 L323) ───────────────────────────────


def compute_humaneval_passes() -> None:
    """Per-model HumanEval Pass@1 from cloud sweep (164 problems)."""
    print("\n== HumanEval Pass@1 (cloud FP16, 164 problems) ==")
    entries = [
        ("qwen-3-4b", "base", "passQwenThreeFourBBaseHumanEval",
         "Qwen 3 (4B) HumanEval Base", "73\\% (Base)"),
        ("qwen-3-4b", "tool_submission", "passQwenThreeFourBToolSubHumanEval",
         "Qwen 3 (4B) HumanEval Tool-Sub", "80\\% in both tool modes"),
        ("qwen-3-4b", "full_tool", "passQwenThreeFourBFullToolHumanEval",
         "Qwen 3 (4B) HumanEval Full-Tool", "80\\% in both tool modes"),
    ]
    for model, mode, macro, label, paper_claim in entries:
        try:
            rows = load_humaneval_sweep(model, mode)
        except FileNotFoundError:
            print(f"  SKIP {macro}: HumanEval sweep missing")
            continue
        add_pass_rate_claim(
            macro=macro,
            label=label,
            source=f"data/results/humaneval_cloud/.../{model}_{mode}.jsonl",
            paper_claim=paper_claim,
            rows_by_pid=rows,
        )


# ── Tool-benefit / -harm deltas (§5 L321-325) ───────────────────────────────


def compute_tool_deltas() -> None:
    """Paired deltas like Qwen 3 4B Base->Full-Tool (+9pp) etc."""
    print("\n== Tool-mode deltas on MBPP (paired cluster-bootstrap) ==")
    entries = [
        # (model, mode_a, mode_b, macro, label, paper_claim)
        ("qwen-3-4b", "full_tool", "base", "deltaQwenThreeFourBFullToolMBPP",
         "Qwen 3 (4B) Full-Tool - Base", "net +9pp gain from structured tool use"),
        ("qwen-3-4b", "tool_submission", "base", "deltaQwenThreeFourBToolSubMBPP",
         "Qwen 3 (4B) Tool-Sub - Base", "(Base 68%, Tool-Submission 79%)"),
        ("llama-3.2-3b-instruct", "full_tool", "base", "deltaLlamaFullToolMBPP",
         "Llama 3.2 Full-Tool - Base", "Llama 3.2 drops from 50% to 43% (-7pp)"),
        ("deepseek-r1-distill-qwen-1.5b", "full_tool", "base",
         "deltaDeepSeekFullToolMBPP", "DeepSeek R1 Full-Tool - Base",
         "DeepSeek R1 from 42% to 33% (-9pp)"),
        ("gemma-3n-e2b-it", "full_tool", "base", "deltaGemmaFullToolMBPP",
         "Gemma 3n Full-Tool - Base", "Full-Tool drops sharply to 35%"),
        ("qwen-3-4b", "full_tool", "base", "deltaQwenThreeFourBFullToolHumanEvalDup",
         "(placeholder, overwritten below)", ""),
    ]
    for model, mode_a, mode_b, macro, label, paper_claim in entries[:-1]:
        try:
            a = load_mbpp_sweep(model, mode_a)
            b = load_mbpp_sweep(model, mode_b)
        except FileNotFoundError:
            print(f"  SKIP {macro}: sweep missing")
            continue
        add_diff_claim(
            macro=macro,
            label=label,
            source=f"MBPP sweep ({model}_{mode_a} - {model}_{mode_b})",
            paper_claim=paper_claim,
            a_by_pid=a,
            b_by_pid=b,
        )

    # HumanEval Qwen 3 4B (+7pp)
    try:
        a = load_humaneval_sweep("qwen-3-4b", "full_tool")
        b = load_humaneval_sweep("qwen-3-4b", "base")
        add_diff_claim(
            macro="deltaQwenThreeFourBFullToolHumanEval",
            label="Qwen 3 (4B) HumanEval Full-Tool - Base",
            source="HumanEval sweep (qwen-3-4b Full-Tool - Base)",
            paper_claim="Qwen 3 (4B) rises from 73% (Base) to 80% in both tool modes (+7pp)",
            a_by_pid=a,
            b_by_pid=b,
        )
    except FileNotFoundError:
        print("  SKIP deltaQwenThreeFourBFullToolHumanEval: HumanEval missing")


# ── Early-exit thinking budget (§5 L412-417, abstract L64) ─────────────────


def load_early_exit_run(
    run_dir: Path, budget_label: str
) -> dict[str, list[bool]]:
    """Early-exit JSONL: one row per problem, `passed: bool`, 50 problems."""
    # Budget labels on disk: "0", "256", ..., "4096", "unlimited"
    # File name pattern: <model>_budget_<label>.jsonl (one per model)
    files = list(run_dir.glob(f"*_budget_{budget_label}.jsonl"))
    if not files:
        raise FileNotFoundError(f"no file matching *_budget_{budget_label}.jsonl in {run_dir}")
    # Each early-exit run directory is single-model.
    path = files[0]
    return load_sweep_jsonl(path)


def compute_early_exit_gains() -> None:
    """Capped-budget vs unlimited gains in accuracy and energy per model.

    The paper cites a +6--10pp accuracy gain and -39--84% energy drop
    across four thinking models. We bootstrap the accuracy gain
    (paired over 50 problems) and the point energy ratio -- energy
    per run is a trace-level aggregate with one number per budget,
    so we skip a CI on the energy percent (would require per-problem
    energy, which these JSONLs do not carry consistently).
    """
    print("\n== Early-exit thinking budget gains (MBPP 50-problem) ==")
    # Paper-optimal budget per model (from Fig. \ref{fig:earlyexit} caption).
    optimal_budget = {
        "qwen-3-4b": "2048",
        "qwen-3-0.6b": "2048",
        "deepseek-r1-distill-qwen-1.5b": "2048",
        "qwen-3.5-4b": "1024",
    }
    macros = {
        "qwen-3-4b": "thinkBudgetGainQwenThreeFourB",
        "qwen-3-0.6b": "thinkBudgetGainQwenZeroSixB",
        "deepseek-r1-distill-qwen-1.5b": "thinkBudgetGainDeepSeek",
        "qwen-3.5-4b": "thinkBudgetGainQwenThreePointFiveFourB",
    }
    paper_claim_per_model = {
        "qwen-3-4b": "+6pp (Qwen 3 4B unlimited 72% -> 2048 78%)",
        "qwen-3-0.6b": "flat 30% (0.6B too weak to benefit)",
        "deepseek-r1-distill-qwen-1.5b":
            "+8pp (DeepSeek unlimited 40% -> 2048 48%)",
        "qwen-3.5-4b": "+10pp (Qwen 3.5 4B unlimited 56% -> 1024 66%)",
    }
    for model, run_dir in EARLY_EXIT_DIRS.items():
        if not run_dir.exists():
            print(f"  SKIP {model}: dir missing")
            continue
        budget = optimal_budget[model]
        try:
            capped = load_early_exit_run(run_dir, budget)
            unlimited = load_early_exit_run(run_dir, "unlimited")
        except FileNotFoundError as e:
            print(f"  SKIP {model}: {e}")
            continue
        add_diff_claim(
            macro=macros[model],
            label=f"Early-exit gain @B={budget} ({model}): capped - unlimited",
            source=f"data/results/early_exit/{run_dir.name}",
            paper_claim=paper_claim_per_model[model],
            a_by_pid=capped,
            b_by_pid=unlimited,
        )


# ── Placement-policy claims: 97.5% of oracle, 62% energy (abstract) ────────


def compute_placement_claims() -> None:
    """Cluster-bootstrap COST_AWARE vs ALWAYS_LOCAL and vs ORACLE.

    We reconstruct per-problem outcomes from the 3-arch trace:
    for each problem and each (model, mode, network_condition), a policy
    picks one architecture; we then read off `passed` and an analytical
    energy from the matching row. Clustering on problem_id gives 50
    clusters.
    """
    print("\n== Placement-policy CIs (3-arch trace, 50-problem clusters) ==")
    if not THREEARCH_TRACE.exists():
        print("  SKIP: 3-arch trace missing")
        return
    df = pd.read_json(THREEARCH_TRACE, lines=True)

    # The 3-arch trace is the "ground truth" for pass/fail per
    # (problem, model, mode, architecture, network). We read the policies'
    # architecture decisions from the already-logged
    # `policy_evaluation.json`, so we do NOT need to re-implement the
    # policy rules here (and in particular never edit placement_policy.py).
    pe_path = REPO / "data/results/placement_policy/policy_evaluation.json"
    if not pe_path.exists():
        print("  SKIP: policy_evaluation.json missing")
        return
    with pe_path.open() as f:
        pe = json.load(f)

    # Build a {(model, mode, network): chosen_arch} lookup per policy.
    def policy_choices(policy: str) -> dict[tuple, str]:
        choices = {}
        for rec in pe["per_policy_results"][policy]:
            choices[(rec["model"], rec["mode"], rec["network_condition"])] = (
                rec["chosen_architecture"]
            )
        return choices

    cost_aware_choices = policy_choices("COST_AWARE")
    oracle_choices = policy_choices("ORACLE")
    always_local_choices = policy_choices("ALWAYS_LOCAL")

    # Per-problem passes for each policy, keyed by problem_id.
    def per_problem_passes(
        choices: dict[tuple, str],
    ) -> dict[str, list[bool]]:
        rows_by_pid: dict[str, list[bool]] = defaultdict(list)
        # Index the trace by (model, mode, architecture, network, problem_id).
        by_key = defaultdict(list)
        for row in df.itertuples(index=False):
            by_key[
                (row.model, row.mode, row.architecture, row.network_condition)
            ].append(row)
        for (model, mode, net), arch in choices.items():
            key = (model, mode, arch, net)
            for row in by_key.get(key, []):
                rows_by_pid[str(row.problem_id)].append(bool(row.passed))
        return dict(rows_by_pid)

    cost_aware_passes = per_problem_passes(cost_aware_choices)
    oracle_passes = per_problem_passes(oracle_choices)
    always_local_passes = per_problem_passes(always_local_choices)

    # Pass rates (just for sanity / separate CIs)
    add_pass_rate_claim(
        macro="costAwarePassRate",
        label="COST_AWARE policy pass rate (across eval points)",
        source="3-arch trace + policy_evaluation.json COST_AWARE",
        paper_claim="cost-aware heuristic ... 97.5% of a clairvoyant oracle's pass rate",
        rows_by_pid=cost_aware_passes,
    )
    add_pass_rate_claim(
        macro="oraclePassRate",
        label="ORACLE policy pass rate (across eval points)",
        source="3-arch trace + policy_evaluation.json ORACLE",
        paper_claim="oracle upper-bound reference",
        rows_by_pid=oracle_passes,
    )

    # Ratio: cost_aware / oracle  (paper says 97.5%)
    # We bootstrap the ratio of means clustered on problem_id.
    shared_pids = sorted(set(cost_aware_passes) & set(oracle_passes))
    if shared_pids:
        ca_arr = [np.asarray(cost_aware_passes[k], dtype=float) for k in shared_pids]
        or_arr = [np.asarray(oracle_passes[k], dtype=float) for k in shared_pids]
        ca_flat = np.concatenate(ca_arr)
        or_flat = np.concatenate(or_arr)
        point_ratio = float(ca_flat.mean() / or_flat.mean())
        ca_lengths = np.asarray([a.size for a in ca_arr], dtype=np.int64)
        or_lengths = np.asarray([a.size for a in or_arr], dtype=np.int64)
        ca_off = np.zeros(len(shared_pids) + 1, dtype=np.int64)
        or_off = np.zeros(len(shared_pids) + 1, dtype=np.int64)
        ca_off[1:] = np.cumsum(ca_lengths)
        or_off[1:] = np.cumsum(or_lengths)

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(shared_pids), size=(N_RESAMPLES, len(shared_pids)))
        stats = np.empty(N_RESAMPLES, dtype=float)
        for i in range(N_RESAMPLES):
            picks = idx[i]
            ca_parts = [ca_flat[ca_off[p] : ca_off[p + 1]] for p in picks]
            or_parts = [or_flat[or_off[p] : or_off[p + 1]] for p in picks]
            ca_s = np.concatenate(ca_parts) if ca_parts else np.array([])
            or_s = np.concatenate(or_parts) if or_parts else np.array([])
            or_mean = or_s.mean() if or_s.size else 1.0
            stats[i] = ca_s.mean() / or_mean if or_mean else float("nan")
        lo = float(np.nanquantile(stats, 0.025))
        hi = float(np.nanquantile(stats, 0.975))
        CLAIMS.append(
            Claim(
                macro="costAwareVsOracle",
                label="COST_AWARE / ORACLE pass-rate ratio",
                source="3-arch trace clustered on problem_id",
                paper_claim="97.5\\% of a clairvoyant oracle's pass rate",
                unit="%",
                point=point_ratio,
                lo=lo,
                hi=hi,
                n=int(ca_flat.size + or_flat.size),
                clusters=len(shared_pids),
                notes="ratio of cluster-bootstrap means",
            )
        )
        print(
            f"  [costAwareVsOracle] ratio: {point_ratio * 100:.1f}% "
            f"[{lo * 100:.1f}, {hi * 100:.1f}]  (clusters={len(shared_pids)})"
        )

    # Energy 62% claim: (avg_energy[ALWAYS_LOCAL] - avg_energy[COST_AWARE]) /
    # avg_energy[ALWAYS_LOCAL]. Raw trace `energy_j` is only populated for
    # local rows (cloud/hybrid use an analytical estimator downstream).
    # We therefore apply the same analytical formula as
    # `scripts/placement_policy.py::estimate_energy` per-problem (using
    # each row's `tokens` and `iterations` instead of the per-group
    # averages) so that clustering over problem_id is meaningful.
    # This mirrors -- and does NOT modify -- the paper's estimator.
    sys.path.insert(0, str(REPO / "scripts"))
    from placement_policy import estimate_energy  # noqa: E402

    cost_params_path = REPO / "data/results/cost_model/parameters.json"
    radio_path = (
        REPO / "data/results/traffic_characterization/"
        "traffic_char_radio_states.json"
    )
    if not cost_params_path.exists() or not radio_path.exists():
        print(
            "  SKIP costAwareEnergyReduction: cost_model/parameters.json or "
            "traffic_char_radio_states.json missing"
        )
        return
    with cost_params_path.open() as f:
        cost_params = json.load(f)
    with radio_path.open() as f:
        radio_data = json.load(f)

    by_key_rows = defaultdict(list)
    for row in df.itertuples(index=False):
        by_key_rows[
            (row.model, row.mode, row.architecture, row.network_condition)
        ].append(row)

    def row_energy(row) -> float:
        # Build a per-row dict shaped like what estimate_energy expects,
        # but substituting per-problem `tokens` and `iterations` for the
        # per-group `avg_tokens`/`avg_iterations`.
        entry = {
            "model": row.model,
            "architecture": row.architecture,
            "avg_tokens": float(row.tokens),
            "avg_iterations": float(row.iterations),
            "avg_radio_tail_energy_j": float(row.radio_tail_energy_j or 0.0),
        }
        return float(estimate_energy(entry, cost_params, radio_data))

    def per_problem_energy(choices: dict[tuple, str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = defaultdict(list)
        for (model, mode, net), arch in choices.items():
            for row in by_key_rows.get((model, mode, arch, net), []):
                out[str(row.problem_id)].append(row_energy(row))
        return dict(out)

    ca_e = per_problem_energy(cost_aware_choices)
    al_e = per_problem_energy(always_local_choices)
    shared_e_pids = sorted(set(ca_e) & set(al_e))
    if shared_e_pids:
        ca_arr = [np.asarray(ca_e[k], dtype=float) for k in shared_e_pids]
        al_arr = [np.asarray(al_e[k], dtype=float) for k in shared_e_pids]
        ca_flat = np.concatenate(ca_arr)
        al_flat = np.concatenate(al_arr)
        point_red = float(1.0 - ca_flat.mean() / al_flat.mean())
        ca_lengths = np.asarray([a.size for a in ca_arr], dtype=np.int64)
        al_lengths = np.asarray([a.size for a in al_arr], dtype=np.int64)
        ca_off = np.zeros(len(shared_e_pids) + 1, dtype=np.int64)
        al_off = np.zeros(len(shared_e_pids) + 1, dtype=np.int64)
        ca_off[1:] = np.cumsum(ca_lengths)
        al_off[1:] = np.cumsum(al_lengths)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(
            0, len(shared_e_pids), size=(N_RESAMPLES, len(shared_e_pids))
        )
        stats = np.empty(N_RESAMPLES, dtype=float)
        for i in range(N_RESAMPLES):
            picks = idx[i]
            ca_parts = [ca_flat[ca_off[p] : ca_off[p + 1]] for p in picks]
            al_parts = [al_flat[al_off[p] : al_off[p + 1]] for p in picks]
            ca_s = np.concatenate(ca_parts) if ca_parts else np.array([])
            al_s = np.concatenate(al_parts) if al_parts else np.array([])
            al_mean = al_s.mean() if al_s.size else 1.0
            stats[i] = (
                1.0 - (ca_s.mean() / al_mean) if al_mean else float("nan")
            )
        lo = float(np.nanquantile(stats, 0.025))
        hi = float(np.nanquantile(stats, 0.975))
        CLAIMS.append(
            Claim(
                macro="costAwareEnergyReduction",
                label="Energy reduction: 1 - COST_AWARE/ALWAYS_LOCAL",
                source="3-arch trace raw energy_j clustered on problem_id",
                paper_claim="reducing energy by 62\\% versus always-local execution",
                unit="%",
                point=point_red,
                lo=lo,
                hi=hi,
                n=int(ca_flat.size + al_flat.size),
                clusters=len(shared_e_pids),
                notes=(
                    "raw trace energy_j; paper uses analytical estimator in "
                    "scripts/placement_policy.py (avg 470.68 J local, 177.48 J "
                    "cost_aware -> 62.2%). This CI brackets the measured-energy "
                    "analog; review before overwriting the paper number."
                ),
            )
        )
        print(
            f"  [costAwareEnergyReduction] raw energy reduction: "
            f"{point_red * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]  "
            f"(clusters={len(shared_e_pids)})  "
            f"-- paper quotes 62% from analytical estimator"
        )


# ── Write outputs ───────────────────────────────────────────────────────────


def write_tex() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by analysis_scripts/compute_headline_cis.py",
        "% Do not edit by hand. Re-run the script to refresh.",
        "% Each claim exposes two macros: the point estimate (\\*macro) and",
        "% the 95% cluster-bootstrap CI string (\\*macroCI).",
        "",
    ]
    for c in CLAIMS:
        lines.append(f"% {c.label}")
        lines.append(f"%   source: {c.source}")
        lines.append(f"%   paper claim: {c.paper_claim}")
        lines.append(
            f"\\newcommand{{\\{c.macro}}}{{{format_pct(c.point, c.unit)}}}"
        )
        lines.append(
            f"\\newcommand{{\\{c.macro}CI}}{{{format_ci(c.lo, c.hi, c.unit)}}}"
        )
        lines.append("")
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_TEX}  ({len(CLAIMS)} claims, "
          f"{len(CLAIMS) * 2} macros)")


def write_json() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generator": "analysis_scripts/compute_headline_cis.py",
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "alpha": 0.05,
        "claims": [
            {
                "macro": c.macro,
                "label": c.label,
                "source": c.source,
                "paper_claim": c.paper_claim,
                "unit": c.unit,
                "point": c.point,
                "lo": c.lo,
                "hi": c.hi,
                "n": c.n,
                "clusters": c.clusters,
                "notes": c.notes,
            }
            for c in CLAIMS
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_JSON}")


def compute_compression_claims() -> None:
    """Context-compression L2 vs L0 delta on MBPP (Qwen 3.5 4B, 30 problems).

    Paper claim: "L2 (strip thinking tokens) improves accuracy by +6.6pp"
    vs L0 uncompressed. Underlying data is in compression_per_problem.jsonl
    for run 20260407_105657 (L0-L3 x 30 problems).
    """
    print("\n== Context compression (MBPP, Qwen 3.5 4B, 30 problems) ==")
    if not COMPRESSION_L0L3_DIR.exists():
        print("  SKIP: compression dir missing")
        return
    path = COMPRESSION_L0L3_DIR / "compression_per_problem.jsonl"
    rows = [json.loads(l) for l in path.open() if l.strip()]
    by_level: dict[int, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_level[r["compression_level"]][str(r["problem_id"])].append(
            bool(r["passed"])
        )
    by_level = {k: dict(v) for k, v in by_level.items()}

    # L2 vs L0 paired delta
    if 0 in by_level and 2 in by_level:
        add_diff_claim(
            macro="deltaCompressionLTwoMinusLZero",
            label="Compression L2 (strip thinking) - L0 (uncompressed)",
            source=f"data/results/compression_experiment/"
            f"{COMPRESSION_L0L3_DIR.name}/compression_per_problem.jsonl",
            paper_claim="stripping thinking tokens ... improves accuracy by 6.6pp",
            a_by_pid={k: [bool(x) for x in v] for k, v in by_level[2].items()},
            b_by_pid={k: [bool(x) for x in v] for k, v in by_level[0].items()},
        )
    # Also expose the individual L0, L2 pass-rate point estimates with CIs
    for level, macro, label in [
        (0, "passCompressionLZero", "Compression L0 Pass@1"),
        (2, "passCompressionLTwo", "Compression L2 Pass@1"),
    ]:
        if level not in by_level:
            continue
        add_pass_rate_claim(
            macro=macro,
            label=label,
            source=f"data/results/compression_experiment/"
            f"{COMPRESSION_L0L3_DIR.name}/compression_per_problem.jsonl",
            paper_claim=f"L{level} pass rate in Table 4 of paper.tex",
            rows_by_pid=by_level[level],
        )


def main() -> int:
    compute_mbpp_base_passes()
    compute_humaneval_passes()
    compute_tool_deltas()
    compute_early_exit_gains()
    compute_placement_claims()
    compute_compression_claims()
    write_tex()
    write_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
