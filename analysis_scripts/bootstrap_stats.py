"""Cluster-bootstrap confidence intervals.

MobiHoc 2026 sprint, Day 2 AM. This module exposes a single primitive
(`cluster_bootstrap_ci`) used to attach a 95% CI to every headline
percentage in the paper. Resampling is over clusters (typically
`problem_id`), not individual rows, so problem-level dependencies
(e.g. the same hard problem appearing across multiple rate estimates)
are preserved.

Design notes
------------
- Pure-numpy implementation. A 10k-resample pass over ~500 clusters
  runs in well under 2 seconds on a laptop.
- `stat_fn` receives the flattened list of values across resampled
  clusters and returns a single scalar. The default `pass_rate_ci`
  wraps this with `numpy.mean`, which is the right statistic for a
  Bernoulli pass array.
- The percentile CI is reported. Bias-corrected-and-accelerated
  (BCa) would be marginally tighter but adds complexity we do not
  need for the headline-claim pipeline.

This module is NOT intended to replace any existing estimator; it is
a reviewer-facing sanity-check layer on top of the paper's point
estimates. See the provenance note in `scripts/online_placement.py`
for what data sources are trustworthy for which claims.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

Stat = Callable[[np.ndarray], float]


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]],
    stat_fn: Stat = np.mean,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster-bootstrap (point, lo, hi) for the given statistic.

    Parameters
    ----------
    values_by_cluster
        Mapping from cluster key (typically a `problem_id`) to a list
        of per-row values for that cluster. Values can be bools, ints,
        or floats; they are coerced to float arrays internally.
    stat_fn
        Reduction from a flat 1-D numpy array of values to a scalar.
        Defaults to `np.mean` (which gives a pass-rate or mean).
    n_resamples
        Number of bootstrap replicates. 10k is the default and keeps
        the 2.5/97.5 percentile CI stable to ~0.1pp.
    seed
        RNG seed for reproducibility.
    alpha
        Two-sided miscoverage. Default 0.05 -> 95% CI.

    Returns
    -------
    (point, lo, hi)
        `point` is `stat_fn` applied to the full concatenated value
        array (i.e. the original statistic, not a bootstrap mean).
        `lo`/`hi` are the `alpha/2` and `1-alpha/2` percentiles of
        the bootstrap distribution.
    """
    if not values_by_cluster:
        raise ValueError("values_by_cluster is empty")

    cluster_keys = list(values_by_cluster.keys())
    cluster_arrays: list[np.ndarray] = [
        np.asarray(values_by_cluster[k], dtype=float) for k in cluster_keys
    ]
    n_clusters = len(cluster_keys)

    # Point estimate on the original sample.
    full = np.concatenate(cluster_arrays) if cluster_arrays else np.array([])
    point = float(stat_fn(full))

    # Resample whole clusters with replacement.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    # Precompute cluster lengths for fast concat via fancy indexing over
    # a flat backing buffer + a per-cluster slice table.
    cluster_lengths = np.asarray([a.size for a in cluster_arrays], dtype=np.int64)
    flat_values = np.concatenate(cluster_arrays) if cluster_arrays else np.array([])
    offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(cluster_lengths)

    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picks = indices[i]
        # Build the resampled flat array by concatenating selected cluster
        # slices. Avoids Python-level loops over rows.
        parts = [flat_values[offsets[p] : offsets[p + 1]] for p in picks]
        sample = np.concatenate(parts) if parts else np.array([])
        stats[i] = stat_fn(sample)

    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return point, lo, hi


def pass_rate_ci(
    rows_by_pid: dict[str, Iterable[bool]],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI for a pass rate over problems.

    Convenience wrapper around `cluster_bootstrap_ci` with `stat_fn`
    fixed to `np.mean`. Accepts lists of bools (or 0/1 ints).
    """
    as_floats: dict[str, list[float]] = {
        pid: [float(bool(v)) for v in vals] for pid, vals in rows_by_pid.items()
    }
    return cluster_bootstrap_ci(
        as_floats,
        stat_fn=np.mean,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )


def diff_ci(
    values_a_by_cluster: dict[str, list[float]],
    values_b_by_cluster: dict[str, list[float]],
    stat_fn: Stat = np.mean,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired cluster-bootstrap CI for stat(A) - stat(B).

    Clusters must be shared between A and B (keys intersected); each
    bootstrap resample draws cluster keys and applies the same draw to
    both A and B, preserving the within-problem pairing. This is the
    right CI for things like "tool mode gain vs base mode" where the
    two pass arrays are over the same problems.
    """
    shared = sorted(set(values_a_by_cluster) & set(values_b_by_cluster))
    if not shared:
        raise ValueError("no shared clusters between A and B")

    a_arrays = [np.asarray(values_a_by_cluster[k], dtype=float) for k in shared]
    b_arrays = [np.asarray(values_b_by_cluster[k], dtype=float) for k in shared]
    full_a = np.concatenate(a_arrays) if a_arrays else np.array([])
    full_b = np.concatenate(b_arrays) if b_arrays else np.array([])
    point = float(stat_fn(full_a) - stat_fn(full_b))

    n_clusters = len(shared)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    a_lengths = np.asarray([a.size for a in a_arrays], dtype=np.int64)
    b_lengths = np.asarray([b.size for b in b_arrays], dtype=np.int64)
    a_flat = np.concatenate(a_arrays) if a_arrays else np.array([])
    b_flat = np.concatenate(b_arrays) if b_arrays else np.array([])
    a_off = np.zeros(n_clusters + 1, dtype=np.int64)
    b_off = np.zeros(n_clusters + 1, dtype=np.int64)
    a_off[1:] = np.cumsum(a_lengths)
    b_off[1:] = np.cumsum(b_lengths)

    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picks = indices[i]
        a_parts = [a_flat[a_off[p] : a_off[p + 1]] for p in picks]
        b_parts = [b_flat[b_off[p] : b_off[p + 1]] for p in picks]
        sa = np.concatenate(a_parts) if a_parts else np.array([])
        sb = np.concatenate(b_parts) if b_parts else np.array([])
        stats[i] = stat_fn(sa) - stat_fn(sb)

    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return point, lo, hi


if __name__ == "__main__":
    # Smoke test: 500 "problems", each with one Bernoulli draw at p=0.7.
    rng = np.random.default_rng(42)
    synth = {
        f"p{i}": [bool(rng.random() < 0.7)] for i in range(500)
    }
    point, lo, hi = pass_rate_ci(synth, n_resamples=5000)
    print(f"smoke test: {point:.3f} [{lo:.3f}, {hi:.3f}] (expected ~0.70)")
