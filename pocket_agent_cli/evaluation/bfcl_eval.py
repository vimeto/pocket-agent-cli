"""AST-based evaluation for BFCL tool-calling benchmark.

Compares model-produced function calls against expected calls using:
- Function name exact match
- Argument name set comparison
- Argument value comparison with type coercion and tolerance
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------

def _coerce_numeric(value: Any) -> Optional[float]:
    """Try to interpret *value* as a number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None


def values_match(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare two argument values with type coercion.

    - Numeric values: compared with ``tolerance``.
    - Strings: case-insensitive substring check (expected in actual or
      actual in expected) to handle minor phrasing differences.
    - Lists: element-wise comparison (order-insensitive for short lists).
    - Dicts: recursive key-value comparison.
    """
    # Numeric comparison
    num_a = _coerce_numeric(actual)
    num_e = _coerce_numeric(expected)
    if num_a is not None and num_e is not None:
        return math.isclose(num_a, num_e, rel_tol=tolerance, abs_tol=tolerance)

    # String comparison — normalised
    if isinstance(actual, str) and isinstance(expected, str):
        a_lower = actual.strip().lower()
        e_lower = expected.strip().lower()
        if a_lower == e_lower:
            return True
        # Allow substring containment for flexible matching
        if e_lower in a_lower or a_lower in e_lower:
            return True
        return False

    # List comparison (order-insensitive for short lists)
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        # Try order-sensitive first
        if all(values_match(a, e, tolerance) for a, e in zip(actual, expected)):
            return True
        # Try order-insensitive for short lists
        if len(expected) <= 10:
            used = set()
            for e in expected:
                found = False
                for i, a in enumerate(actual):
                    if i not in used and values_match(a, e, tolerance):
                        used.add(i)
                        found = True
                        break
                if not found:
                    return False
            return True
        return False

    # Dict comparison
    if isinstance(actual, dict) and isinstance(expected, dict):
        for k, v in expected.items():
            if k not in actual:
                return False
            if not values_match(actual[k], v, tolerance):
                return False
        return True

    # Fallback: string representation
    return str(actual).strip().lower() == str(expected).strip().lower()


# ---------------------------------------------------------------------------
# Single-call evaluation
# ---------------------------------------------------------------------------

def evaluate_single_call(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one actual call against one expected call.

    Returns a dict with:
        - name_match: bool
        - args_match: bool  (all argument values correct)
        - arg_names_match: bool  (required argument names present)
        - full_match: bool  (name + all args correct)
        - partial_match: bool  (name correct, some/all args wrong)
        - details: dict with per-argument results
    """
    act_name = actual.get("name", "")
    exp_name = expected.get("name", "")

    name_match = act_name == exp_name

    act_args = actual.get("arguments", actual.get("parameters", {}))
    exp_args = expected.get("arguments", expected.get("parameters", {}))

    # Ensure dicts
    if isinstance(act_args, str):
        try:
            act_args = json.loads(act_args)
        except (json.JSONDecodeError, TypeError):
            act_args = {}
    if isinstance(exp_args, str):
        try:
            exp_args = json.loads(exp_args)
        except (json.JSONDecodeError, TypeError):
            exp_args = {}
    if not isinstance(act_args, dict):
        act_args = {}
    if not isinstance(exp_args, dict):
        exp_args = {}

    # Argument name comparison
    exp_keys = set(exp_args.keys())
    act_keys = set(act_args.keys())
    arg_names_match = exp_keys.issubset(act_keys)

    # Per-argument value comparison (only check expected keys)
    arg_details = {}
    all_args_ok = True
    for key in exp_keys:
        if key not in act_args:
            arg_details[key] = {"match": False, "reason": "missing"}
            all_args_ok = False
        else:
            match = values_match(act_args[key], exp_args[key])
            arg_details[key] = {
                "match": match,
                "actual": act_args[key],
                "expected": exp_args[key],
            }
            if not match:
                all_args_ok = False

    # If expected has no args, matching the name is enough
    if not exp_args:
        all_args_ok = True

    full_match = name_match and all_args_ok
    partial_match = name_match and not all_args_ok

    return {
        "name_match": name_match,
        "args_match": all_args_ok,
        "arg_names_match": arg_names_match,
        "full_match": full_match,
        "partial_match": partial_match,
        "details": arg_details,
    }


# ---------------------------------------------------------------------------
# Example-level evaluation
# ---------------------------------------------------------------------------

def evaluate_example(
    actual_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]],
    category: str = "simple",
) -> Dict[str, Any]:
    """Evaluate all calls for one BFCL example.

    Scoring:
        - full_match: all expected calls matched (name + args)
        - partial_match: at least one name matched but args wrong
        - no_match: no expected call matched at all

    For *relevance* category (expected=[]):
        - full_match = model produced NO tool calls
        - no_match = model incorrectly called a function

    Returns dict with score fields and per-call details.
    """
    # Relevance: expect zero calls
    if category == "relevance" or not expected_calls:
        if not actual_calls:
            return {
                "score": "full_match",
                "full_match": True,
                "partial_match": False,
                "no_match": False,
                "expected_count": 0,
                "actual_count": 0,
                "call_results": [],
            }
        else:
            return {
                "score": "no_match",
                "full_match": False,
                "partial_match": False,
                "no_match": True,
                "expected_count": 0,
                "actual_count": len(actual_calls),
                "call_results": [],
            }

    if not actual_calls:
        return {
            "score": "no_match",
            "full_match": False,
            "partial_match": False,
            "no_match": True,
            "expected_count": len(expected_calls),
            "actual_count": 0,
            "call_results": [],
        }

    # Match each expected call to the best actual call (greedy)
    call_results = []
    used_actual = set()
    all_full = True
    any_name = False

    for exp in expected_calls:
        best_result = None
        best_idx = -1

        for i, act in enumerate(actual_calls):
            if i in used_actual:
                continue
            result = evaluate_single_call(act, exp)
            if best_result is None:
                best_result = result
                best_idx = i
            elif result["full_match"] and not best_result["full_match"]:
                best_result = result
                best_idx = i
            elif result["name_match"] and not best_result["name_match"]:
                best_result = result
                best_idx = i

        if best_result is None:
            # No more actual calls to compare
            best_result = {
                "name_match": False,
                "args_match": False,
                "arg_names_match": False,
                "full_match": False,
                "partial_match": False,
                "details": {},
            }

        if best_idx >= 0:
            used_actual.add(best_idx)

        call_results.append(best_result)
        if not best_result["full_match"]:
            all_full = False
        if best_result["name_match"]:
            any_name = True

    if all_full:
        score = "full_match"
    elif any_name:
        score = "partial_match"
    else:
        score = "no_match"

    return {
        "score": score,
        "full_match": all_full,
        "partial_match": any_name and not all_full,
        "no_match": not any_name,
        "expected_count": len(expected_calls),
        "actual_count": len(actual_calls),
        "call_results": call_results,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of per-example results.

    Each result dict must have at least:
        - score: "full_match" | "partial_match" | "no_match"
        - category: str

    Returns overall and per-category percentages.
    """
    total = len(results)
    if total == 0:
        return {"total": 0}

    full = sum(1 for r in results if r.get("score") == "full_match")
    partial = sum(1 for r in results if r.get("score") == "partial_match")
    no = sum(1 for r in results if r.get("score") == "no_match")

    summary = {
        "total": total,
        "full_match": full,
        "partial_match": partial,
        "no_match": no,
        "full_match_pct": round(full / total * 100, 1),
        "partial_match_pct": round(partial / total * 100, 1),
        "no_match_pct": round(no / total * 100, 1),
    }

    # Per category
    categories: Dict[str, List] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    per_category = {}
    for cat, cat_results in categories.items():
        n = len(cat_results)
        f = sum(1 for r in cat_results if r.get("score") == "full_match")
        per_category[cat] = {
            "total": n,
            "full_match": f,
            "full_match_pct": round(f / n * 100, 1) if n else 0,
        }

    summary["per_category"] = per_category
    return summary
