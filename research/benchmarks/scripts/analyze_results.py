#!/usr/bin/env python3
"""
Analyze benchmark results and extract performance metrics.
"""

import json
import statistics as stats
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = Path.home() / ".pocket-agent-cli" / "results" / "benchmarks"
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"

def load_benchmark_results(model_name: str, mode: str) -> List[Dict[str, Any]]:
    """Load all benchmark results for a given model and mode."""
    results = []
    model_dir = RESULTS_DIR / model_name / mode
    
    if not model_dir.exists():
        print(f"No results found for {model_name} in {mode} mode")
        return results
    
    # Load all JSON files
    for json_file in model_dir.glob("bench_*.json"):
        if json_file.name.startswith("bench_") and not json_file.name.endswith("_runs.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results

def extract_metrics_from_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a benchmark session."""
    metrics = {
        "session_id": session.get("session_id"),
        "start_time": session.get("start_time"),
        "problems": []
    }
    
    for problem in session.get("problems", []):
        problem_metrics = {
            "problem_id": problem.get("problem_id"),
            "success": problem.get("success", False),
            "duration_seconds": problem.get("duration_seconds", 0),
            "context_length": problem.get("context_length_used", 0),
            "temperature": problem.get("temperature", 0.7),
            "cold_start": problem.get("cold_start", False),
            "test_results": len([t for t in problem.get("test_results", []) if t.get("passed", False)]),
            "total_tests": len(problem.get("test_results", []))
        }
        
        # Extract inference metrics if available
        if problem.get("metrics"):
            m = problem["metrics"]
            problem_metrics.update({
                "ttft_ms": m.get("ttft"),
                "tps": m.get("tps"),
                "total_tokens": m.get("tokens"),
                "energy_joules": m.get("energy_summary", {}).get("total_energy_joules"),
                "energy_per_token": m.get("energy_per_token_joules"),
                "avg_power_watts": m.get("energy_summary", {}).get("avg_power_watts"),
                "cpu_avg_percent": m.get("energy_summary", {}).get("cpu_avg_percent"),
                "gpu_avg_percent": m.get("energy_summary", {}).get("gpu_utilization_avg_percent"),
                "iteration_count": m.get("iteration_count", 1),
                "submission_via_tool": m.get("submission_via_tool", False)
            })
        
        # Calculate prefilling latency if possible
        if problem_metrics.get("ttft_ms") and problem_metrics.get("context_length"):
            # Prefilling rate = prompt tokens / TTFT
            problem_metrics["prefill_tokens_per_second"] = (
                problem_metrics["context_length"] / (problem_metrics["ttft_ms"] / 1000)
            )
        
        metrics["problems"].append(problem_metrics)
    
    # Add system metrics if available
    if session.get("system_metrics"):
        system_stats = analyze_system_metrics(session["system_metrics"])
        metrics["system_stats"] = system_stats
    
    return metrics

def analyze_system_metrics(system_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze system-level metrics."""
    if not system_metrics:
        return {}
    
    # Extract values
    power_values = [m["power_watts"] for m in system_metrics if "power_watts" in m]
    cpu_values = [m["cpu_percent"] for m in system_metrics if "cpu_percent" in m]
    gpu_values = [m["gpu_utilization_percent"] for m in system_metrics if "gpu_utilization_percent" in m]
    memory_values = [m["memory_percent"] for m in system_metrics if "memory_percent" in m]
    
    stats = {}
    
    if power_values:
        stats["power"] = {
            "avg": stats.mean(power_values),
            "min": min(power_values),
            "max": max(power_values),
            "std": stats.stdev(power_values) if len(power_values) > 1 else 0
        }
    
    if cpu_values:
        stats["cpu"] = {
            "avg": stats.mean(cpu_values),
            "min": min(cpu_values),
            "max": max(cpu_values),
            "std": stats.stdev(cpu_values) if len(cpu_values) > 1 else 0
        }
    
    if gpu_values:
        stats["gpu"] = {
            "avg": stats.mean(gpu_values),
            "min": min(gpu_values),
            "max": max(gpu_values),
            "std": stats.stdev(gpu_values) if len(gpu_values) > 1 else 0
        }
    
    if memory_values:
        stats["memory"] = {
            "avg": stats.mean(memory_values),
            "min": min(memory_values),
            "max": max(memory_values)
        }
    
    return stats

def create_summary_report(all_metrics: List[Dict[str, Any]], model_name: str, mode: str) -> Dict[str, Any]:
    """Create a comprehensive summary report."""
    # Flatten all problem metrics
    all_problems = []
    for session in all_metrics:
        all_problems.extend(session["problems"])
    
    if not all_problems:
        return {"error": "No problems found in results"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_problems)
    
    report = {
        "model": model_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "total_problems": len(df),
        "unique_problems": df["problem_id"].nunique(),
        "success_rate": (df["success"].sum() / len(df)) * 100 if len(df) > 0 else 0,
    }
    
    # Performance metrics
    perf_metrics = {}
    
    # TTFT statistics
    ttft_data = df["ttft_ms"].dropna()
    if not ttft_data.empty:
        perf_metrics["ttft_ms"] = {
            "mean": ttft_data.mean(),
            "median": ttft_data.median(),
            "min": ttft_data.min(),
            "max": ttft_data.max(),
            "std": ttft_data.std(),
            "p95": ttft_data.quantile(0.95),
            "p99": ttft_data.quantile(0.99)
        }
    
    # TPS statistics
    tps_data = df["tps"].dropna()
    if not tps_data.empty:
        perf_metrics["tps"] = {
            "mean": tps_data.mean(),
            "median": tps_data.median(),
            "min": tps_data.min(),
            "max": tps_data.max(),
            "std": tps_data.std(),
            "p5": tps_data.quantile(0.05),  # 5th percentile (worst)
            "p95": tps_data.quantile(0.95)
        }
    
    # Prefilling performance
    prefill_data = df["prefill_tokens_per_second"].dropna()
    if not prefill_data.empty:
        perf_metrics["prefill_tokens_per_second"] = {
            "mean": prefill_data.mean(),
            "median": prefill_data.median(),
            "min": prefill_data.min(),
            "max": prefill_data.max(),
            "std": prefill_data.std()
        }
    
    # Energy metrics
    energy_data = df["energy_joules"].dropna()
    if not energy_data.empty:
        perf_metrics["energy_joules"] = {
            "mean": energy_data.mean(),
            "median": energy_data.median(),
            "total": energy_data.sum(),
            "per_problem": energy_data.mean()
        }
    
    energy_per_token = df["energy_per_token"].dropna()
    if not energy_per_token.empty:
        perf_metrics["energy_per_token_joules"] = {
            "mean": energy_per_token.mean(),
            "median": energy_per_token.median(),
            "min": energy_per_token.min(),
            "max": energy_per_token.max()
        }
    
    # Power consumption
    power_data = df["avg_power_watts"].dropna()
    if not power_data.empty:
        perf_metrics["avg_power_watts"] = {
            "mean": power_data.mean(),
            "median": power_data.median(),
            "min": power_data.min(),
            "max": power_data.max()
        }
    
    # Resource utilization
    cpu_data = df["cpu_avg_percent"].dropna()
    if not cpu_data.empty:
        perf_metrics["cpu_utilization_percent"] = {
            "mean": cpu_data.mean(),
            "median": cpu_data.median(),
            "max": cpu_data.max()
        }
    
    gpu_data = df["gpu_avg_percent"].dropna()
    if not gpu_data.empty:
        perf_metrics["gpu_utilization_percent"] = {
            "mean": gpu_data.mean(),
            "median": gpu_data.median(),
            "max": gpu_data.max()
        }
    
    report["performance_metrics"] = perf_metrics
    
    # Context length analysis
    context_data = df["context_length"].dropna()
    if not context_data.empty:
        report["context_length_stats"] = {
            "mean": context_data.mean(),
            "median": context_data.median(),
            "min": context_data.min(),
            "max": context_data.max(),
            "std": context_data.std()
        }
    
    # For full_tool mode, add iteration statistics
    if mode == "full_tool":
        iteration_data = df["iteration_count"].dropna()
        if not iteration_data.empty:
            report["iteration_stats"] = {
                "mean": iteration_data.mean(),
                "median": iteration_data.median(),
                "min": iteration_data.min(),
                "max": iteration_data.max(),
                "submission_via_tool_rate": (df["submission_via_tool"].sum() / len(df)) * 100
            }
    
    # Problem difficulty analysis (based on success rate per problem)
    problem_success = df.groupby("problem_id")["success"].agg(["mean", "count"])
    hardest_problems = problem_success.nsmallest(10, "mean").index.tolist()
    easiest_problems = problem_success.nlargest(10, "mean").index.tolist()
    
    report["problem_analysis"] = {
        "hardest_problems": hardest_problems,
        "easiest_problems": easiest_problems,
        "problems_with_zero_success": problem_success[problem_success["mean"] == 0].index.tolist()
    }
    
    return report

def save_analysis_results(report: Dict[str, Any], model_name: str, mode: str):
    """Save analysis results to file."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{model_name}_{mode}_{timestamp}.json"
    filepath = ANALYSIS_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis saved to: {filepath}")
    
    # Also save a CSV for easy viewing
    csv_filename = f"metrics_{model_name}_{mode}_{timestamp}.csv"
    csv_filepath = ANALYSIS_DIR / csv_filename
    
    # Extract key metrics to CSV
    if "performance_metrics" in report:
        metrics_data = []
        for metric_type, values in report["performance_metrics"].items():
            for stat_name, stat_value in values.items():
                metrics_data.append({
                    "metric": f"{metric_type}_{stat_name}",
                    "value": stat_value
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(csv_filepath, index=False)
        print(f"Metrics CSV saved to: {csv_filepath}")

def main():
    """Main analysis function."""
    model_name = "gemma-3n-e2b-it"
    mode = "full_tool"
    
    print(f"Analyzing results for {model_name} in {mode} mode...")
    
    # Load all benchmark results
    sessions = load_benchmark_results(model_name, mode)
    print(f"Found {len(sessions)} benchmark sessions")
    
    if not sessions:
        print("No results to analyze")
        return
    
    # Extract metrics from each session
    all_metrics = []
    for session in sessions:
        metrics = extract_metrics_from_session(session)
        all_metrics.append(metrics)
    
    # Create summary report
    report = create_summary_report(all_metrics, model_name, mode)
    
    # Print key findings
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total problems evaluated: {report['total_problems']}")
    print(f"Success rate: {report['success_rate']:.1f}%")
    
    if "performance_metrics" in report:
        perf = report["performance_metrics"]
        
        if "ttft_ms" in perf:
            print(f"\nTime to First Token (TTFT):")
            print(f"  Mean: {perf['ttft_ms']['mean']:.1f} ms")
            print(f"  Median: {perf['ttft_ms']['median']:.1f} ms")
            print(f"  95th percentile: {perf['ttft_ms']['p95']:.1f} ms")
        
        if "tps" in perf:
            print(f"\nTokens Per Second (TPS):")
            print(f"  Mean: {perf['tps']['mean']:.1f}")
            print(f"  Median: {perf['tps']['median']:.1f}")
            print(f"  5th percentile: {perf['tps']['p5']:.1f}")
        
        if "prefill_tokens_per_second" in perf:
            print(f"\nPrefilling Performance:")
            print(f"  Mean: {perf['prefill_tokens_per_second']['mean']:.1f} tokens/s")
            print(f"  Median: {perf['prefill_tokens_per_second']['median']:.1f} tokens/s")
        
        if "avg_power_watts" in perf:
            print(f"\nPower Consumption:")
            print(f"  Mean: {perf['avg_power_watts']['mean']:.1f} W")
            print(f"  Max: {perf['avg_power_watts']['max']:.1f} W")
        
        if "gpu_utilization_percent" in perf:
            print(f"\nGPU Utilization:")
            print(f"  Mean: {perf['gpu_utilization_percent']['mean']:.1f}%")
            print(f"  Max: {perf['gpu_utilization_percent']['max']:.1f}%")
    
    # Save results
    save_analysis_results(report, model_name, mode)

if __name__ == "__main__":
    main()