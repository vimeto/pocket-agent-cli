"""Export functionality for benchmark results."""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from ..benchmarks import BenchmarkSession


def export_results(session: BenchmarkSession, output_path: Path) -> None:
    """Export benchmark results to a file.
    
    Args:
        session: Benchmark session to export
        output_path: Path to output file
    """
    # Convert session to dict
    data = session.to_dict()
    
    # Add metadata
    data["export_metadata"] = {
        "exported_at": datetime.now().isoformat(),
        "format_version": "1.0",
        "tool": "pocket-agent-cli",
    }
    
    # Save based on file extension
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    elif output_path.suffix == ".csv":
        import csv
        
        # Export problem results as CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "problem_id",
                "success",
                "duration_seconds",
                "ttft_ms",
                "tps",
                "tokens",
                "test_passed",
                "test_total",
            ])
            
            # Data rows
            for problem in session.problems:
                test_passed = sum(1 for tr in problem.test_results if tr.passed)
                test_total = len(problem.test_results)
                
                writer.writerow([
                    problem.problem_id,
                    problem.success,
                    (problem.end_time - problem.start_time).total_seconds(),
                    problem.metrics.get("ttft") if problem.metrics else None,
                    problem.metrics.get("tps") if problem.metrics else None,
                    problem.metrics.get("tokens") if problem.metrics else None,
                    test_passed,
                    test_total,
                ])
    
    elif output_path.suffix == ".md":
        # Export as Markdown report
        with open(output_path, "w") as f:
            f.write(f"# Benchmark Results\n\n")
            f.write(f"**Model:** {session.model_id}\n")
            f.write(f"**Mode:** {session.mode}\n")
            f.write(f"**Date:** {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration:** {(session.end_time - session.start_time).total_seconds():.1f}s\n\n")
            
            # Summary stats
            stats = session.aggregate_stats
            f.write("## Summary\n\n")
            f.write(f"- **Total Problems:** {stats['total_problems']}\n")
            f.write(f"- **Passed:** {stats['passed_problems']} ({stats['pass_rate']*100:.1f}%)\n")
            
            if "avg_ttft_ms" in stats:
                f.write(f"- **Average TTFT:** {stats['avg_ttft_ms']:.0f}ms\n")
            if "avg_tps" in stats:
                f.write(f"- **Average TPS:** {stats['avg_tps']:.1f}\n")
            
            # System metrics
            if "system_metrics" in stats and stats["system_metrics"]:
                f.write("\n## System Metrics\n\n")
                sys_metrics = stats["system_metrics"]
                
                if "cpu" in sys_metrics:
                    f.write(f"- **CPU Usage:** avg={sys_metrics['cpu']['avg_percent']:.1f}%, "
                           f"max={sys_metrics['cpu']['max_percent']:.1f}%\n")
                
                if "memory" in sys_metrics:
                    f.write(f"- **Memory Usage:** avg={sys_metrics['memory']['avg_used_mb']:.0f}MB, "
                           f"max={sys_metrics['memory']['max_used_mb']:.0f}MB\n")
                
                if "temperature" in sys_metrics:
                    f.write(f"- **Temperature:** avg={sys_metrics['temperature']['avg_celsius']:.1f}°C, "
                           f"max={sys_metrics['temperature']['max_celsius']:.1f}°C\n")
            
            # Problem details
            f.write("\n## Problem Results\n\n")
            for problem in session.problems:
                f.write(f"### Problem {problem.problem_id}\n\n")
                f.write(f"- **Success:** {'✓' if problem.success else '✗'}\n")
                f.write(f"- **Duration:** {(problem.end_time - problem.start_time).total_seconds():.2f}s\n")
                
                if problem.metrics:
                    if problem.metrics.get("ttft"):
                        f.write(f"- **TTFT:** {problem.metrics['ttft']:.0f}ms\n")
                    if problem.metrics.get("tps"):
                        f.write(f"- **TPS:** {problem.metrics['tps']:.1f}\n")
                
                # Test results
                f.write("\n**Test Results:**\n")
                for tr in problem.test_results:
                    status = "✓" if tr.passed else "✗"
                    f.write(f"- {status} `{tr.test_case}`\n")
                
                f.write("\n")
    
    else:
        # Default to JSON
        output_path = output_path.with_suffix(".json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load benchmark results from a file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Loaded results as dictionary
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, "r") as f:
        return json.load(f)


def compare_results(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two benchmark results.
    
    Args:
        results1: First results
        results2: Second results
        
    Returns:
        Comparison summary
    """
    stats1 = results1.get("aggregate_stats", {})
    stats2 = results2.get("aggregate_stats", {})
    
    comparison = {
        "model1": results1.get("model_id"),
        "model2": results2.get("model_id"),
        "mode": results1.get("mode"),
        "pass_rate_change": stats2.get("pass_rate", 0) - stats1.get("pass_rate", 0),
        "duration_change": stats2.get("total_duration_seconds", 0) - stats1.get("total_duration_seconds", 0),
    }
    
    if "avg_ttft_ms" in stats1 and "avg_ttft_ms" in stats2:
        comparison["ttft_change_ms"] = stats2["avg_ttft_ms"] - stats1["avg_ttft_ms"]
        comparison["ttft_change_percent"] = (
            (stats2["avg_ttft_ms"] - stats1["avg_ttft_ms"]) / stats1["avg_ttft_ms"] * 100
        )
    
    if "avg_tps" in stats1 and "avg_tps" in stats2:
        comparison["tps_change"] = stats2["avg_tps"] - stats1["avg_tps"]
        comparison["tps_change_percent"] = (
            (stats2["avg_tps"] - stats1["avg_tps"]) / stats1["avg_tps"] * 100
        )
    
    return comparison