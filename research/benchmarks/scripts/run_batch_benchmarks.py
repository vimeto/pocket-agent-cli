#!/usr/bin/env python3
"""
Batch benchmark runner for systematic evaluation of gemma-3n-e2b-it model.
This script runs benchmarks in batches and handles failures gracefully.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import os

# Configuration
MODEL = "gemma-3n-e2b-it"
MODE = "full_tool"
SAMPLES_PER_PROBLEM = 1
BATCH_SIZE = 10  # Number of problems per batch
START_PROBLEM = 10
END_PROBLEM = 509

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "raw_results"
LOGS_DIR = SCRIPT_DIR.parent / "logs"
PROGRESS_FILE = RESULTS_DIR / f"batch_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def ensure_directories():
    """Ensure all required directories exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def run_batch(start_id, end_id, batch_num):
    """Run a batch of benchmarks."""
    print(f"\n{'='*60}")
    print(f"Running batch {batch_num}: problems {start_id} to {end_id}")
    print(f"{'='*60}")
    
    # Set timeout environment variables
    env = os.environ.copy()
    env["BASH_DEFAULT_TIMEOUT_MS"] = "600000"  # 10 minutes
    env["BASH_MAX_TIMEOUT_MS"] = "1200000"     # 20 minutes
    env["DEBUG_INFERENCE"] = "true"            # Enable debug to get detailed metrics
    
    # Run benchmark for each problem in the batch
    batch_results = []
    
    for problem_id in range(start_id, end_id + 1):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running problem {problem_id}...")
        
        start_time = time.time()
        
        try:
            # Run the benchmark command
            cmd = [
                "uv", "run", "pocket-agent", "benchmark",
                "--model", MODEL,
                "--problems", str(problem_id),
                "--num-samples", str(SAMPLES_PER_PROBLEM),
                "--mode", MODE
            ]
            
            # Create log file for this run
            log_file = LOGS_DIR / f"problem_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, "w") as log:
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                # Write full output to log
                log.write(f"Command: {' '.join(cmd)}\n")
                log.write(f"Exit code: {result.returncode}\n")
                log.write(f"\n--- STDOUT ---\n{result.stdout}")
                log.write(f"\n--- STDERR ---\n{result.stderr}")
            
            duration = time.time() - start_time
            
            # Extract metrics from output if available
            metrics = extract_metrics_from_output(result.stdout)
            
            batch_results.append({
                "problem_id": problem_id,
                "status": "success" if result.returncode == 0 else "failed",
                "duration_seconds": duration,
                "metrics": metrics,
                "log_file": str(log_file),
                "timestamp": datetime.now().isoformat()
            })
            
            if result.returncode == 0:
                print(f"✓ Problem {problem_id} completed in {duration:.1f}s")
                if metrics:
                    print(f"  TTFT: {metrics.get('ttft_ms', 'N/A')}ms, TPS: {metrics.get('tps', 'N/A')}")
            else:
                print(f"✗ Problem {problem_id} failed with exit code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"✗ Problem {problem_id} timed out after {duration:.1f}s")
            batch_results.append({
                "problem_id": problem_id,
                "status": "timeout",
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"✗ Problem {problem_id} error: {e}")
            batch_results.append({
                "problem_id": problem_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        # Small delay between problems
        time.sleep(2)
    
    return batch_results

def extract_metrics_from_output(output):
    """Extract metrics from benchmark output."""
    metrics = {}
    
    # Look for TTFT and TPS in output
    lines = output.split('\n')
    for line in lines:
        if "TTFT:" in line:
            try:
                ttft_part = line.split("TTFT:")[1].split("ms")[0].strip()
                metrics["ttft_ms"] = float(ttft_part)
            except:
                pass
        if "TPS:" in line:
            try:
                tps_part = line.split("TPS:")[1].split()[0].strip()
                metrics["tps"] = float(tps_part)
            except:
                pass
    
    return metrics

def save_progress(progress_data):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f, indent=2)

def main():
    """Main execution function."""
    ensure_directories()
    
    print(f"Starting batch benchmark run")
    print(f"Model: {MODEL}")
    print(f"Mode: {MODE}")
    print(f"Problems: {START_PROBLEM} to {END_PROBLEM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total problems: {END_PROBLEM - START_PROBLEM + 1}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Initialize progress tracking
    progress = {
        "config": {
            "model": MODEL,
            "mode": MODE,
            "start_problem": START_PROBLEM,
            "end_problem": END_PROBLEM,
            "batch_size": BATCH_SIZE,
            "samples_per_problem": SAMPLES_PER_PROBLEM
        },
        "start_time": datetime.now().isoformat(),
        "batches": []
    }
    
    # Run batches
    batch_num = 1
    for start_id in range(START_PROBLEM, END_PROBLEM + 1, BATCH_SIZE):
        end_id = min(start_id + BATCH_SIZE - 1, END_PROBLEM)
        
        batch_start = time.time()
        batch_results = run_batch(start_id, end_id, batch_num)
        batch_duration = time.time() - batch_start
        
        # Update progress
        batch_data = {
            "batch_num": batch_num,
            "start_id": start_id,
            "end_id": end_id,
            "duration_seconds": batch_duration,
            "results": batch_results,
            "timestamp": datetime.now().isoformat()
        }
        progress["batches"].append(batch_data)
        
        # Save progress after each batch
        save_progress(progress)
        
        # Summary for batch
        successful = sum(1 for r in batch_results if r["status"] == "success")
        failed = sum(1 for r in batch_results if r["status"] == "failed")
        timeouts = sum(1 for r in batch_results if r["status"] == "timeout")
        
        print(f"\nBatch {batch_num} summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Timeouts: {timeouts}")
        print(f"  Duration: {batch_duration:.1f}s")
        
        batch_num += 1
        
        # Longer delay between batches
        if end_id < END_PROBLEM:
            print("\nWaiting 10 seconds before next batch...")
            time.sleep(10)
    
    # Final summary
    progress["end_time"] = datetime.now().isoformat()
    save_progress(progress)
    
    total_successful = sum(
        sum(1 for r in b["results"] if r["status"] == "success")
        for b in progress["batches"]
    )
    total_problems = END_PROBLEM - START_PROBLEM + 1
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {total_problems}")
    print(f"Successful: {total_successful}")
    print(f"Success rate: {(total_successful/total_problems)*100:.1f}%")
    print(f"Progress saved to: {PROGRESS_FILE}")

if __name__ == "__main__":
    main()