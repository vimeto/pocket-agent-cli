#!/usr/bin/env python3
"""
Process benchmark data and extract metrics for analysis.
Handles multiple runs, pass@k calculations, and system metrics aggregation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import csv
from collections import defaultdict
import statistics as stats

class BenchmarkDataProcessor:
    def __init__(self, processed_dir: Path, output_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrames for different metrics
        self.problem_results = []
        self.system_metrics = []
        self.pass_at_k_results = defaultdict(list)
        self.aggregated_stats = {}
    
    def load_job_data(self, job_dir: Path) -> Dict[str, Any]:
        """Load all data for a single job."""
        job_data = {
            'metadata': None,
            'sessions': [],
            'gpu_metrics': None,
            'cpu_metrics': None,
            'failures': []
        }
        
        # Load metadata
        metadata_file = job_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                job_data['metadata'] = json.load(f)
        
        # Load benchmark sessions
        results_dir = job_dir / 'benchmark_results'
        if results_dir.exists():
            for session_dir in results_dir.iterdir():
                if session_dir.is_dir():
                    # Look for nested structure: session_dir/model_name/mode/*.json
                    for model_dir in session_dir.glob('*/'):
                        if model_dir.is_dir():
                            for mode_dir in model_dir.glob('*/'):
                                if mode_dir.is_dir():
                                    # Load main session files
                                    for json_file in mode_dir.glob('*.json'):
                                        if not json_file.name.startswith('benchmark_summary'):
                                            with open(json_file, 'r') as f:
                                                session = json.load(f)
                                                job_data['sessions'].append(session)
                                    
                                    # Load individual runs if available
                                    runs_dir = mode_dir / 'runs'
                                    if runs_dir.exists():
                                        for run_dir in runs_dir.glob('*/'):
                                            if run_dir.is_dir():
                                                for run_file in run_dir.glob('*.json'):
                                                    with open(run_file, 'r') as f:
                                                        run_data = json.load(f)
                                                        # Add to sessions with run marker
                                                        run_data['is_individual_run'] = True
                                                        job_data['sessions'].append(run_data)
        
        # Load system metrics
        gpu_file = job_dir / 'system_metrics' / 'gpu_monitor.csv'
        if gpu_file.exists():
            job_data['gpu_metrics'] = pd.read_csv(gpu_file)
        
        cpu_file = job_dir / 'system_metrics' / 'cpu_monitor.csv'
        if cpu_file.exists():
            # Handle custom CSV format from monitoring script
            cpu_data = []
            with open(cpu_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        cpu_data.append({
                            'timestamp': int(parts[0]),
                            'cpu_percent': float(parts[1].replace('%', '')),
                            'memory_percent': float(parts[2])
                        })
            if cpu_data:
                job_data['cpu_metrics'] = pd.DataFrame(cpu_data)
        
        # Load failures log
        failures_file = job_dir / 'logs' / 'failures.log'
        if failures_file.exists():
            with open(failures_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        job_data['failures'].append({
                            'timestamp': parts[0],
                            'mode': parts[1],
                            'problems': parts[2],
                            'status': parts[3]
                        })
        
        return job_data
    
    def extract_problem_metrics(self, session: Dict[str, Any], job_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract metrics from individual problems in a session."""
        metrics_list = []
        
        # Check if this is a full session or individual run
        if session.get('is_individual_run'):
            # Individual run - treat as single problem
            problem = session
            metrics = self._extract_single_problem_metrics(problem, session, job_metadata)
            metrics_list.append(metrics)
        else:
            # Full session with multiple problems
            for problem in session.get('problems', []):
                metrics = self._extract_single_problem_metrics(problem, session, job_metadata)
                metrics_list.append(metrics)
        
        return metrics_list
    
    def _extract_single_problem_metrics(self, problem: Dict[str, Any], 
                                       session: Dict[str, Any], 
                                       job_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from a single problem."""
        metrics = {
            # Identifiers
            'job_id': job_metadata.get('job_id'),
            'session_id': session.get('session_id', 'unknown'),
            'problem_id': problem.get('problem_id'),
            'run_id': problem.get('run_id', 0),
            
            # Model configuration
            'model': session.get('model_id', job_metadata.get('model')),
            'mode': session.get('mode', job_metadata.get('mode')),
            'temperature': problem.get('temperature', 0.7),
            
            # Performance metrics
            'success': problem.get('success', False),
            'duration_seconds': problem.get('duration_seconds', 0),
            'context_length': problem.get('context_length_used', 0),
            'cold_start': problem.get('cold_start', False),
            
            # Test results
            'test_results': problem.get('test_results', []),
            'tests_passed': 0,
            'tests_total': 0,
            'test_pass_rate': 0.0,
        }
        
        # Calculate test statistics
        if metrics['test_results']:
            metrics['tests_total'] = len(metrics['test_results'])
            metrics['tests_passed'] = sum(1 for t in metrics['test_results'] if t.get('passed', False))
            metrics['test_pass_rate'] = metrics['tests_passed'] / metrics['tests_total'] if metrics['tests_total'] > 0 else 0
        
        # Extract detailed inference metrics
        if problem.get('metrics'):
            m = problem['metrics']
            metrics.update({
                'iteration_count': m.get('iteration_count', 1),
                'submission_via_tool': m.get('submission_via_tool', False),
                
                # Energy metrics
                'energy_joules': m.get('energy_summary', {}).get('total_energy_joules'),
                'energy_per_token': m.get('energy_per_token_joules'),
                'avg_power_watts': m.get('energy_summary', {}).get('avg_power_watts'),
                'cpu_avg_percent': m.get('energy_summary', {}).get('cpu_avg_percent'),
                'gpu_avg_percent': m.get('energy_summary', {}).get('gpu_utilization_avg_percent'),
                'memory_avg_percent': m.get('energy_summary', {}).get('memory_avg_percent'),
            })
        
        # Extract inter-token latencies and compute metrics
        if problem.get('inter_token_latencies'):
            latencies = problem['inter_token_latencies']
            if latencies and len(latencies) > 0:
                # TTFT is the first latency
                metrics['ttft_ms'] = latencies[0] if latencies else None
                
                # Count tokens (length of latency array)
                metrics['total_tokens'] = len(latencies)
                
                # Calculate TPS from latencies (excluding outliers for tool calls)
                # Tool calls are typically > 100ms
                normal_latencies = [l for l in latencies[1:] if l < 100]  # Skip TTFT and tool calls
                if normal_latencies:
                    avg_latency_ms = np.mean(normal_latencies)
                    metrics['tps'] = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else None
                
                # Overall TPS including all tokens
                if metrics['duration_seconds'] and metrics['total_tokens']:
                    metrics['overall_tps'] = metrics['total_tokens'] / metrics['duration_seconds']
                
                # Inter-token latency statistics
                metrics['itl_mean_ms'] = np.mean(latencies)
                metrics['itl_median_ms'] = np.median(latencies)
                metrics['itl_p95_ms'] = np.percentile(latencies, 95)
                metrics['itl_p99_ms'] = np.percentile(latencies, 99)
                
                # Tool call detection - latencies > 500ms are likely tool calls
                tool_latencies = [l for l in latencies if l > 500]
                metrics['num_tool_calls'] = len(tool_latencies)
                metrics['avg_tool_call_ms'] = np.mean(tool_latencies) if tool_latencies else None
                metrics['total_tool_time_ms'] = sum(tool_latencies) if tool_latencies else 0
        
        # Also check for explicit tool_calls field
        if problem.get('tool_calls'):
            metrics['explicit_tool_calls'] = len(problem['tool_calls'])
        
        # Get response length if available
        if problem.get('response'):
            metrics['response_length'] = len(problem['response'])
        
        return metrics
    
    def calculate_pass_at_k(self, problem_results: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Calculate pass@k rates for different k values."""
        pass_at_k = {}
        k_values = [1, 3, 5, 10]
        
        # Group by problem_id, model, and mode
        grouped = problem_results.groupby(['problem_id', 'model', 'mode'])
        
        for k in k_values:
            pass_at_k[k] = {
                'total_problems': 0,
                'passed_problems': 0,
                'pass_rate': 0.0,
                'by_model': defaultdict(lambda: {'total': 0, 'passed': 0}),
                'by_mode': defaultdict(lambda: {'total': 0, 'passed': 0}),
            }
            
            for (problem_id, model, mode), group in grouped:
                # Get up to k samples for this problem
                samples = group.head(k)
                
                # Check if any sample passed all tests
                problem_passed = any(samples['test_pass_rate'] == 1.0)
                
                pass_at_k[k]['total_problems'] += 1
                if problem_passed:
                    pass_at_k[k]['passed_problems'] += 1
                
                # Update model stats
                model_key = f"{model}_{mode}"
                pass_at_k[k]['by_model'][model_key]['total'] += 1
                if problem_passed:
                    pass_at_k[k]['by_model'][model_key]['passed'] += 1
                
                # Update mode stats
                pass_at_k[k]['by_mode'][mode]['total'] += 1
                if problem_passed:
                    pass_at_k[k]['by_mode'][mode]['passed'] += 1
            
            # Calculate overall pass rate
            if pass_at_k[k]['total_problems'] > 0:
                pass_at_k[k]['pass_rate'] = (
                    pass_at_k[k]['passed_problems'] / pass_at_k[k]['total_problems']
                )
            
            # Calculate pass rates for each model and mode
            for model_key in pass_at_k[k]['by_model']:
                stats = pass_at_k[k]['by_model'][model_key]
                if stats['total'] > 0:
                    stats['pass_rate'] = stats['passed'] / stats['total']
            
            for mode in pass_at_k[k]['by_mode']:
                stats = pass_at_k[k]['by_mode'][mode]
                if stats['total'] > 0:
                    stats['pass_rate'] = stats['passed'] / stats['total']
        
        return pass_at_k
    
    def aggregate_system_metrics(self, gpu_df: pd.DataFrame, cpu_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate system metrics for a job."""
        metrics = {}
        
        if gpu_df is not None and not gpu_df.empty:
            # Parse GPU metrics (assuming NVIDIA SMI CSV format)
            # Column names may have leading/trailing spaces
            gpu_df.columns = gpu_df.columns.str.strip()
            
            if 'utilization.gpu [%]' in gpu_df.columns:
                gpu_util = gpu_df['utilization.gpu [%]'].str.replace(' %', '').str.strip()
                gpu_util = pd.to_numeric(gpu_util, errors='coerce')
                gpu_util = gpu_util.dropna()
                if not gpu_util.empty:
                    metrics['gpu_util_mean'] = gpu_util.mean()
                    metrics['gpu_util_max'] = gpu_util.max()
                    metrics['gpu_util_p95'] = gpu_util.quantile(0.95)
            
            if 'memory.used [MiB]' in gpu_df.columns:
                mem_used = gpu_df['memory.used [MiB]'].str.replace(' MiB', '').str.strip()
                mem_used = pd.to_numeric(mem_used, errors='coerce')
                mem_used = mem_used.dropna()
                if not mem_used.empty:
                    metrics['gpu_memory_mean_mb'] = mem_used.mean()
                    metrics['gpu_memory_max_mb'] = mem_used.max()
            
            if 'power.draw [W]' in gpu_df.columns:
                power = gpu_df['power.draw [W]'].str.replace(' W', '').str.strip()
                power = pd.to_numeric(power, errors='coerce')
                power = power.dropna()
                if not power.empty:
                    metrics['gpu_power_mean_w'] = power.mean()
                    metrics['gpu_power_max_w'] = power.max()
                    
                    # Calculate total energy consumption
                    if len(power) > 1:
                        # Assuming 5-second intervals
                        metrics['gpu_energy_total_j'] = power.sum() * 5
        
        if cpu_df is not None and not cpu_df.empty:
            metrics['cpu_util_mean'] = cpu_df['cpu_percent'].mean()
            metrics['cpu_util_max'] = cpu_df['cpu_percent'].max()
            metrics['cpu_util_p95'] = cpu_df['cpu_percent'].quantile(0.95)
            
            metrics['memory_util_mean'] = cpu_df['memory_percent'].mean()
            metrics['memory_util_max'] = cpu_df['memory_percent'].max()
        
        return metrics
    
    def process_all_jobs(self):
        """Process all jobs in the processed directory."""
        print("Processing benchmark data...")
        
        # Load master index
        master_index_file = self.processed_dir / 'master_index.json'
        if not master_index_file.exists():
            print("Error: master_index.json not found. Run organize_benchmark_data.py first.")
            return
        
        with open(master_index_file, 'r') as f:
            master_index = json.load(f)
        
        all_problem_metrics = []
        all_system_metrics = []
        
        # Process each job
        for job_info in master_index['jobs']:
            job_id = job_info['job_id']
            job_dir = self.processed_dir / f"job_{job_id}"
            
            if not job_dir.exists():
                continue
            
            print(f"\nProcessing job {job_id}...")
            
            # Load job data
            job_data = self.load_job_data(job_dir)
            
            # Extract problem metrics
            for session in job_data['sessions']:
                problem_metrics = self.extract_problem_metrics(session, job_data['metadata'])
                all_problem_metrics.extend(problem_metrics)
            
            # Aggregate system metrics
            if job_data['gpu_metrics'] is not None or job_data['cpu_metrics'] is not None:
                system_stats = self.aggregate_system_metrics(
                    job_data['gpu_metrics'], 
                    job_data['cpu_metrics']
                )
                system_stats['job_id'] = job_id
                system_stats.update({
                    'model': job_info.get('model'),
                    'model_version': job_info.get('model_version'),
                    'mode': job_info.get('mode'),
                    'total_problems': job_info.get('total_problems'),
                    'successful_problems': job_info.get('successful_problems'),
                })
                all_system_metrics.append(system_stats)
        
        # Convert to DataFrames
        self.problem_results = pd.DataFrame(all_problem_metrics)
        self.system_metrics = pd.DataFrame(all_system_metrics)
        
        print(f"\nProcessed {len(self.problem_results)} problem results")
        print(f"Processed {len(self.system_metrics)} system metric records")
    
    def calculate_aggregate_statistics(self):
        """Calculate aggregate statistics across all results."""
        if self.problem_results.empty:
            print("No problem results to aggregate")
            return
        
        print("\nCalculating aggregate statistics...")
        
        # Overall statistics
        self.aggregated_stats['overall'] = {
            'total_problems': len(self.problem_results['problem_id'].unique()),
            'total_runs': len(self.problem_results),
            'overall_success_rate': self.problem_results['success'].mean(),
            'overall_test_pass_rate': self.problem_results['test_pass_rate'].mean(),
        }
        
        # Performance statistics
        perf_cols = ['duration_seconds', 'ttft_ms', 'tps', 'total_tokens', 
                     'energy_joules', 'energy_per_token']
        for col in perf_cols:
            if col in self.problem_results.columns:
                valid_data = self.problem_results[col].dropna()
                if not valid_data.empty:
                    self.aggregated_stats['overall'][f'{col}_mean'] = valid_data.mean()
                    self.aggregated_stats['overall'][f'{col}_median'] = valid_data.median()
                    self.aggregated_stats['overall'][f'{col}_p95'] = valid_data.quantile(0.95)
        
        # Statistics by model and mode
        self.aggregated_stats['by_model'] = {}
        for (model, mode), group in self.problem_results.groupby(['model', 'mode']):
            key = f"{model}_{mode}"
            self.aggregated_stats['by_model'][key] = {
                'total_runs': len(group),
                'success_rate': group['success'].mean(),
                'test_pass_rate': group['test_pass_rate'].mean(),
                'duration_mean': group['duration_seconds'].mean(),
                'duration_median': group['duration_seconds'].median(),
            }
            
            # Add performance metrics if available
            if 'ttft_ms' in group.columns:
                self.aggregated_stats['by_model'][key]['ttft_mean'] = group['ttft_ms'].dropna().mean()
            if 'tps' in group.columns:
                self.aggregated_stats['by_model'][key]['tps_mean'] = group['tps'].dropna().mean()
            if 'energy_per_token' in group.columns:
                self.aggregated_stats['by_model'][key]['energy_per_token_mean'] = group['energy_per_token'].dropna().mean()
        
        # Calculate pass@k rates
        self.pass_at_k_results = self.calculate_pass_at_k(self.problem_results)
        
        print("Aggregate statistics calculated")
    
    def save_processed_data(self):
        """Save all processed data to output directory."""
        print("\nSaving processed data...")
        
        # Save problem results
        if not self.problem_results.empty:
            self.problem_results.to_csv(
                self.output_dir / 'problem_results.csv', 
                index=False
            )
            print(f"  Saved problem_results.csv ({len(self.problem_results)} rows)")
        
        # Save system metrics
        if not self.system_metrics.empty:
            self.system_metrics.to_csv(
                self.output_dir / 'system_metrics.csv', 
                index=False
            )
            print(f"  Saved system_metrics.csv ({len(self.system_metrics)} rows)")
        
        # Save aggregate statistics
        if self.aggregated_stats:
            with open(self.output_dir / 'aggregate_stats.json', 'w') as f:
                json.dump(self.aggregated_stats, f, indent=2, default=str)
            print(f"  Saved aggregate_stats.json")
        
        # Save pass@k results
        if self.pass_at_k_results:
            # Convert defaultdicts to regular dicts for JSON serialization
            pass_at_k_serializable = {}
            for k, data in self.pass_at_k_results.items():
                pass_at_k_serializable[k] = {
                    'total_problems': data['total_problems'],
                    'passed_problems': data['passed_problems'],
                    'pass_rate': data['pass_rate'],
                    'by_model': dict(data['by_model']),
                    'by_mode': dict(data['by_mode']),
                }
            
            with open(self.output_dir / 'pass_at_k.json', 'w') as f:
                json.dump(pass_at_k_serializable, f, indent=2, default=str)
            print(f"  Saved pass_at_k.json")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of the processed data."""
        report_lines = [
            "# Benchmark Data Processing Summary",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Overall Statistics",
        ]
        
        if self.aggregated_stats and 'overall' in self.aggregated_stats:
            stats = self.aggregated_stats['overall']
            report_lines.extend([
                f"- Total unique problems: {stats.get('total_problems', 'N/A')}",
                f"- Total runs: {stats.get('total_runs', 'N/A')}",
                f"- Overall success rate: {stats.get('overall_success_rate', 0)*100:.1f}%",
                f"- Overall test pass rate: {stats.get('overall_test_pass_rate', 0)*100:.1f}%",
            ])
        
        # Add pass@k summary
        if self.pass_at_k_results:
            report_lines.append("\n## Pass@k Rates")
            for k in sorted(self.pass_at_k_results.keys()):
                rate = self.pass_at_k_results[k]['pass_rate'] * 100
                report_lines.append(f"- Pass@{k}: {rate:.1f}%")
        
        # Add model comparison
        if self.aggregated_stats and 'by_model' in self.aggregated_stats:
            report_lines.append("\n## Model Performance")
            for model_key, stats in sorted(self.aggregated_stats['by_model'].items()):
                report_lines.extend([
                    f"\n### {model_key}",
                    f"- Runs: {stats.get('total_runs', 'N/A')}",
                    f"- Success rate: {stats.get('success_rate', 0)*100:.1f}%",
                    f"- Test pass rate: {stats.get('test_pass_rate', 0)*100:.1f}%",
                    f"- Mean duration: {stats.get('duration_mean', 0):.2f}s",
                ])
                
                if 'ttft_mean' in stats:
                    report_lines.append(f"- Mean TTFT: {stats['ttft_mean']:.1f}ms")
                if 'tps_mean' in stats:
                    report_lines.append(f"- Mean TPS: {stats['tps_mean']:.1f}")
        
        # Save report
        with open(self.output_dir / 'processing_summary.md', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  Saved processing_summary.md")
    
    def run(self):
        """Run the complete processing pipeline."""
        print("Starting benchmark data processing...")
        self.process_all_jobs()
        
        if not self.problem_results.empty:
            self.calculate_aggregate_statistics()
            self.save_processed_data()
            print("\nProcessing complete!")
        else:
            print("\nNo data to process. Ensure organize_benchmark_data.py has been run first.")


if __name__ == "__main__":
    import sys
    
    # Default paths
    processed_dir = Path("benchmark_analysis/processed")
    output_dir = Path("benchmark_analysis/processed")
    
    # Allow command-line override
    if len(sys.argv) > 1:
        processed_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    processor = BenchmarkDataProcessor(processed_dir, output_dir)
    processor.run()