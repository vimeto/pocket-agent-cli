#!/usr/bin/env python3
"""
Organize benchmark data from Mahti HPC cluster.
Groups files by SLURM job ID and creates metadata index.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re
import csv
from collections import defaultdict

class BenchmarkDataOrganizer:
    def __init__(self, raw_data_dir: Path, processed_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern to extract job IDs from filenames
        self.job_id_patterns = {
            'benchmark_results': re.compile(r'bench_.*_job(\d+)'),
            'gpu_monitor': re.compile(r'gpu_monitor_(\d+)\.csv'),
            'cpu_monitor': re.compile(r'cpu_monitor_(\d+)\.csv'),
            'failures': re.compile(r'failures_(\d+)\.log'),
            'slurm_out': re.compile(r'benchmark_(\d+)\.out'),
            'slurm_err': re.compile(r'benchmark_(\d+)\.err'),
        }
        
        self.job_metadata = defaultdict(lambda: {
            'job_id': None,
            'benchmark_results': [],
            'gpu_monitor': None,
            'cpu_monitor': None,
            'failures_log': None,
            'slurm_out': None,
            'slurm_err': None,
            'model_configs': set(),
            'modes': set(),
            'start_time': None,
            'end_time': None,
            'total_problems': 0,
            'successful_problems': 0,
        })
    
    def scan_raw_data(self):
        """Scan raw data directory and organize by job ID."""
        print("Scanning raw data directory...")
        
        # Scan benchmark results
        results_dir = self.raw_data_dir / 'data' / 'results'
        if results_dir.exists():
            for result_dir in results_dir.glob('bench_*'):
                match = self.job_id_patterns['benchmark_results'].search(result_dir.name)
                if match:
                    job_id = match.group(1)
                    self.job_metadata[job_id]['job_id'] = job_id
                    self.job_metadata[job_id]['benchmark_results'].append(result_dir)
                    
                    # Extract model and mode from directory name
                    parts = result_dir.name.split('_')
                    if len(parts) >= 4:
                        model_name = parts[1]
                        model_version = parts[2]
                        mode = parts[3]
                        self.job_metadata[job_id]['model_configs'].add(f"{model_name}_{model_version}")
                        self.job_metadata[job_id]['modes'].add(mode)
        
        # Scan monitoring logs
        logs_dir = self.raw_data_dir / 'data' / 'logs'
        if logs_dir.exists():
            for log_file in logs_dir.iterdir():
                for log_type, pattern in self.job_id_patterns.items():
                    if log_type in ['gpu_monitor', 'cpu_monitor', 'failures']:
                        match = pattern.search(log_file.name)
                        if match:
                            job_id = match.group(1)
                            self.job_metadata[job_id]['job_id'] = job_id
                            self.job_metadata[job_id][log_type] = log_file
        
        # Scan SLURM logs
        slurm_logs_dir = self.raw_data_dir / 'logs'
        if slurm_logs_dir.exists():
            for log_file in slurm_logs_dir.iterdir():
                for log_type in ['slurm_out', 'slurm_err']:
                    pattern = self.job_id_patterns[log_type]
                    match = pattern.search(log_file.name)
                    if match:
                        job_id = match.group(1)
                        self.job_metadata[job_id]['job_id'] = job_id
                        self.job_metadata[job_id][log_type] = log_file
        
        print(f"Found {len(self.job_metadata)} unique job IDs")
    
    def extract_job_details(self, job_id: str) -> Dict[str, Any]:
        """Extract detailed information from job files."""
        job_data = self.job_metadata[job_id]
        
        # Parse benchmark results
        all_problems = []
        for result_dir in job_data['benchmark_results']:
            # First check for benchmark_summary.json for aggregate stats
            summary_file = result_dir / 'benchmark_summary.json'
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                        # Extract aggregate stats from summary
                        for session in summary_data.get('sessions', []):
                            if 'aggregate_stats' in session:
                                stats = session['aggregate_stats']
                                job_data['total_problems'] += stats.get('total_problems', 0)
                                job_data['successful_problems'] += stats.get('passed_problems', 0)
                except Exception as e:
                    print(f"Error reading summary {summary_file}: {e}")
            
            # Look for session JSON files in nested structure
            # Pattern: result_dir/model_name/mode/*.json
            for model_dir in result_dir.glob('*/'):
                if model_dir.is_dir():
                    for mode_dir in model_dir.glob('*/'):
                        if mode_dir.is_dir():
                            for json_file in mode_dir.glob('*.json'):
                                if not json_file.name.startswith('benchmark_summary'):
                                    try:
                                        with open(json_file, 'r') as f:
                                            session_data = json.load(f)
                                            
                                            # Extract timing information
                                            if session_data.get('start_time'):
                                                start_time = datetime.fromisoformat(session_data['start_time'])
                                                if not job_data['start_time'] or start_time < job_data['start_time']:
                                                    job_data['start_time'] = start_time
                                            
                                            if session_data.get('end_time'):
                                                end_time = datetime.fromisoformat(session_data['end_time'])
                                                if not job_data['end_time'] or end_time > job_data['end_time']:
                                                    job_data['end_time'] = end_time
                                            
                                            # Count problems if not already counted from summary
                                            if job_data['total_problems'] == 0:
                                                problems = session_data.get('problems', [])
                                                all_problems.extend(problems)
                                                job_data['total_problems'] += len(problems)
                                                job_data['successful_problems'] += sum(1 for p in problems if p.get('success'))
                                            
                                    except Exception as e:
                                        print(f"Error reading {json_file}: {e}")
        
        # Parse SLURM output for additional metadata
        if job_data['slurm_out'] and job_data['slurm_out'].exists():
            try:
                with open(job_data['slurm_out'], 'r') as f:
                    content = f.read()
                    
                    # Extract configuration from SLURM output
                    if 'Model:' in content:
                        for line in content.split('\n'):
                            if line.startswith('Model:'):
                                job_data['model'] = line.split(':', 1)[1].strip()
                            elif line.startswith('Model Version:'):
                                job_data['model_version'] = line.split(':', 1)[1].strip()
                            elif line.startswith('Mode:'):
                                job_data['mode'] = line.split(':', 1)[1].strip()
                            elif line.startswith('Problems:'):
                                job_data['problem_range'] = line.split(':', 1)[1].strip()
                            elif line.startswith('Samples per problem:'):
                                job_data['num_samples'] = int(line.split(':', 1)[1].strip())
            except Exception as e:
                print(f"Error reading SLURM output: {e}")
        
        return job_data
    
    def organize_by_job(self):
        """Organize files by job ID in processed directory."""
        for job_id, job_data in self.job_metadata.items():
            if not job_data['job_id']:
                continue
            
            print(f"\nProcessing Job {job_id}...")
            
            # Create job directory
            job_dir = self.processed_dir / f"job_{job_id}"
            job_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (job_dir / 'benchmark_results').mkdir(exist_ok=True)
            (job_dir / 'system_metrics').mkdir(exist_ok=True)
            (job_dir / 'logs').mkdir(exist_ok=True)
            
            # Copy benchmark results
            for result_dir in job_data['benchmark_results']:
                if result_dir.exists():
                    dest = job_dir / 'benchmark_results' / result_dir.name
                    if not dest.exists():
                        shutil.copytree(result_dir, dest)
                        print(f"  Copied benchmark results: {result_dir.name}")
            
            # Copy system metrics
            if job_data['gpu_monitor'] and job_data['gpu_monitor'].exists():
                shutil.copy2(job_data['gpu_monitor'], job_dir / 'system_metrics' / 'gpu_monitor.csv')
                print(f"  Copied GPU monitor data")
            
            if job_data['cpu_monitor'] and job_data['cpu_monitor'].exists():
                shutil.copy2(job_data['cpu_monitor'], job_dir / 'system_metrics' / 'cpu_monitor.csv')
                print(f"  Copied CPU monitor data")
            
            # Copy logs
            if job_data['slurm_out'] and job_data['slurm_out'].exists():
                shutil.copy2(job_data['slurm_out'], job_dir / 'logs' / 'slurm.out')
                print(f"  Copied SLURM output")
            
            if job_data['slurm_err'] and job_data['slurm_err'].exists():
                shutil.copy2(job_data['slurm_err'], job_dir / 'logs' / 'slurm.err')
                print(f"  Copied SLURM errors")
            
            if job_data['failures_log'] and job_data['failures_log'].exists():
                shutil.copy2(job_data['failures_log'], job_dir / 'logs' / 'failures.log')
                print(f"  Copied failures log")
            
            # Extract and save job metadata
            job_details = self.extract_job_details(job_id)
            
            # Convert sets to lists for JSON serialization
            metadata = {
                'job_id': job_id,
                'model_configs': list(job_details['model_configs']),
                'modes': list(job_details['modes']),
                'start_time': job_details['start_time'].isoformat() if job_details['start_time'] else None,
                'end_time': job_details['end_time'].isoformat() if job_details['end_time'] else None,
                'duration_seconds': (job_details['end_time'] - job_details['start_time']).total_seconds() 
                    if job_details['start_time'] and job_details['end_time'] else None,
                'total_problems': job_details['total_problems'],
                'successful_problems': job_details['successful_problems'],
                'success_rate': job_details['successful_problems'] / job_details['total_problems'] 
                    if job_details['total_problems'] > 0 else 0,
            }
            
            # Add additional metadata if available
            for key in ['model', 'model_version', 'mode', 'problem_range', 'num_samples']:
                if key in job_details:
                    metadata[key] = job_details[key]
            
            # Save metadata
            with open(job_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Saved metadata: {job_details['total_problems']} problems, "
                  f"{job_details['successful_problems']} successful")
    
    def create_master_index(self):
        """Create a master index of all jobs."""
        index = {
            'created_at': datetime.now().isoformat(),
            'total_jobs': len(self.job_metadata),
            'jobs': []
        }
        
        for job_id in sorted(self.job_metadata.keys()):
            job_dir = self.processed_dir / f"job_{job_id}"
            metadata_file = job_dir / 'metadata.json'
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    index['jobs'].append(metadata)
        
        # Save master index
        with open(self.processed_dir / 'master_index.json', 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\nCreated master index with {len(index['jobs'])} jobs")
        
        # Print summary
        print("\n" + "="*60)
        print("ORGANIZATION SUMMARY")
        print("="*60)
        
        total_problems = sum(j.get('total_problems', 0) for j in index['jobs'])
        successful_problems = sum(j.get('successful_problems', 0) for j in index['jobs'])
        
        print(f"Total jobs processed: {len(index['jobs'])}")
        print(f"Total problems evaluated: {total_problems}")
        print(f"Successful problems: {successful_problems}")
        if total_problems > 0:
            print(f"Overall success rate: {successful_problems/total_problems*100:.1f}%")
        else:
            print(f"Overall success rate: N/A (no problems found)")
        
        # Group by model configuration
        model_stats = defaultdict(lambda: {'jobs': 0, 'problems': 0, 'successful': 0})
        for job in index['jobs']:
            for config in job.get('model_configs', []):
                model_stats[config]['jobs'] += 1
                model_stats[config]['problems'] += job.get('total_problems', 0)
                model_stats[config]['successful'] += job.get('successful_problems', 0)
        
        if model_stats:
            print("\nBy Model Configuration:")
            for config, stats in sorted(model_stats.items()):
                success_rate = stats['successful'] / stats['problems'] * 100 if stats['problems'] > 0 else 0
                if stats['problems'] > 0:
                    print(f"  {config}: {stats['jobs']} jobs, {stats['problems']} problems, "
                          f"{success_rate:.1f}% success")
                else:
                    print(f"  {config}: {stats['jobs']} jobs, {stats['problems']} problems (no data found)")
    
    def run(self):
        """Run the complete organization process."""
        print("Starting benchmark data organization...")
        self.scan_raw_data()
        self.organize_by_job()
        self.create_master_index()
        print("\nOrganization complete!")


if __name__ == "__main__":
    import sys
    
    # Default paths
    raw_data_dir = Path("benchmark_analysis/raw_data")
    processed_dir = Path("benchmark_analysis/processed")
    
    # Allow command-line override
    if len(sys.argv) > 1:
        raw_data_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        processed_dir = Path(sys.argv[2])
    
    organizer = BenchmarkDataOrganizer(raw_data_dir, processed_dir)
    organizer.run()