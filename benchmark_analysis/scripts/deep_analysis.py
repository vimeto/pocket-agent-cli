#!/usr/bin/env python3
"""
Deep analysis of benchmark results with detailed insights.
Focuses on TPS, TTFT, tool calls, and energy metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Any

class DeepBenchmarkAnalyzer:
    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.df = pd.read_csv(self.processed_dir / 'problem_results.csv')
        self.system_df = pd.read_csv(self.processed_dir / 'system_metrics.csv')
        
        # Create model+version column for better grouping
        self.df['model_version'] = self.df['model'].apply(self._extract_version)
        self.df['model_base'] = self.df['model'].apply(self._extract_base_model)
        
    def _extract_version(self, model_name):
        """Extract quantization version from model name."""
        if '_Q4' in model_name or 'Q4_K_M' in model_name:
            return 'Q4_K_M'
        elif '_F16' in model_name:
            return 'F16'
        else:
            # Try to infer from model name patterns
            parts = model_name.split('_')
            if len(parts) > 1 and parts[-1] in ['Q4', 'F16', 'BF16']:
                return parts[-1]
            return 'unknown'
    
    def _extract_base_model(self, model_name):
        """Extract base model name without version."""
        # Remove common version suffixes
        for suffix in ['_Q4', '_F16', '_BF16', '_Q4_K_M']:
            if suffix in model_name:
                return model_name.split(suffix)[0]
        return model_name
    
    def analyze_tps_metrics(self):
        """Detailed TPS analysis."""
        print("\n" + "="*60)
        print("TPS (Tokens Per Second) Analysis")
        print("="*60)
        
        # Filter out outliers (TPS > 1000 are likely errors)
        df_clean = self.df[self.df['tps'] < 1000].copy()
        
        # Overall statistics
        print("\n### Overall TPS Statistics")
        print(f"Mean TPS: {df_clean['tps'].mean():.1f} tokens/sec")
        print(f"Median TPS: {df_clean['tps'].median():.1f} tokens/sec")
        print(f"P95 TPS: {df_clean['tps'].quantile(0.95):.1f} tokens/sec")
        
        # By model
        print("\n### TPS by Model (median values)")
        model_tps = df_clean.groupby('model')['tps'].agg(['median', 'mean', 'std']).round(1)
        model_tps = model_tps.sort_values('median', ascending=False)
        for model, row in model_tps.iterrows():
            print(f"  {model:40s}: {row['median']:6.1f} (±{row['std']:5.1f}) tokens/sec")
        
        # By quantization
        print("\n### TPS by Quantization Level")
        for base_model in df_clean['model_base'].unique():
            model_data = df_clean[df_clean['model_base'] == base_model]
            if model_data['model_version'].nunique() > 1:
                print(f"\n{base_model}:")
                for version in model_data['model_version'].unique():
                    if version != 'unknown':
                        version_data = model_data[model_data['model_version'] == version]
                        if not version_data.empty:
                            median_tps = version_data['tps'].median()
                            print(f"  {version:10s}: {median_tps:6.1f} tokens/sec")
        
        return df_clean
    
    def analyze_ttft_metrics(self):
        """Detailed TTFT analysis."""
        print("\n" + "="*60)
        print("TTFT (Time To First Token) Analysis")
        print("="*60)
        
        # Many TTFT values are 0, which likely means immediate response
        # Let's focus on non-zero values
        df_nonzero = self.df[self.df['ttft_ms'] > 0].copy()
        
        print(f"\n### TTFT Statistics (excluding zeros)")
        print(f"Responses with TTFT > 0: {len(df_nonzero)} out of {len(self.df)} ({len(df_nonzero)/len(self.df)*100:.1f}%)")
        
        if len(df_nonzero) > 0:
            print(f"Mean TTFT: {df_nonzero['ttft_ms'].mean():.2f} ms")
            print(f"Median TTFT: {df_nonzero['ttft_ms'].median():.2f} ms")
            print(f"P95 TTFT: {df_nonzero['ttft_ms'].quantile(0.95):.2f} ms")
            
            # By model
            print("\n### TTFT by Model (median, excluding zeros)")
            model_ttft = df_nonzero.groupby('model')['ttft_ms'].agg(['median', 'mean', 'count']).round(2)
            model_ttft = model_ttft.sort_values('median')
            for model, row in model_ttft.iterrows():
                if row['count'] > 10:  # Only show if we have enough samples
                    print(f"  {model:40s}: {row['median']:8.2f} ms (n={row['count']:.0f})")
    
    def analyze_tool_calls(self):
        """Detailed tool call analysis."""
        print("\n" + "="*60)
        print("Tool Call Analysis")
        print("="*60)
        
        # Problems with tool calls
        df_with_tools = self.df[self.df['num_tool_calls'] > 0].copy()
        
        print(f"\n### Tool Call Usage")
        print(f"Problems with tool calls: {len(df_with_tools)} out of {len(self.df)} ({len(df_with_tools)/len(self.df)*100:.1f}%)")
        print(f"Average tool calls when used: {df_with_tools['num_tool_calls'].mean():.1f}")
        print(f"Max tool calls in a single problem: {df_with_tools['num_tool_calls'].max():.0f}")
        
        # Tool call time analysis
        if 'avg_tool_call_ms' in df_with_tools.columns:
            tool_time = df_with_tools['avg_tool_call_ms'].dropna()
            if len(tool_time) > 0:
                print(f"\n### Tool Call Timing")
                print(f"Average tool call duration: {tool_time.mean():.1f} ms")
                print(f"Median tool call duration: {tool_time.median():.1f} ms")
                print(f"Total tool time per problem: {df_with_tools['total_tool_time_ms'].mean():.1f} ms")
        
        # Success rate with/without tools
        print(f"\n### Success Rate Impact")
        with_tools_success = df_with_tools['success'].mean()
        without_tools = self.df[self.df['num_tool_calls'] == 0]
        without_tools_success = without_tools['success'].mean()
        
        print(f"Success rate WITH tool calls: {with_tools_success*100:.1f}%")
        print(f"Success rate WITHOUT tool calls: {without_tools_success*100:.1f}%")
        
        # By mode
        print(f"\n### Tool Usage by Mode")
        for mode in self.df['mode'].unique():
            mode_data = self.df[self.df['mode'] == mode]
            tool_usage = (mode_data['num_tool_calls'] > 0).mean()
            print(f"  {mode:20s}: {tool_usage*100:.1f}% of problems use tools")
    
    def analyze_energy_metrics(self):
        """Energy consumption analysis."""
        print("\n" + "="*60)
        print("Energy Consumption Analysis")
        print("="*60)
        
        # Energy per problem
        print(f"\n### Energy per Problem Solve")
        energy_by_model = self.df.groupby('model')['energy_joules'].agg(['mean', 'median', 'sum']).round(2)
        energy_by_model = energy_by_model.sort_values('mean')
        
        for model, row in energy_by_model.iterrows():
            print(f"  {model:40s}: {row['mean']:8.2f} J (median: {row['median']:.2f} J)")
        
        # Energy efficiency (joules per token)
        print(f"\n### Energy Efficiency (Joules per Token)")
        efficiency = self.df.groupby('model')['energy_per_token'].agg(['mean', 'median']).round(4)
        efficiency = efficiency.sort_values('mean')
        
        for model, row in efficiency.iterrows():
            if not pd.isna(row['mean']):
                print(f"  {model:40s}: {row['mean']:8.4f} J/token")
        
        # Compare quantization levels
        print(f"\n### Energy Comparison: Q4 vs F16")
        models_with_both = set()
        
        for base_model in self.df['model_base'].unique():
            model_data = self.df[self.df['model_base'] == base_model]
            versions = model_data['model_version'].unique()
            
            if 'Q4_K_M' in versions and 'F16' in versions:
                q4_energy = model_data[model_data['model_version'] == 'Q4_K_M']['energy_joules'].mean()
                f16_energy = model_data[model_data['model_version'] == 'F16']['energy_joules'].mean()
                
                if not pd.isna(q4_energy) and not pd.isna(f16_energy):
                    saving = (1 - q4_energy/f16_energy) * 100
                    print(f"  {base_model}:")
                    print(f"    Q4_K_M: {q4_energy:.2f} J")
                    print(f"    F16:    {f16_energy:.2f} J")
                    print(f"    Savings with Q4: {saving:.1f}%")
    
    def analyze_cpu_gpu_usage(self):
        """CPU and GPU usage analysis."""
        print("\n" + "="*60)
        print("CPU and GPU Usage Analysis")
        print("="*60)
        
        # From problem-level data
        print(f"\n### Average Resource Usage (from problem data)")
        resource_cols = ['cpu_avg_percent', 'gpu_avg_percent', 'memory_avg_percent']
        
        for col in resource_cols:
            if col in self.df.columns:
                by_model = self.df.groupby('model')[col].mean().round(1)
                by_model = by_model.sort_values(ascending=False)
                
                print(f"\n{col.replace('_', ' ').title()}:")
                for model, value in by_model.items():
                    if not pd.isna(value):
                        print(f"  {model:40s}: {value:5.1f}%")
        
        # From system-level monitoring
        if not self.system_df.empty:
            print(f"\n### System-Level Monitoring")
            
            sys_metrics = self.system_df.groupby('model')[['cpu_util_mean', 'memory_util_mean']].mean().round(1)
            
            print("\nCPU Utilization (system level):")
            for model, row in sys_metrics.iterrows():
                if not pd.isna(row['cpu_util_mean']):
                    print(f"  {model:40s}: {row['cpu_util_mean']:5.1f}%")
            
            print("\nMemory Utilization (system level):")
            for model, row in sys_metrics.iterrows():
                if not pd.isna(row['memory_util_mean']):
                    print(f"  {model:40s}: {row['memory_util_mean']:5.1f}%")
    
    def create_comparison_table(self):
        """Create a comprehensive comparison table."""
        print("\n" + "="*60)
        print("Comprehensive Model Comparison Table")
        print("="*60)
        
        # Aggregate metrics by model
        metrics = self.df.groupby('model').agg({
            'success': 'mean',
            'tps': 'median',
            'ttft_ms': lambda x: x[x > 0].median() if len(x[x > 0]) > 0 else 0,
            'energy_joules': 'mean',
            'energy_per_token': 'mean',
            'num_tool_calls': 'mean',
            'duration_seconds': 'median',
            'total_tokens': 'mean'
        }).round(2)
        
        # Add problem counts
        problem_counts = self.df.groupby('model')['problem_id'].nunique()
        metrics['problems'] = problem_counts
        
        # Sort by success rate
        metrics = metrics.sort_values('success', ascending=False)
        
        # Print formatted table
        print("\n")
        print(f"{'Model':<40} {'Success':<8} {'TPS':<8} {'TTFT':<8} {'Energy':<10} {'Tools':<8} {'Problems':<10}")
        print(f"{'':40} {'(%)':<8} {'(t/s)':<8} {'(ms)':<8} {'(J/prob)':<10} {'(avg)':<8} {'(count)':<10}")
        print("-" * 120)
        
        for model, row in metrics.iterrows():
            print(f"{model:<40} {row['success']*100:<8.1f} {row['tps']:<8.1f} "
                  f"{row['ttft_ms']:<8.1f} {row['energy_joules']:<10.1f} "
                  f"{row['num_tool_calls']:<8.2f} {row['problems']:<10.0f}")
        
        return metrics
    
    def save_detailed_report(self, output_file: Path):
        """Save all analysis to a detailed report file."""
        import sys
        from io import StringIO
        
        # Capture all output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        # Run all analyses
        self.analyze_tps_metrics()
        self.analyze_ttft_metrics()
        self.analyze_tool_calls()
        self.analyze_energy_metrics()
        self.analyze_cpu_gpu_usage()
        metrics_table = self.create_comparison_table()
        
        # Get the output
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write("# Deep Benchmark Analysis Report\n\n")
            f.write(output)
        
        print(f"Detailed report saved to: {output_file}")
        
        # Also save metrics table as CSV
        csv_file = output_file.parent / f"{output_file.stem}_metrics.csv"
        metrics_table.to_csv(csv_file)
        print(f"Metrics table saved to: {csv_file}")


if __name__ == "__main__":
    import sys
    
    # Default path
    processed_dir = Path("benchmark_analysis/processed")
    
    if len(sys.argv) > 1:
        processed_dir = Path(sys.argv[1])
    
    analyzer = DeepBenchmarkAnalyzer(processed_dir)
    
    # Print analysis to console
    analyzer.analyze_tps_metrics()
    analyzer.analyze_ttft_metrics()
    analyzer.analyze_tool_calls()
    analyzer.analyze_energy_metrics()
    analyzer.analyze_cpu_gpu_usage()
    analyzer.create_comparison_table()
    
    # Save detailed report
    report_file = Path("benchmark_analysis/reports/deep_analysis_report.txt")
    analyzer.save_detailed_report(report_file)