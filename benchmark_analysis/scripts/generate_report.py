#!/usr/bin/env python3
"""
Generate comprehensive benchmark reports in Markdown and LaTeX formats.
Creates publication-ready tables and formatted reports for analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from jinja2 import Template
import re

class ReportGenerator:
    def __init__(self, processed_dir: Path, output_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.problem_results = None
        self.system_metrics = None
        self.aggregate_stats = None
        self.pass_at_k = None
        self.statistical_tests = None
        
        self.load_data()
    
    def load_data(self):
        """Load all processed data and analysis results."""
        print("Loading data for report generation...")
        
        # Load problem results
        problem_file = self.processed_dir / 'problem_results.csv'
        if problem_file.exists():
            self.problem_results = pd.read_csv(problem_file)
        
        # Load system metrics
        system_file = self.processed_dir / 'system_metrics.csv'
        if system_file.exists():
            self.system_metrics = pd.read_csv(system_file)
        
        # Load aggregate statistics
        stats_file = self.processed_dir / 'aggregate_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.aggregate_stats = json.load(f)
        
        # Load pass@k results
        pass_file = self.processed_dir / 'pass_at_k.json'
        if pass_file.exists():
            with open(pass_file, 'r') as f:
                self.pass_at_k = json.load(f)
        
        # Load statistical test results if available
        stats_test_file = self.reports_dir / 'tables' / 'statistical_tests.csv'
        if stats_test_file.exists():
            self.statistical_tests = pd.read_csv(stats_test_file)
    
    def generate_markdown_report(self):
        """Generate comprehensive Markdown report."""
        print("\nGenerating Markdown report...")
        
        report_lines = [
            "# Pocket Agent Benchmark Analysis Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n---\n",
            "## Executive Summary",
            ""
        ]
        
        # Executive summary
        if self.aggregate_stats and 'overall' in self.aggregate_stats:
            stats = self.aggregate_stats['overall']
            report_lines.extend([
                "### Key Findings",
                "",
                f"- **Total benchmark runs:** {stats.get('total_runs', 'N/A')}",
                f"- **Unique problems evaluated:** {stats.get('total_problems', 'N/A')}",
                f"- **Overall success rate:** {stats.get('overall_success_rate', 0)*100:.1f}%",
                f"- **Average test pass rate:** {stats.get('overall_test_pass_rate', 0)*100:.1f}%",
                ""
            ])
            
            # Performance highlights
            if 'duration_seconds_median' in stats:
                report_lines.extend([
                    "### Performance Metrics",
                    "",
                    f"- **Median execution time:** {stats['duration_seconds_median']:.2f} seconds",
                    f"- **95th percentile execution time:** {stats.get('duration_seconds_p95', 0):.2f} seconds",
                ])
            
            if 'ttft_ms_median' in stats:
                report_lines.append(f"- **Median TTFT:** {stats['ttft_ms_median']:.1f} ms")
            
            if 'tps_median' in stats:
                report_lines.append(f"- **Median generation speed:** {stats['tps_median']:.1f} tokens/second")
            
            report_lines.append("")
        
        # Pass@k results
        if self.pass_at_k:
            report_lines.extend([
                "## Pass@k Analysis",
                "",
                "Pass@k measures the probability of generating at least one correct solution within k attempts.",
                "",
                "| k | Pass Rate | Improvement |",
                "|---|-----------|-------------|"
            ])
            
            prev_rate = 0
            for k_str in sorted(self.pass_at_k.keys(), key=int):
                k = int(k_str)
                rate = self.pass_at_k[k_str]['pass_rate'] * 100
                improvement = f"+{rate - prev_rate:.1f}%" if prev_rate > 0 else "baseline"
                report_lines.append(f"| {k} | {rate:.1f}% | {improvement} |")
                prev_rate = rate
            
            report_lines.append("")
        
        # Model comparison
        if self.aggregate_stats and 'by_model' in self.aggregate_stats:
            report_lines.extend([
                "## Model Performance Comparison",
                "",
                "| Model Configuration | Success Rate | Test Pass Rate | Median Duration | Mean TPS |",
                "|-------------------|--------------|----------------|-----------------|----------|"
            ])
            
            for model_key, stats in sorted(self.aggregate_stats['by_model'].items()):
                success_rate = stats.get('success_rate', 0) * 100
                test_pass_rate = stats.get('test_pass_rate', 0) * 100
                duration = stats.get('duration_median', stats.get('duration_mean', 0))
                tps = stats.get('tps_mean', 0)
                
                report_lines.append(
                    f"| {model_key} | {success_rate:.1f}% | {test_pass_rate:.1f}% | "
                    f"{duration:.2f}s | {tps:.1f} |"
                )
            
            report_lines.append("")
        
        # Resource utilization
        if self.system_metrics is not None and not self.system_metrics.empty:
            report_lines.extend([
                "## Resource Utilization",
                "",
                "### GPU Metrics",
                ""
            ])
            
            if 'gpu_util_mean' in self.system_metrics.columns:
                gpu_summary = self.system_metrics.groupby('model').agg({
                    'gpu_util_mean': 'mean',
                    'gpu_memory_max_mb': 'max',
                    'gpu_power_mean_w': 'mean'
                }).round(1)
                
                report_lines.extend([
                    "| Model | Avg GPU Util (%) | Max Memory (MB) | Avg Power (W) |",
                    "|-------|------------------|-----------------|---------------|"
                ])
                
                for model, row in gpu_summary.iterrows():
                    report_lines.append(
                        f"| {model} | {row.get('gpu_util_mean', 0):.1f} | "
                        f"{row.get('gpu_memory_max_mb', 0):.0f} | "
                        f"{row.get('gpu_power_mean_w', 0):.1f} |"
                    )
                
                report_lines.append("")
        
        # Problem difficulty analysis
        if self.problem_results is not None and not self.problem_results.empty:
            report_lines.extend([
                "## Problem Difficulty Analysis",
                "",
                "### Hardest Problems (Lowest Success Rate)",
                ""
            ])
            
            problem_difficulty = self.problem_results.groupby('problem_id')['success'].mean()
            hardest = problem_difficulty.nsmallest(10)
            
            report_lines.extend([
                "| Problem ID | Success Rate |",
                "|------------|--------------|"
            ])
            
            for prob_id, success_rate in hardest.items():
                report_lines.append(f"| {prob_id} | {success_rate*100:.1f}% |")
            
            report_lines.append("")
        
        # Statistical significance
        if self.statistical_tests is not None and not self.statistical_tests.empty:
            report_lines.extend([
                "## Statistical Significance Tests",
                "",
                "| Comparison | Metric | Test | p-value | Significant |",
                "|------------|--------|------|---------|-------------|"
            ])
            
            for _, row in self.statistical_tests.iterrows():
                sig = "Yes ✓" if row['significant'] else "No"
                report_lines.append(
                    f"| {row['comparison']} | {row['metric']} | {row['test']} | "
                    f"{row['p_value']:.4f} | {sig} |"
                )
            
            report_lines.append("")
        
        # Failure analysis
        report_lines.extend([
            "## Failure Analysis",
            ""
        ])
        
        if self.problem_results is not None and not self.problem_results.empty:
            failed = self.problem_results[self.problem_results['success'] == False]
            if not failed.empty:
                failure_reasons = failed.groupby('model').size()
                
                report_lines.extend([
                    "### Failure Count by Model",
                    "",
                    "| Model | Failed Runs |",
                    "|-------|-------------|"
                ])
                
                for model, count in failure_reasons.items():
                    report_lines.append(f"| {model} | {count} |")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "Based on the benchmark analysis, the following recommendations are made:",
            ""
        ])
        
        # Generate recommendations based on data
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "",
            "## Appendix",
            "",
            "### Benchmark Configuration",
            ""
        ])
        
        # Add configuration details if available
        if self.aggregate_stats:
            report_lines.extend([
                "- **Dataset:** MBPP (Mostly Basic Python Problems)",
                f"- **Total problems evaluated:** {self.aggregate_stats['overall'].get('total_problems', 'N/A')}",
                f"- **Samples per problem:** 10 (for pass@k calculation)",
                "- **Temperature:** 0.7",
                "- **Evaluation modes:** base, tool_submission, full_tool",
                ""
            ])
        
        # Visualizations reference
        report_lines.extend([
            "### Generated Visualizations",
            "",
            "The following visualizations have been generated and saved:",
            "",
            "- `visualizations/performance/distributions.png` - Performance metric distributions",
            "- `visualizations/performance/pass_at_k.png` - Pass@k rate curves",
            "- `visualizations/comparisons/model_comparison.png` - Model performance comparison",
            "- `visualizations/comparisons/difficulty_heatmap.png` - Problem difficulty heatmap",
            "- `visualizations/resources/utilization.png` - Resource utilization analysis",
            "- `reports/interactive_dashboard.html` - Interactive Plotly dashboard",
            "",
            "---",
            "",
            "*This report was automatically generated by the Pocket Agent benchmark analysis pipeline.*"
        ])
        
        # Save report
        report_file = self.reports_dir / 'benchmark_report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  Saved Markdown report to {report_file}")
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for academic paper."""
        print("\nGenerating LaTeX tables...")
        
        latex_dir = self.reports_dir / 'latex'
        latex_dir.mkdir(exist_ok=True)
        
        # Table 1: Model Performance Summary
        if self.aggregate_stats and 'by_model' in self.aggregate_stats:
            latex_lines = [
                "% Table: Model Performance Summary",
                "\\begin{table}[h]",
                "\\centering",
                "\\caption{Model Performance Comparison on MBPP Benchmark}",
                "\\label{tab:model_performance}",
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "Model & Success Rate & Test Pass & Duration & TPS \\\\",
                "Configuration & (\\%) & Rate (\\%) & (seconds) & (tokens/s) \\\\",
                "\\midrule"
            ]
            
            for model_key, stats in sorted(self.aggregate_stats['by_model'].items()):
                # Clean model name for LaTeX
                model_name = model_key.replace('_', '\\_')
                success_rate = stats.get('success_rate', 0) * 100
                test_pass_rate = stats.get('test_pass_rate', 0) * 100
                duration = stats.get('duration_median', stats.get('duration_mean', 0))
                tps = stats.get('tps_mean', 0)
                
                latex_lines.append(
                    f"{model_name} & {success_rate:.1f} & {test_pass_rate:.1f} & "
                    f"{duration:.2f} & {tps:.1f} \\\\"
                )
            
            latex_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
            
            with open(latex_dir / 'model_performance.tex', 'w') as f:
                f.write('\n'.join(latex_lines))
        
        # Table 2: Pass@k Results
        if self.pass_at_k:
            latex_lines = [
                "% Table: Pass@k Results",
                "\\begin{table}[h]",
                "\\centering",
                "\\caption{Pass@k Success Rates}",
                "\\label{tab:pass_at_k}",
                "\\begin{tabular}{cc}",
                "\\toprule",
                "k & Pass Rate (\\%) \\\\",
                "\\midrule"
            ]
            
            for k_str in sorted(self.pass_at_k.keys(), key=int):
                k = int(k_str)
                rate = self.pass_at_k[k_str]['pass_rate'] * 100
                latex_lines.append(f"{k} & {rate:.1f} \\\\")
            
            latex_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
            
            with open(latex_dir / 'pass_at_k.tex', 'w') as f:
                f.write('\n'.join(latex_lines))
        
        # Table 3: Resource Utilization
        if self.system_metrics is not None and not self.system_metrics.empty:
            if 'gpu_util_mean' in self.system_metrics.columns:
                gpu_summary = self.system_metrics.groupby('model').agg({
                    'gpu_util_mean': 'mean',
                    'gpu_memory_max_mb': 'max',
                    'gpu_power_mean_w': 'mean'
                }).round(1)
                
                latex_lines = [
                    "% Table: Resource Utilization",
                    "\\begin{table}[h]",
                    "\\centering",
                    "\\caption{Average Resource Utilization by Model}",
                    "\\label{tab:resource_util}",
                    "\\begin{tabular}{lccc}",
                    "\\toprule",
                    "Model & GPU Util. & Max Memory & Avg Power \\\\",
                    "& (\\%) & (MB) & (W) \\\\",
                    "\\midrule"
                ]
                
                for model, row in gpu_summary.iterrows():
                    model_name = model.replace('_', '\\_')
                    latex_lines.append(
                        f"{model_name} & {row.get('gpu_util_mean', 0):.1f} & "
                        f"{row.get('gpu_memory_max_mb', 0):.0f} & "
                        f"{row.get('gpu_power_mean_w', 0):.1f} \\\\"
                    )
                
                latex_lines.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}"
                ])
                
                with open(latex_dir / 'resource_utilization.tex', 'w') as f:
                    f.write('\n'.join(latex_lines))
        
        print(f"  Saved LaTeX tables to {latex_dir}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if self.aggregate_stats and 'by_model' in self.aggregate_stats:
            # Find best performing model
            best_model = None
            best_success = 0
            for model_key, stats in self.aggregate_stats['by_model'].items():
                if stats.get('success_rate', 0) > best_success:
                    best_success = stats['success_rate']
                    best_model = model_key
            
            if best_model:
                recommendations.append(
                    f"**Best performing configuration:** {best_model} with "
                    f"{best_success*100:.1f}% success rate"
                )
            
            # Check for efficiency trade-offs
            fastest_model = None
            fastest_tps = 0
            for model_key, stats in self.aggregate_stats['by_model'].items():
                if stats.get('tps_mean', 0) > fastest_tps:
                    fastest_tps = stats['tps_mean']
                    fastest_model = model_key
            
            if fastest_model and fastest_model != best_model:
                recommendations.append(
                    f"**For speed-critical applications:** Consider {fastest_model} "
                    f"with {fastest_tps:.1f} tokens/second"
                )
        
        # Pass@k recommendations
        if self.pass_at_k:
            pass_1 = self.pass_at_k.get('1', {}).get('pass_rate', 0) * 100
            pass_10 = self.pass_at_k.get('10', {}).get('pass_rate', 0) * 100
            
            if pass_10 - pass_1 > 20:
                recommendations.append(
                    f"**Multiple attempts significantly improve success:** "
                    f"Pass@1={pass_1:.1f}% vs Pass@10={pass_10:.1f}%. "
                    f"Consider implementing retry mechanisms."
                )
        
        # Resource recommendations
        if self.system_metrics is not None and not self.system_metrics.empty:
            if 'gpu_memory_max_mb' in self.system_metrics.columns:
                max_memory = self.system_metrics['gpu_memory_max_mb'].max()
                if max_memory > 8000:  # 8GB
                    recommendations.append(
                        f"**High memory usage detected:** Peak GPU memory usage reached "
                        f"{max_memory:.0f}MB. Consider quantization for memory-constrained environments."
                    )
        
        # Tool usage recommendations
        if self.aggregate_stats and 'by_model' in self.aggregate_stats:
            tool_modes = ['tool_submission', 'full_tool']
            base_success = 0
            tool_success = 0
            
            for model_key, stats in self.aggregate_stats['by_model'].items():
                if 'base' in model_key:
                    base_success = max(base_success, stats.get('success_rate', 0))
                elif any(mode in model_key for mode in tool_modes):
                    tool_success = max(tool_success, stats.get('success_rate', 0))
            
            if tool_success > base_success * 1.1:  # 10% improvement
                recommendations.append(
                    f"**Tool usage improves performance:** Tool-enabled modes show "
                    f"{(tool_success/base_success - 1)*100:.1f}% improvement over base mode."
                )
        
        if not recommendations:
            recommendations.append("Continue monitoring performance across different workloads")
        
        return recommendations
    
    def generate_csv_exports(self):
        """Export key data tables to CSV for further analysis."""
        print("\nExporting data to CSV...")
        
        csv_dir = self.reports_dir / 'csv'
        csv_dir.mkdir(exist_ok=True)
        
        # Export problem-level results
        if self.problem_results is not None:
            # Select key columns for export
            export_cols = [
                'job_id', 'problem_id', 'model', 'mode', 'success', 
                'duration_seconds', 'test_pass_rate', 'total_tokens',
                'ttft_ms', 'tps', 'energy_per_token'
            ]
            
            # Only include columns that exist
            export_cols = [col for col in export_cols if col in self.problem_results.columns]
            
            export_df = self.problem_results[export_cols].copy()
            export_df.to_csv(csv_dir / 'problem_results_export.csv', index=False)
            print(f"  Exported problem results ({len(export_df)} rows)")
        
        # Export aggregated statistics
        if self.aggregate_stats and 'by_model' in self.aggregate_stats:
            rows = []
            for model_key, stats in self.aggregate_stats['by_model'].items():
                row = {'model_configuration': model_key}
                row.update(stats)
                rows.append(row)
            
            stats_df = pd.DataFrame(rows)
            stats_df.to_csv(csv_dir / 'model_statistics.csv', index=False)
            print(f"  Exported model statistics ({len(stats_df)} models)")
        
        # Export pass@k data
        if self.pass_at_k:
            pass_k_rows = []
            for k_str, data in self.pass_at_k.items():
                row = {
                    'k': int(k_str),
                    'overall_pass_rate': data['pass_rate'],
                    'total_problems': data['total_problems'],
                    'passed_problems': data['passed_problems']
                }
                
                # Add model-specific rates
                for model_key, model_data in data.get('by_model', {}).items():
                    row[f'{model_key}_pass_rate'] = model_data['pass_rate']
                
                pass_k_rows.append(row)
            
            pass_k_df = pd.DataFrame(pass_k_rows)
            pass_k_df.to_csv(csv_dir / 'pass_at_k_detailed.csv', index=False)
            print(f"  Exported pass@k data")
    
    def run(self):
        """Run complete report generation pipeline."""
        print("Starting report generation...")
        
        if self.problem_results is None or self.aggregate_stats is None:
            print("Error: Required data not found. Run processing and analysis scripts first.")
            return
        
        # Generate all report formats
        self.generate_markdown_report()
        self.generate_latex_tables()
        self.generate_csv_exports()
        
        print("\nReport generation complete!")
        print(f"Reports saved to: {self.reports_dir}")
        print("\nGenerated files:")
        print("  - benchmark_report.md : Comprehensive Markdown report")
        print("  - latex/*.tex : LaTeX tables for academic papers")
        print("  - csv/*.csv : CSV exports for further analysis")


if __name__ == "__main__":
    import sys
    
    # Default paths
    processed_dir = Path("benchmark_analysis/processed")
    output_dir = Path("benchmark_analysis")
    
    # Allow command-line override
    if len(sys.argv) > 1:
        processed_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    generator = ReportGenerator(processed_dir, output_dir)
    generator.run()