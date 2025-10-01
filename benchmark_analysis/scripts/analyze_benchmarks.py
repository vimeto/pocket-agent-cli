#!/usr/bin/env python3
"""
Analyze benchmark results with statistical analysis and visualizations.
Generates graphs, statistical tests, and comparative analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as scipy_stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BenchmarkAnalyzer:
    def __init__(self, processed_dir: Path, output_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'visualizations'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Load data
        self.problem_results = None
        self.system_metrics = None
        self.aggregate_stats = None
        self.pass_at_k = None
        
        self.load_processed_data()
    
    def load_processed_data(self):
        """Load processed data from CSV and JSON files."""
        print("Loading processed data...")
        
        # Load problem results
        problem_file = self.processed_dir / 'problem_results.csv'
        if problem_file.exists():
            self.problem_results = pd.read_csv(problem_file)
            print(f"  Loaded {len(self.problem_results)} problem results")
        
        # Load system metrics
        system_file = self.processed_dir / 'system_metrics.csv'
        if system_file.exists():
            self.system_metrics = pd.read_csv(system_file)
            print(f"  Loaded {len(self.system_metrics)} system metric records")
        
        # Load aggregate statistics
        stats_file = self.processed_dir / 'aggregate_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.aggregate_stats = json.load(f)
            print(f"  Loaded aggregate statistics")
        
        # Load pass@k results
        pass_file = self.processed_dir / 'pass_at_k.json'
        if pass_file.exists():
            with open(pass_file, 'r') as f:
                self.pass_at_k = json.load(f)
            print(f"  Loaded pass@k results")
    
    def create_performance_distributions(self):
        """Create distribution plots for key performance metrics."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for performance distributions")
            return
        
        print("\nCreating performance distribution plots...")
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Metric Distributions', fontsize=16)
        
        # Duration distribution
        ax = axes[0, 0]
        data = self.problem_results['duration_seconds'].dropna()
        if not data.empty:
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(data.median(), color='red', linestyle='--', label=f'Median: {data.median():.2f}s')
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Execution Duration')
            ax.legend()
        
        # TTFT distribution
        ax = axes[0, 1]
        if 'ttft_ms' in self.problem_results.columns:
            data = self.problem_results['ttft_ms'].dropna()
            if not data.empty:
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(data.median(), color='red', linestyle='--', label=f'Median: {data.median():.1f}ms')
                ax.set_xlabel('TTFT (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('Time to First Token')
                ax.legend()
        
        # TPS distribution
        ax = axes[0, 2]
        if 'tps' in self.problem_results.columns:
            data = self.problem_results['tps'].dropna()
            if not data.empty:
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(data.median(), color='red', linestyle='--', label=f'Median: {data.median():.1f}')
                ax.set_xlabel('Tokens per Second')
                ax.set_ylabel('Frequency')
                ax.set_title('Generation Speed (TPS)')
                ax.legend()
        
        # Token count distribution
        ax = axes[1, 0]
        if 'total_tokens' in self.problem_results.columns:
            data = self.problem_results['total_tokens'].dropna()
            if not data.empty:
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(data.median(), color='red', linestyle='--', label=f'Median: {data.median():.0f}')
                ax.set_xlabel('Total Tokens')
                ax.set_ylabel('Frequency')
                ax.set_title('Token Count')
                ax.legend()
        
        # Success rate by model
        ax = axes[1, 1]
        if 'model' in self.problem_results.columns:
            model_success = self.problem_results.groupby('model')['success'].mean() * 100
            model_success.plot(kind='bar', ax=ax)
            ax.set_xlabel('Model')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rate by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Test pass rate distribution
        ax = axes[1, 2]
        data = self.problem_results['test_pass_rate'] * 100
        ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}%')
        ax.set_xlabel('Test Pass Rate (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Test Pass Rate Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance' / 'distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved performance distributions plot")
    
    def create_model_comparison_plots(self):
        """Create comparison plots between different models and modes."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for model comparison")
            return
        
        print("\nCreating model comparison plots...")
        
        # Prepare data
        df = self.problem_results.copy()
        if 'model_mode' not in df.columns:
            df['model_mode'] = df['model'] + '_' + df['mode']
        
        # Create comparison boxplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Duration comparison
        ax = axes[0, 0]
        if 'duration_seconds' in df.columns:
            df.boxplot(column='duration_seconds', by='model_mode', ax=ax)
            ax.set_xlabel('Model Configuration')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title('Execution Time by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        # TPS comparison
        ax = axes[0, 1]
        if 'tps' in df.columns:
            df.boxplot(column='tps', by='model_mode', ax=ax)
            ax.set_xlabel('Model Configuration')
            ax.set_ylabel('Tokens per Second')
            ax.set_title('Generation Speed by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Success rate comparison
        ax = axes[1, 0]
        success_rates = df.groupby('model_mode')['success'].mean() * 100
        success_rates.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Energy efficiency comparison
        ax = axes[1, 1]
        if 'energy_per_token' in df.columns:
            energy_data = df.groupby('model_mode')['energy_per_token'].mean()
            energy_data.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Model Configuration')
            ax.set_ylabel('Energy per Token (Joules)')
            ax.set_title('Energy Efficiency Comparison')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comparisons' / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved model comparison plots")
    
    def create_pass_at_k_visualization(self):
        """Create pass@k rate visualizations."""
        if not self.pass_at_k:
            print("No pass@k data available")
            return
        
        print("\nCreating pass@k visualizations...")
        
        # Prepare data for plotting
        k_values = []
        overall_rates = []
        model_data = {}
        
        for k_str in sorted(self.pass_at_k.keys(), key=int):
            k = int(k_str)
            k_values.append(k)
            overall_rates.append(self.pass_at_k[k_str]['pass_rate'] * 100)
            
            # Collect model-specific data
            for model_key, stats in self.pass_at_k[k_str].get('by_model', {}).items():
                if model_key not in model_data:
                    model_data[model_key] = []
                model_data[model_key].append(stats['pass_rate'] * 100)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall pass@k curve
        ax1.plot(k_values, overall_rates, 'o-', linewidth=2, markersize=8, label='Overall')
        ax1.set_xlabel('k (number of attempts)', fontsize=12)
        ax1.set_ylabel('Pass Rate (%)', fontsize=12)
        ax1.set_title('Pass@k Rates', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        
        # Add value labels
        for k, rate in zip(k_values, overall_rates):
            ax1.annotate(f'{rate:.1f}%', (k, rate), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # Model-specific pass@k curves
        for model_key, rates in model_data.items():
            ax2.plot(k_values, rates, 'o-', linewidth=2, markersize=6, label=model_key, alpha=0.7)
        
        ax2.set_xlabel('k (number of attempts)', fontsize=12)
        ax2.set_ylabel('Pass Rate (%)', fontsize=12)
        ax2.set_title('Pass@k Rates by Model Configuration', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        ax2.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance' / 'pass_at_k.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved pass@k visualization")
    
    def create_resource_utilization_plots(self):
        """Create resource utilization plots."""
        if self.system_metrics is None or self.system_metrics.empty:
            print("No system metrics data available")
            return
        
        print("\nCreating resource utilization plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resource Utilization Analysis', fontsize=16)
        
        # GPU utilization
        ax = axes[0, 0]
        if 'gpu_util_mean' in self.system_metrics.columns:
            data = self.system_metrics.groupby('model')['gpu_util_mean'].mean()
            data.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_xlabel('Model')
            ax.set_ylabel('GPU Utilization (%)')
            ax.set_title('Average GPU Utilization by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # GPU memory usage
        ax = axes[0, 1]
        if 'gpu_memory_max_mb' in self.system_metrics.columns:
            data = self.system_metrics.groupby('model')['gpu_memory_max_mb'].mean()
            data.plot(kind='bar', ax=ax, color='purple', edgecolor='black')
            ax.set_xlabel('Model')
            ax.set_ylabel('GPU Memory (MB)')
            ax.set_title('Maximum GPU Memory Usage')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # CPU utilization
        ax = axes[1, 0]
        if 'cpu_util_mean' in self.system_metrics.columns:
            data = self.system_metrics.groupby('model')['cpu_util_mean'].mean()
            data.plot(kind='bar', ax=ax, color='cyan', edgecolor='black')
            ax.set_xlabel('Model')
            ax.set_ylabel('CPU Utilization (%)')
            ax.set_title('Average CPU Utilization by Model')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Power consumption
        ax = axes[1, 1]
        if 'gpu_power_mean_w' in self.system_metrics.columns:
            data = self.system_metrics.groupby('model')['gpu_power_mean_w'].mean()
            data.plot(kind='bar', ax=ax, color='red', edgecolor='black')
            ax.set_xlabel('Model')
            ax.set_ylabel('Power (Watts)')
            ax.set_title('Average GPU Power Consumption')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'resources' / 'utilization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved resource utilization plots")
    
    def create_problem_difficulty_heatmap(self):
        """Create a heatmap showing problem difficulty across models."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for difficulty heatmap")
            return
        
        print("\nCreating problem difficulty heatmap...")
        
        # Create model_mode column if it doesn't exist
        if 'model_mode' not in self.problem_results.columns:
            self.problem_results['model_mode'] = self.problem_results['model'] + '_' + self.problem_results['mode']
        
        # Create pivot table of success rates
        pivot = self.problem_results.pivot_table(
            index='problem_id',
            columns='model_mode',
            values='success',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, cmap='RdYlGn', center=0.5, vmin=0, vmax=1, 
                   cbar_kws={'label': 'Success Rate'}, 
                   linewidths=0.5, linecolor='gray')
        plt.title('Problem Difficulty Heatmap (Success Rate by Problem and Model)', fontsize=14)
        plt.xlabel('Model Configuration', fontsize=12)
        plt.ylabel('Problem ID', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comparisons' / 'difficulty_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Saved difficulty heatmap")
    
    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for interactive dashboard")
            return
        
        print("\nCreating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Success Rate by Model', 'Duration Distribution',
                          'Pass@k Rates', 'Token Generation Speed',
                          'Resource Utilization', 'Problem Success Matrix'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}]]
        )
        
        # 1. Success rate by model
        model_success = self.problem_results.groupby('model')['success'].mean() * 100
        fig.add_trace(
            go.Bar(x=model_success.index, y=model_success.values, name='Success Rate'),
            row=1, col=1
        )
        
        # 2. Duration distribution
        fig.add_trace(
            go.Histogram(x=self.problem_results['duration_seconds'], name='Duration', nbinsx=30),
            row=1, col=2
        )
        
        # 3. Pass@k rates
        if self.pass_at_k:
            k_values = []
            rates = []
            for k_str in sorted(self.pass_at_k.keys(), key=int):
                k_values.append(int(k_str))
                rates.append(self.pass_at_k[k_str]['pass_rate'] * 100)
            
            fig.add_trace(
                go.Scatter(x=k_values, y=rates, mode='lines+markers', name='Pass@k'),
                row=2, col=1
            )
        
        # 4. TPS boxplot
        if 'tps' in self.problem_results.columns:
            for model in self.problem_results['model'].unique():
                model_data = self.problem_results[self.problem_results['model'] == model]['tps'].dropna()
                fig.add_trace(
                    go.Box(y=model_data, name=model),
                    row=2, col=2
                )
        
        # 5. Resource utilization
        if self.system_metrics is not None and not self.system_metrics.empty:
            if 'gpu_util_mean' in self.system_metrics.columns:
                gpu_util = self.system_metrics.groupby('model')['gpu_util_mean'].mean()
                fig.add_trace(
                    go.Bar(x=gpu_util.index, y=gpu_util.values, name='GPU Utilization'),
                    row=3, col=1
                )
        
        # 6. Problem success matrix (simplified)
        pivot = self.problem_results.pivot_table(
            index='problem_id',
            columns='model',
            values='success',
            aggfunc='mean'
        )
        
        # Sample problems for visualization (too many makes it unreadable)
        # Take every nth problem to show full range
        step = max(1, len(pivot) // 50)  # Show about 50 problems
        sample_problems = pivot.iloc[::step]
        
        fig.add_trace(
            go.Heatmap(
                z=sample_problems.values,
                x=sample_problems.columns,
                y=sample_problems.index,
                colorscale='RdYlGn',
                zmid=0.5
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Benchmark Analysis Dashboard",
            title_font_size=20
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (s)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="k", row=2, col=1)
        fig.update_yaxes(title_text="Pass Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="TPS", row=2, col=2)
        fig.update_xaxes(title_text="Model", row=3, col=1)
        fig.update_yaxes(title_text="GPU Util (%)", row=3, col=1)
        fig.update_xaxes(title_text="Model", row=3, col=2)
        fig.update_yaxes(title_text="Problem ID", row=3, col=2)
        
        # Save interactive dashboard
        dashboard_file = self.output_dir / 'reports' / 'interactive_dashboard.html'
        fig.write_html(dashboard_file)
        
        print(f"  Saved interactive dashboard to {dashboard_file}")
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests between models."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for statistical tests")
            return
        
        print("\nPerforming statistical tests...")
        
        results = []
        
        # Compare success rates between models
        models = self.problem_results['model'].unique()
        if len(models) >= 2:
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1_success = self.problem_results[self.problem_results['model'] == models[i]]['success']
                    model2_success = self.problem_results[self.problem_results['model'] == models[j]]['success']
                    
                    # Chi-square test for success rates
                    if len(model1_success) > 0 and len(model2_success) > 0:
                        model_labels = ['model1']*len(model1_success) + ['model2']*len(model2_success)
                        success_values = pd.concat([model1_success, model2_success])
                        
                        try:
                            contingency_table = pd.crosstab(
                                pd.Series(model_labels),
                                success_values
                            )
                            
                            if contingency_table.size > 0:
                                chi2, p_value, _, _ = scipy_stats.chi2_contingency(contingency_table)
                                
                                results.append({
                                    'comparison': f"{models[i]} vs {models[j]}",
                                    'metric': 'success_rate',
                                    'test': 'chi-square',
                                    'p_value': p_value,
                                    'significant': p_value < 0.05
                                })
                        except Exception as e:
                            print(f"    Warning: Could not perform chi-square test for {models[i]} vs {models[j]}: {e}")
            
            # Compare performance metrics
            if 'duration_seconds' in self.problem_results.columns:
                for i in range(len(models)):
                    for j in range(i+1, len(models)):
                        model1_duration = self.problem_results[self.problem_results['model'] == models[i]]['duration_seconds'].dropna()
                        model2_duration = self.problem_results[self.problem_results['model'] == models[j]]['duration_seconds'].dropna()
                        
                        if len(model1_duration) > 0 and len(model2_duration) > 0:
                            # Mann-Whitney U test for duration
                            statistic, p_value = scipy_stats.mannwhitneyu(model1_duration, model2_duration)
                            
                            results.append({
                                'comparison': f"{models[i]} vs {models[j]}",
                                'metric': 'duration',
                                'test': 'mann-whitney-u',
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
        
        # Save statistical test results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.output_dir / 'reports' / 'tables' / 'statistical_tests.csv', index=False)
            
            print("  Statistical test results:")
            for result in results:
                sig = "***" if result['significant'] else "ns"
                print(f"    {result['comparison']} ({result['metric']}): p={result['p_value']:.4f} {sig}")
        
        return results
    
    def create_summary_tables(self):
        """Create summary tables for the report."""
        if self.problem_results is None or self.problem_results.empty:
            print("No data for summary tables")
            return
        
        print("\nCreating summary tables...")
        
        tables_dir = self.output_dir / 'reports' / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall performance summary
        summary = self.problem_results.groupby('model').agg({
            'success': ['count', 'mean'],
            'duration_seconds': ['mean', 'median', 'std'],
            'test_pass_rate': 'mean',
            'total_tokens': 'mean'
        }).round(3)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.to_csv(tables_dir / 'performance_summary.csv')
        
        # 2. Pass@k summary
        if self.pass_at_k:
            pass_k_data = []
            for k_str in sorted(self.pass_at_k.keys(), key=int):
                k_data = self.pass_at_k[k_str]
                row = {
                    'k': int(k_str),
                    'overall_pass_rate': f"{k_data['pass_rate']*100:.1f}%"
                }
                
                # Add model-specific rates
                for model_key, stats in k_data.get('by_model', {}).items():
                    row[f'{model_key}_pass_rate'] = f"{stats['pass_rate']*100:.1f}%"
                
                pass_k_data.append(row)
            
            pass_k_df = pd.DataFrame(pass_k_data)
            pass_k_df.to_csv(tables_dir / 'pass_at_k_summary.csv', index=False)
        
        # 3. Resource utilization summary
        if self.system_metrics is not None and not self.system_metrics.empty:
            # Check which columns are available
            agg_dict = {}
            if 'gpu_util_mean' in self.system_metrics.columns:
                agg_dict['gpu_util_mean'] = 'mean'
            if 'gpu_memory_max_mb' in self.system_metrics.columns:
                agg_dict['gpu_memory_max_mb'] = 'max'
            if 'gpu_power_mean_w' in self.system_metrics.columns:
                agg_dict['gpu_power_mean_w'] = 'mean'
            if 'cpu_util_mean' in self.system_metrics.columns:
                agg_dict['cpu_util_mean'] = 'mean'
            if 'memory_util_mean' in self.system_metrics.columns:
                agg_dict['memory_util_mean'] = 'mean'
            
            if agg_dict:
                resource_summary = self.system_metrics.groupby('model').agg(agg_dict).round(2)
                resource_summary.to_csv(tables_dir / 'resource_summary.csv')
        
        print("  Saved summary tables")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting benchmark analysis...")
        
        if self.problem_results is None or self.problem_results.empty:
            print("Error: No problem results data found. Run process_benchmark_data.py first.")
            return
        
        # Create subdirectories for different plot types
        (self.figures_dir / 'performance').mkdir(exist_ok=True)
        (self.figures_dir / 'comparisons').mkdir(exist_ok=True)
        (self.figures_dir / 'resources').mkdir(exist_ok=True)
        
        # Generate all visualizations and analyses
        self.create_performance_distributions()
        self.create_model_comparison_plots()
        self.create_pass_at_k_visualization()
        self.create_resource_utilization_plots()
        self.create_problem_difficulty_heatmap()
        self.create_interactive_dashboard()
        self.perform_statistical_tests()
        self.create_summary_tables()
        
        print("\nAnalysis complete! Check the output directory for results.")


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
    
    analyzer = BenchmarkAnalyzer(processed_dir, output_dir)
    analyzer.run_analysis()