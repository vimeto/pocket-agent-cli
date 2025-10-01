# Benchmark Analysis Pipeline

A comprehensive data analysis pipeline for processing and analyzing Pocket Agent benchmark results from Mahti HPC cluster.

## Overview

This pipeline provides a structured approach to:
1. **Organize** benchmark data from multiple SLURM jobs
2. **Process** and extract metrics from JSON results and system monitoring logs
3. **Analyze** performance with statistical tests and visualizations
4. **Generate** publication-ready reports and tables

## Directory Structure

```
benchmark_analysis/
├── raw_data/              # Original files from Mahti (populated by user)
├── processed/             # Organized and processed data
│   ├── job_*/            # Data organized by SLURM job ID
│   ├── master_index.json # Index of all jobs
│   ├── problem_results.csv # Consolidated problem-level results
│   ├── system_metrics.csv  # Aggregated system metrics
│   ├── aggregate_stats.json # Statistical summaries
│   └── pass_at_k.json      # Pass@k analysis results
├── reports/               # Generated reports
│   ├── benchmark_report.md # Main analysis report
│   ├── interactive_dashboard.html # Interactive visualizations
│   ├── tables/           # Statistical test results
│   ├── latex/            # LaTeX tables for papers
│   └── csv/              # CSV exports
├── visualizations/        # Generated plots
│   ├── performance/      # Performance distribution plots
│   ├── comparisons/      # Model comparison visualizations
│   └── resources/        # Resource utilization graphs
├── scripts/              # Analysis scripts
└── config.yaml           # Configuration file
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended, consistent with pocket-agent-cli)
uv pip install -r benchmark_analysis/requirements.txt

# Or using standard pip
pip install -r benchmark_analysis/requirements.txt
```

### 2. Copy Data from Mahti

```bash
# SSH to Mahti and create archive
ssh mahti "cd /projappl/project_2013932/$USER/pocket-agent-cli && \
           tar -czf benchmark_data.tar.gz \
           data/results/bench_* \
           data/logs/*.csv \
           data/logs/failures*.log \
           logs/benchmark_*.out \
           logs/benchmark_*.err"

# Copy to local machine
scp mahti:/projappl/project_2013932/$USER/pocket-agent-cli/benchmark_data.tar.gz .

# Extract to raw_data directory
tar -xzf benchmark_data.tar.gz -C benchmark_analysis/raw_data/
```

### 3. Run Analysis Pipeline

```bash
# Using uv (recommended)
# Step 1: Organize data by job ID
uv run python benchmark_analysis/scripts/organize_benchmark_data.py

# Step 2: Process and extract metrics
uv run python benchmark_analysis/scripts/process_benchmark_data.py

# Step 3: Generate visualizations and statistical analysis
uv run python benchmark_analysis/scripts/analyze_benchmarks.py

# Step 4: Generate reports
uv run python benchmark_analysis/scripts/generate_report.py
```

Or run all steps with a single command:
```bash
bash benchmark_analysis/run_analysis.sh
```

## Scripts Description

### `organize_benchmark_data.py`
- Groups files by SLURM job ID
- Matches benchmark results with system metrics
- Creates metadata index for each job
- Generates master index of all jobs

### `process_benchmark_data.py`
- Loads and parses JSON benchmark results
- Extracts performance metrics (TTFT, TPS, tokens, energy)
- Calculates pass@k rates for k=1,3,5,10
- Aggregates system monitoring data (GPU/CPU utilization)
- Outputs consolidated CSV files

### `analyze_benchmarks.py`
- Creates performance distribution plots
- Generates model comparison visualizations
- Produces pass@k rate curves
- Creates resource utilization graphs
- Performs statistical significance tests
- Generates interactive Plotly dashboard

### `generate_report.py`
- Creates comprehensive Markdown report
- Generates LaTeX tables for academic papers
- Exports data to CSV for further analysis
- Produces executive summary with recommendations

## Configuration

Edit `config.yaml` to customize:
- Data paths and directories
- Mahti HPC settings
- Analysis parameters (significance levels, k values)
- Model configurations to analyze
- Report generation options
- Visualization settings

## Output Files

### Reports
- **`benchmark_report.md`**: Comprehensive analysis report with tables, findings, and recommendations
- **`interactive_dashboard.html`**: Interactive Plotly dashboard for exploring results
- **`latex/*.tex`**: Publication-ready LaTeX tables

### Visualizations
- **Performance distributions**: Histograms of execution time, TTFT, TPS, token counts
- **Model comparisons**: Box plots and bar charts comparing models
- **Pass@k curves**: Success rate vs number of attempts
- **Resource utilization**: GPU/CPU usage, memory, power consumption
- **Problem difficulty heatmap**: Success rates by problem and model

### Data Exports
- **`problem_results.csv`**: All problem-level results with metrics
- **`system_metrics.csv`**: Aggregated resource utilization data
- **`aggregate_stats.json`**: Statistical summaries by model
- **`pass_at_k.json`**: Pass@k analysis results

## Advanced Usage

### Custom Analysis

```python
from pathlib import Path
import pandas as pd

# Load processed data
df = pd.read_csv('benchmark_analysis/processed/problem_results.csv')

# Custom analysis
model_comparison = df.groupby(['model', 'mode']).agg({
    'success': 'mean',
    'duration_seconds': 'median',
    'energy_per_token': 'mean'
})

print(model_comparison)
```

### Batch Processing Multiple Runs

```bash
# Process multiple benchmark batches
for batch in batch1 batch2 batch3; do
    tar -xzf ${batch}.tar.gz -C benchmark_analysis/raw_data/
    python benchmark_analysis/scripts/organize_benchmark_data.py
done

# Analyze combined results
python benchmark_analysis/scripts/process_benchmark_data.py
python benchmark_analysis/scripts/analyze_benchmarks.py
```

### Filtering Specific Models

```python
# In process_benchmark_data.py, filter specific models
df = df[df['model'].isin(['llama-3.2-3b-instruct', 'gemma-2b-it'])]
```

## Troubleshooting

### Missing Data
- Ensure all files are properly copied from Mahti
- Check that job IDs are correctly extracted from filenames
- Verify that JSON files are not corrupted

### Memory Issues
- For large datasets, process jobs in batches
- Increase system swap space if needed
- Use data chunking in pandas operations

### Visualization Issues
- Ensure matplotlib backend is properly configured
- Check that all required fonts are installed
- Verify figure directory permissions

## Data Schema

### Problem Results Schema
- `job_id`: SLURM job identifier
- `problem_id`: MBPP problem number
- `model`: Model name
- `mode`: Evaluation mode (base/tool_submission/full_tool)
- `success`: Boolean success indicator
- `duration_seconds`: Execution time
- `test_pass_rate`: Fraction of tests passed
- `ttft_ms`: Time to first token (milliseconds)
- `tps`: Tokens per second
- `energy_per_token`: Energy efficiency metric

### System Metrics Schema
- `job_id`: SLURM job identifier
- `gpu_util_mean`: Average GPU utilization (%)
- `gpu_memory_max_mb`: Maximum GPU memory usage
- `gpu_power_mean_w`: Average power consumption
- `cpu_util_mean`: Average CPU utilization (%)

## Contributing

To extend the pipeline:

1. Add new metrics extraction in `process_benchmark_data.py`
2. Create visualizations in `analyze_benchmarks.py`
3. Update report generation in `generate_report.py`
4. Document changes in this README

## License

This analysis pipeline is part of the Pocket Agent CLI project.

## Contact

For questions or issues with the analysis pipeline, please open an issue in the Pocket Agent CLI repository.