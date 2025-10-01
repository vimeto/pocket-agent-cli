#!/bin/bash
# Automated benchmark analysis pipeline runner

set -e  # Exit on error

echo "======================================"
echo "Pocket Agent Benchmark Analysis Pipeline"
echo "======================================"
echo ""

# Check if virtual environment should be activated
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies using uv
echo "Checking dependencies with uv..."
uv pip install -q -r benchmark_analysis/requirements.txt

# Step 1: Organize data
echo ""
echo "Step 1: Organizing benchmark data..."
echo "-------------------------------------"
uv run python benchmark_analysis/scripts/organize_benchmark_data.py

# Step 2: Process data
echo ""
echo "Step 2: Processing benchmark data..."
echo "-------------------------------------"
uv run python benchmark_analysis/scripts/process_benchmark_data.py

# Step 3: Analyze and visualize
echo ""
echo "Step 3: Analyzing and creating visualizations..."
echo "------------------------------------------------"
uv run python benchmark_analysis/scripts/analyze_benchmarks.py

# Step 4: Generate reports
echo ""
echo "Step 4: Generating reports..."
echo "-----------------------------"
uv run python benchmark_analysis/scripts/generate_report.py

echo ""
echo "======================================"
echo "Analysis Complete!"
echo "======================================"
echo ""
echo "Generated outputs:"
echo "  - Reports: benchmark_analysis/reports/"
echo "  - Visualizations: benchmark_analysis/visualizations/"
echo "  - Processed data: benchmark_analysis/processed/"
echo ""
echo "View the main report: benchmark_analysis/reports/benchmark_report.md"
echo "Open interactive dashboard: benchmark_analysis/reports/interactive_dashboard.html"