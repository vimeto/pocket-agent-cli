# Pocket Agent Benchmark Analysis Report

**Generated:** 2025-08-18 13:48:24

---

## Executive Summary

### Key Findings

- **Total benchmark runs:** 55606
- **Unique problems evaluated:** 513
- **Overall success rate:** 13.8%
- **Average test pass rate:** 21.9%

### Performance Metrics

- **Median execution time:** 4.10 seconds
- **95th percentile execution time:** 30.06 seconds

## Pass@k Analysis

Pass@k measures the probability of generating at least one correct solution within k attempts.

| k | Pass Rate | Improvement |
|---|-----------|-------------|
| 1 | 47.8% | baseline |
| 3 | 56.5% | +8.7% |
| 5 | 59.1% | +2.6% |
| 10 | 62.1% | +3.0% |

## Model Performance Comparison

| Model Configuration | Success Rate | Test Pass Rate | Median Duration | Mean TPS |
|-------------------|--------------|----------------|-----------------|----------|
| deepseek-r1-distill-qwen-1.5b_all | 10.3% | 12.1% | 18.05s | nan |
| deepseek-r1-distill-qwen-1.5b_base | 10.3% | 12.1% | 18.05s | nan |
| gemma-3n-e2b-it_all | 14.7% | 28.3% | 4.05s | nan |
| gemma-3n-e2b-it_base | 15.6% | 27.7% | 4.05s | nan |
| gemma-3n-e2b-it_full_tool | 12.4% | 27.4% | 4.05s | nan |
| gemma-3n-e2b-it_tool_submission | 15.6% | 29.6% | 4.05s | nan |
| llama-3.2-3b-instruct_all | 13.7% | 24.4% | 2.05s | nan |
| llama-3.2-3b-instruct_base | 13.6% | 26.0% | 2.05s | nan |
| llama-3.2-3b-instruct_full_tool | 12.5% | 20.3% | 4.06s | nan |
| llama-3.2-3b-instruct_tool_submission | 15.2% | 27.2% | 2.05s | nan |
| qwen-3-0.6b_all | 9.6% | 11.8% | 16.05s | nan |
| qwen-3-0.6b_base | 8.7% | 10.1% | 16.05s | nan |
| qwen-3-0.6b_full_tool | 12.0% | 16.5% | 26.06s | nan |
| qwen-3-0.6b_tool_submission | 8.4% | 9.7% | 16.05s | nan |
| qwen-3-4b_all | 17.2% | 18.4% | 22.06s | nan |
| qwen-3-4b_base | 10.8% | 11.2% | 22.06s | nan |
| qwen-3-4b_full_tool | 32.9% | 36.4% | 66.07s | nan |
| qwen-3-4b_tool_submission | 17.3% | 18.1% | 22.06s | nan |

## Resource Utilization

### GPU Metrics

## Problem Difficulty Analysis

### Hardest Problems (Lowest Success Rate)

| Problem ID | Success Rate |
|------------|--------------|
| 1 | 0.0% |
| 31 | 0.0% |
| 60 | 0.0% |
| 77 | 0.0% |
| 110 | 0.0% |
| 124 | 0.0% |
| 136 | 0.0% |
| 138 | 0.0% |
| 139 | 0.0% |
| 143 | 0.0% |

## Statistical Significance Tests

| Comparison | Metric | Test | p-value | Significant |
|------------|--------|------|---------|-------------|
| gemma-3n-e2b-it vs qwen-3-4b | success_rate | chi-square | 0.0000 | Yes ✓ |
| gemma-3n-e2b-it vs llama-3.2-3b-instruct | success_rate | chi-square | 0.0001 | Yes ✓ |
| gemma-3n-e2b-it vs deepseek-r1-distill-qwen-1.5b | success_rate | chi-square | 0.6497 | No |
| gemma-3n-e2b-it vs qwen-3-0.6b | success_rate | chi-square | 0.8112 | No |
| qwen-3-4b vs llama-3.2-3b-instruct | success_rate | chi-square | 1.0000 | No |
| qwen-3-4b vs deepseek-r1-distill-qwen-1.5b | success_rate | chi-square | 1.0000 | No |
| qwen-3-4b vs qwen-3-0.6b | success_rate | chi-square | 1.0000 | No |
| llama-3.2-3b-instruct vs deepseek-r1-distill-qwen-1.5b | success_rate | chi-square | 1.0000 | No |
| llama-3.2-3b-instruct vs qwen-3-0.6b | success_rate | chi-square | 1.0000 | No |
| gemma-3n-e2b-it vs qwen-3-4b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| gemma-3n-e2b-it vs llama-3.2-3b-instruct | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| gemma-3n-e2b-it vs deepseek-r1-distill-qwen-1.5b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| gemma-3n-e2b-it vs qwen-3-0.6b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| qwen-3-4b vs llama-3.2-3b-instruct | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| qwen-3-4b vs deepseek-r1-distill-qwen-1.5b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| qwen-3-4b vs qwen-3-0.6b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| llama-3.2-3b-instruct vs deepseek-r1-distill-qwen-1.5b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| llama-3.2-3b-instruct vs qwen-3-0.6b | duration | mann-whitney-u | 0.0000 | Yes ✓ |
| deepseek-r1-distill-qwen-1.5b vs qwen-3-0.6b | duration | mann-whitney-u | 0.0000 | Yes ✓ |

## Failure Analysis

### Failure Count by Model

| Model | Failed Runs |
|-------|-------------|
| deepseek-r1-distill-qwen-1.5b | 4028 |
| gemma-3n-e2b-it | 18476 |
| llama-3.2-3b-instruct | 9890 |
| qwen-3-0.6b | 7658 |
| qwen-3-4b | 7892 |

## Recommendations

Based on the benchmark analysis, the following recommendations are made:

- **Best performing configuration:** qwen-3-4b_full_tool with 32.9% success rate
- **Tool usage improves performance:** Tool-enabled modes show 110.8% improvement over base mode.

## Appendix

### Benchmark Configuration

- **Dataset:** MBPP (Mostly Basic Python Problems)
- **Total problems evaluated:** 513
- **Samples per problem:** 10 (for pass@k calculation)
- **Temperature:** 0.7
- **Evaluation modes:** base, tool_submission, full_tool

### Generated Visualizations

The following visualizations have been generated and saved:

- `visualizations/performance/distributions.png` - Performance metric distributions
- `visualizations/performance/pass_at_k.png` - Pass@k rate curves
- `visualizations/comparisons/model_comparison.png` - Model performance comparison
- `visualizations/comparisons/difficulty_heatmap.png` - Problem difficulty heatmap
- `visualizations/resources/utilization.png` - Resource utilization analysis
- `reports/interactive_dashboard.html` - Interactive Plotly dashboard

---

*This report was automatically generated by the Pocket Agent benchmark analysis pipeline.*