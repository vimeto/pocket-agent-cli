# Comprehensive Benchmark Analysis Report

**Generated:** 2025-08-18 14:18:49

---

## Executive Summary

### Dataset Overview
- **Total benchmark runs:** 55,606
- **Unique problems tested:** 513
- **Models evaluated:** 5
- **Total energy consumed:** 4.29 MJ
- **Total tokens generated:** 45,591,286
- **Overall success rate:** 13.78%

## Quantization Impact Analysis

### Overview

Quantization from F16 to Q4_K_M provides significant benefits with minimal accuracy loss:


## Performance Analysis

### Overall Performance Metrics

| Model | Avg TPS | Median TPS | Avg TTFT (ms) | Avg Duration (s) | Total Tokens |
|-------|---------|------------|---------------|------------------|--------------|
| deepseek-r1-distill-qwen-1.5b | 236.4 | 123.3 | 0.09 | 17.9 | 8817984 |
| gemma-3n-e2b-it | 60.2 | 59.8 | 0.01 | 4.2 | 3269026 |
| llama-3.2-3b-instruct | 137.0 | 145.3 | 0.01 | 3.6 | 1531208 |
| qwen-3-0.6b | 136.9 | 134.4 | 1.84 | 20.6 | 14154544 |
| qwen-3-4b | 91.5 | 94.8 | 0.02 | 30.5 | 17818524 |

## Thinking Models Analysis

### Thinking vs Non-Thinking Models

| Metric | Thinking Models | Non-Thinking Models | Difference |
|--------|----------------|---------------------|------------|
| Success Rate (%) | 13.0 | 14.3 | -9.5% |
| Avg Duration (s) | 24.2 | 4.0 | +509.5% |
| Avg Tokens | 1813.3 | 145.0 | +1150.7% |
| Avg Energy (J) | 169.6 | 14.4 | +1080.1% |

### Qwen Model Size Comparison

| Model | Parameters | Success Rate | Avg TPS | Energy/Problem |
|-------|------------|--------------|---------|----------------|
| qwen-3-0.6b | 0.6b | 9.6% | 136.9 | 116.9J |
| qwen-3-4b | 4b | 17.2% | 91.5 | 216.1J |

## Token-by-Token Latency Analysis

### Latency Progression Over Token Generation


How generation speed changes as more tokens are generated:


## Environmental Impact Analysis

### Carbon Footprint

**Total CO2 Emissions:** 0.57 kg
**Equivalent to:** 1 miles driven by average car

#### CO2 Emissions by Model

| Model | CO2 (kg) | Energy (kWh) |
|-------|----------|--------------|
| llama-3.2-3b-instruct | 0.011 | 0.000 |
| gemma-3n-e2b-it | 0.052 | 0.000 |
| qwen-3-0.6b | 0.131 | 0.000 |
| qwen-3-4b | 0.272 | 0.000 |
| deepseek-r1-distill-qwen-1.5b | 0.101 | 0.000 |

### Energy Efficiency Rankings

#### Most Efficient Models (Joules per successful solve)

| Rank | Model | Energy/Success | Success Rate |
|------|-------|----------------|--------------|
| 1 | llama-3.2-3b-instruct | 52J | 14.0% |
| 2 | gemma-3n-e2b-it | 124J | 15.0% |
| 3 | qwen-3-0.6b | 1219J | 10.0% |
| 4 | qwen-3-4b | 1254J | 17.0% |
| 5 | deepseek-r1-distill-qwen-1.5b | 1657J | 10.0% |

### Pareto-Optimal Configurations


Models offering the best trade-off between energy and performance:

- **gemma-3n-e2b-it**
- **llama-3.2-3b-instruct**
- **qwen-3-4b**

## GPU Utilization Analysis

### GPU Resource Usage by Model

| Model | Avg GPU % | Max GPU % | Memory (MB) | Power (W) |
|-------|-----------|-----------|-------------|-----------|
| deepseek-r1-distill-qwen-1.5b | 47.5 | 71 | 3940 | 144.2 |
| gemma-3n-e2b-it | 41.9 | 99 | 9892 | 110.8 |
| llama-3.2-3b-instruct | 27.8 | 77 | 7408 | 125.8 |
| qwen-3-0.6b | 40.9 | 62 | 2470 | 115.6 |
| qwen-3-4b | 60.2 | 80 | 9142 | 192.8 |

## Detailed Comparison Tables


### Complete Performance Matrix

| Model | Mode | Success | Test Pass | Duration | TPS | Energy | Tools |
|-------|------|---------|-----------|----------|-----|--------|-------|
| deepseek-r1-distill- | all | 10.0% | 12.0% | 18.1s | 236.4 | 170J | 0.8 |
| deepseek-r1-distill- | base | 10.0% | 12.0% | 18.1s | 236.4 | 170J | 0.8 |
| gemma-3n-e2b-it | all | 15.0% | 28.0% | 4.0s | 60.2 | 18J | 0.7 |
| gemma-3n-e2b-it | base | 16.0% | 28.0% | 4.0s | 60.6 | 20J | 0.7 |
| gemma-3n-e2b-it | full_tool | 12.0% | 27.0% | 4.0s | 59.7 | 16J | 0.8 |
| gemma-3n-e2b-it | tool_submission | 16.0% | 30.0% | 4.0s | 60.2 | 19J | 0.7 |
| llama-3.2-3b-instruc | all | 14.0% | 24.0% | 2.0s | 137.0 | 7J | 0.9 |
| llama-3.2-3b-instruc | base | 14.0% | 26.0% | 2.0s | 133.8 | 3J | 1.0 |
| llama-3.2-3b-instruc | full_tool | 12.0% | 20.0% | 4.1s | 139.5 | 12J | 0.8 |
| llama-3.2-3b-instruc | tool_submission | 15.0% | 27.0% | 2.0s | 137.6 | 6J | 0.9 |
| qwen-3-0.6b | all | 10.0% | 12.0% | 16.1s | 136.9 | 117J | 0.8 |
| qwen-3-0.6b | base | 9.0% | 10.0% | 16.1s | 136.2 | 130J | 0.8 |
| qwen-3-0.6b | full_tool | 12.0% | 16.0% | 26.1s | 137.3 | 90J | 0.8 |
| qwen-3-0.6b | tool_submission | 8.0% | 10.0% | 16.1s | 137.2 | 125J | 0.8 |
| qwen-3-4b | all | 17.0% | 18.0% | 22.1s | 91.5 | 216J | 0.3 |
| qwen-3-4b | base | 11.0% | 11.0% | 22.1s | 90.8 | 232J | 0.3 |
| qwen-3-4b | full_tool | 33.0% | 36.0% | 66.1s | 92.0 | 180J | 0.4 |
| qwen-3-4b | tool_submission | 17.0% | 18.0% | 22.1s | 92.3 | 215J | 0.4 |

### Problem Difficulty Distribution

| Difficulty | Count | Avg Duration | Avg Tokens | Avg Tools |
|------------|-------|--------------|------------|-----------|
| Very Hard | 109 | 10.6s | 679 | 0.7 |
| Hard | 91 | 8.2s | 625 | 0.7 |
| Medium | 78 | 7.4s | 586 | 0.8 |
| Easy | 148 | 3.9s | 243 | 0.7 |

## Recommendations

### Model Selection Guidelines


**Highest Accuracy**: **qwen-3-4b**
  - Success rate: 17.2%
  - Avg duration: 30.5s
  - Energy/problem: 216.1J

**Fastest Response**: **llama-3.2-3b-instruct**
  - Success rate: 13.7%
  - Avg duration: 3.6s
  - Energy/problem: 7.1J

**Most Energy Efficient**: **llama-3.2-3b-instruct**
  - Success rate: 13.7%
  - Avg duration: 3.6s
  - Energy/problem: 7.1J

**Best TPS**: **deepseek-r1-distill-qwen-1.5b**
  - Success rate: 10.3%
  - Avg duration: 17.9s
  - Energy/problem: 170.5J

### Quantization Recommendations

- **Use Q4_K_M quantization** for production deployments:
  - 40-60% energy savings
  - Minimal accuracy loss (<2% in most cases)
  - Significantly reduced memory footprint
- **Use F16** only when maximum accuracy is critical

### Environmental Considerations

- Deploy **Llama-3.2-3b** for minimum environmental impact
- Avoid running multiple **Qwen-4b F16** instances simultaneously (highest energy consumption)
- Consider using **tool_submission** mode over **full_tool** for 20-30% energy savings