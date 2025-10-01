#!/bin/bash
# Round-robin benchmark driver for Pocket Agent on macOS (M2 Max)
# Runs small MBPP batches across all models, modes, and precisions.

set -euo pipefail

# Configuration
BATCH_SIZE=5                # Number of MBPP problems per run
START_PROBLEM=10            # MBPP problems start at ID 10
END_PROBLEM=509             # Inclusive upper bound
NUM_SAMPLES=5               # Samples per problem (balance accuracy vs time)
TEMPERATURE=0.7             # Sampling temperature
CONTEXT=8192                # Context length for desktop runs
BATTERY_THRESHOLD=90        # Pause if battery percentage drops below this
WAIT_BETWEEN_RUNS=00        # Seconds to rest between batches

MODE="all"                  # Run base + tool_submission + full_tool each invocation
MODELS=(
  "gemma-3n-e2b-it"
  "llama-3.2-3b-instruct"
  "qwen-3-4b"
  "qwen-3-0.6b"
  "deepseek-r1-distill-qwen-1.5b"
)
QUANTIZATIONS=("Q4_K_M" "F16")

# Helpers --------------------------------------------------------
get_battery_level() {
  pmset -g batt | grep -Eo "[0-9]+%" | grep -Eo "[0-9]+"
}

wait_for_battery() {
  while true; do
    local level
    level=$(get_battery_level)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Battery level: ${level}%"
    if [[ ${level} -ge ${BATTERY_THRESHOLD} ]]; then
      break
    fi
    echo "Battery below ${BATTERY_THRESHOLD}%. Waiting 60s..."
    sleep 60
  done
}

next_problem_block() {
  local start=$1
  local batch=$2

  local -a ids=()
  for ((i=0; i<batch; i++)); do
    local offset=$(( (start - START_PROBLEM + i) % (END_PROBLEM - START_PROBLEM + 1) ))
    ids+=($((START_PROBLEM + offset)))
  done
  local IFS=','
  echo "${ids[*]}"
}

prune_docker_sandboxes() {
  docker ps -a --filter "label=pocket_agent_cli_sandbox" -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

# Main loop ------------------------------------------------------
current_problem=${START_PROBLEM}
combo_index=0

trap 'echo "\n[INFO] Stopping benchmark loop."; exit 0' SIGINT SIGTERM

echo "Starting round-robin benchmark loop. Press Ctrl+C to stop."

while true; do
  wait_for_battery

  model=${MODELS[$((combo_index % ${#MODELS[@]}))]}
  quant=${QUANTIZATIONS[$(((combo_index / ${#MODELS[@]}) % ${#QUANTIZATIONS[@]}))]}

  # After exhausting one full quantization pass, advance combo index differently
  combo_index=$(( (combo_index + 1) % (${#MODELS[@]} * ${#QUANTIZATIONS[@]}) ))

  problems=$(next_problem_block ${current_problem} ${BATCH_SIZE} ${END_PROBLEM})

  echo "\n============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model: ${model} | Quant: ${quant} | Problems: ${problems}"
  echo "============================================================"

  export DEBUG_BENCHMARK=true
  if ! uv run python -m pocket_agent_cli.cli benchmark \
        --model "${model}" \
        --model-version "${quant}" \
        --mode "${MODE}" \
        --problems "${problems}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --context-length "${CONTEXT}" \
        --parallel 1; then
    echo "[WARN] Benchmark command failed. Retrying this configuration after 30s..."
    sleep 30
    continue
  fi

  prune_docker_sandboxes

  # Advance problem pointer
  current_problem=$((current_problem + BATCH_SIZE))
  if (( current_problem > END_PROBLEM )); then
    current_problem=$START_PROBLEM
  fi

  echo "Completed batch. Next sweep will start at problem ${current_problem}."
  sleep ${WAIT_BETWEEN_RUNS}
done
