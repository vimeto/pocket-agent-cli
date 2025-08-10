#!/bin/bash

# Battery threshold percentage
BATTERY_THRESHOLD=75

# Number of problems to run at once
BATCH_SIZE=5
222222222
# Total number of problems
TOTAL_PROBLEMS=509

# Starting problem index
START_INDEX=225

# Check if resuming from a specific index
if [ "$1" ]; then
    START_INDEX=$1
    echo "Resuming from problem index: $START_INDEX"
fi

# Function to get battery percentage on macOS
get_battery_level() {
    pmset -g batt | grep -Eo "[0-9]+%" | grep -Eo "[0-9]+"
}

# remove all docker containers with tag pocket_agent_cli_sandbox
prune_docker_containers() {
    docker ps -a --filter "label=pocket_agent_cli_sandbox" -q | xargs -r docker rm -f
    echo "Pruned docker containers"
}

# Function to wait for battery to charge
wait_for_battery() {
    while true; do
        battery_level=$(get_battery_level)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Battery level: ${battery_level}%"

        if [ "$battery_level" -gt "$BATTERY_THRESHOLD" ]; then
            echo "Battery level sufficient (>${BATTERY_THRESHOLD}%). Continuing..."
            break
        else
            echo "Battery level low (<=${BATTERY_THRESHOLD}%). Waiting 60 seconds..."
            sleep 60
        fi
    done
}

# Main loop
current_index=$START_INDEX

while [ $current_index -lt $TOTAL_PROBLEMS ]; do
    # Check battery before starting
    wait_for_battery

    # Calculate end index for this batch
    end_index=$((current_index + BATCH_SIZE - 1))
    if [ $end_index -ge $TOTAL_PROBLEMS ]; then
        end_index=$((TOTAL_PROBLEMS - 1))
    fi

    # Build problem IDs list
    problem_ids=""
    for ((i=current_index; i<=end_index; i++)); do
        if [ -z "$problem_ids" ]; then
            problem_ids="$i"
        else
            problem_ids="${problem_ids},$i"
        fi
    done

    echo ""
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running problems: $problem_ids"
    echo "============================================"

    # Run the benchmark command
    export DEBUG_BENCHMARK=true
    uv run python -m pocket_agent_cli.cli benchmark \
        --model gemma-3n-e2b-it \
        --mode all \
        --problems "$problem_ids" \
        --num-samples 10

    # prune docker containers
    prune_docker_containers

    # Check exit code, and retry if it fails
    if [ $? -ne 0 ]; then
        echo "Error running benchmark. Retrying..."
        sleep 10
        continue
    fi

    # Update current index
    current_index=$((end_index + 1))

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed batch. Progress: $current_index/$TOTAL_PROBLEMS problems"

    # If we're not done, wait before next batch
    if [ $current_index -lt $TOTAL_PROBLEMS ]; then
        echo "Waiting 10 seconds before checking battery for next batch..."
        sleep 10
    fi
done

echo ""
echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All benchmarks completed!"
echo "============================================"
