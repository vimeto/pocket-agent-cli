#!/bin/bash
# Fetch benchmark data from Mahti HPC cluster

set -e

echo "======================================"
echo "Fetching Benchmark Data from Mahti"
echo "======================================"
echo ""

# Configuration
MAHTI_HOST=${MAHTI_HOST:-"mahti"}
PROJECT_ID=${PROJECT_ID:-"2013932"}
REMOTE_USER=${REMOTE_USER:-"$USER"}
REMOTE_BASE="/projappl/project_${PROJECT_ID}/${REMOTE_USER}/pocket-agent-cli"

# Local paths
LOCAL_DIR="benchmark_analysis/raw_data"

# Check if ssh to Mahti works
echo "Testing SSH connection to Mahti..."
if ! ssh -o ConnectTimeout=5 $MAHTI_HOST "echo 'Connection successful'" 2>/dev/null; then
    echo "Error: Cannot connect to Mahti. Please check your SSH configuration."
    echo "Make sure you have an entry for 'mahti' in your ~/.ssh/config file."
    exit 1
fi

echo "Connection successful!"
echo ""

# Create archive on Mahti
echo "Creating archive on Mahti..."
ssh $MAHTI_HOST "cd $REMOTE_BASE && \
    echo 'Archiving benchmark data...' && \
    tar -czf benchmark_data.tar.gz \
    data/results/bench_* \
    data/logs/*.csv \
    data/logs/failures*.log \
    logs/benchmark_*.out \
    logs/benchmark_*.err 2>/dev/null || true && \
    echo 'Archive created: benchmark_data.tar.gz' && \
    ls -lh benchmark_data.tar.gz"

echo ""
echo "Copying archive to local machine..."
scp ${MAHTI_HOST}:${REMOTE_BASE}/benchmark_data.tar.gz .

echo ""
echo "Extracting data..."
mkdir -p $LOCAL_DIR
tar -xzf benchmark_data.tar.gz -C $LOCAL_DIR/

# Get file count and size
FILE_COUNT=$(find $LOCAL_DIR -type f | wc -l)
TOTAL_SIZE=$(du -sh $LOCAL_DIR | cut -f1)

echo ""
echo "======================================"
echo "Data Transfer Complete!"
echo "======================================"
echo "Files extracted: $FILE_COUNT"
echo "Total size: $TOTAL_SIZE"
echo "Location: $LOCAL_DIR"
echo ""
echo "To analyze the data, run:"
echo "  bash benchmark_analysis/run_analysis.sh"
echo ""

# Optionally clean up the archive on Mahti
read -p "Remove archive from Mahti? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh $MAHTI_HOST "rm -f ${REMOTE_BASE}/benchmark_data.tar.gz"
    echo "Remote archive removed."
fi

# Optionally remove local archive
read -p "Remove local archive? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f benchmark_data.tar.gz
    echo "Local archive removed."
fi