#!/bin/bash
#SBATCH --job-name=imobench
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# IMOBench Evaluation Script
# This script runs either IMOBench (answer benchmark) or IMOProofBench (grading benchmark)
#
# Usage examples:
# 1. Run IMOBench (answer benchmark):
#    sbatch slurm/launch_imobench.slurm --model-config gpt-5-mini --output-path outputs/gpt-5-mini.jsonl --final-answer
#
# 2. Run IMOProofBench (grading benchmark):
#    sbatch slurm/launch_imobench.slurm --model-config gpt-5-mini --output-path outputs/gpt-5-mini.jsonl
#
# 3. Run with overwrite (ignore cached results):
#    sbatch slurm/launch_imobench.slurm --model-config gpt-5-mini --output-path outputs/gpt-5-mini.jsonl --final-answer --overwrite
#
# Available model configs (see IMOBench/configs/models/):
# - openai/gpt-51.yaml
# - openai/gpt-5-mini.yaml
# - google/gemini-2.5-pro.yaml
# - google/gemini-3-pro.yaml
# - other/grok-41-fast.yaml
# - openai/oss-120b.yaml
# - openai/custom-oss-120b.yaml

set -x -e

# Record start time
SCRIPT_START_TIME=$(date +%s)
echo "IMOBench evaluation started at: $(date)"

source ~/.bashrc
source .venv/bin/activate
# Change to IMOBench directory, assuming this script is run from the repo root

echo "Running IMOBench script with arguments: $@"

# Run the benchmark
python scripts/run.py "$@"

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Evaluation completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes (${TOTAL_SECONDS} seconds)"

echo "Done!"
