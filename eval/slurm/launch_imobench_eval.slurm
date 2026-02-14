#!/bin/bash
#SBATCH --job-name=imobench-eval
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# IMOBench Grading/Evaluation Script
# This script grades model outputs using a grader model
#
# Usage examples:
# 1. Grade IMOBench results (answer benchmark):
#    sbatch slurm/launch_imobench_eval.slurm --model-config gpt-5 --data-path outputs/gpt-5.jsonl --final-answer
#
# 2. Grade IMOProofBench results (grading benchmark):
#    sbatch slurm/launch_imobench_eval.slurm --model-config gpt-5 --data-path outputs/gpt-5.jsonl
#
# 3. Grade ProofBench results:
#    sbatch slurm/launch_imobench_eval.slurm --model-config gpt-5 --data-path outputs/gpt-5.jsonl --proofbench
#
# 4. Grade with custom output path:
#    sbatch slurm/launch_imobench_eval.slurm --model-config gpt-5 --data-path outputs/gpt-5.jsonl --output-path outputs/gpt-5-graded.jsonl --final-answer
#
# 5. Grade with overwrite (ignore cached results):
#    sbatch slurm/launch_imobench_eval.slurm --model-config gpt-5 --data-path outputs/gpt-5.jsonl --final-answer --overwrite
#
# Available grader model configs (see IMOBench/configs/models/):
# - openai/gpt-51.yaml
# - openai/gpt-5-mini.yaml
# - google/gemini-2.5-pro.yaml
# - google/gemini-3-pro.yaml
# - other/grok-41-fast.yaml

set -x -e

# Record start time
SCRIPT_START_TIME=$(date +%s)
echo "IMOBench grading started at: $(date)"

source ~/.bashrc
source .venv/bin/activate
# Activate conda environment if needed (adjust path as necessary)
# source /admin/home/edward/miniconda3/bin/activate lm-provers

echo "Running IMOBench evaluation script with arguments: $@"

# Run the grading
python scripts/eval.py "$@"

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Grading completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes (${TOTAL_SECONDS} seconds)"

echo "Done!"
