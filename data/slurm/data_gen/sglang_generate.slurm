#!/bin/bash
#SBATCH --job-name=sglang_generate
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=7-00:00:00

set -x -e

source ~/.bashrc
source IMOBench/.venv/bin/activate

python scripts/generate_sglang.py \
  --dataset hf-imo-colab/aops-v2-data-260123-missing-prompts \
  --split train \
  --text-column problem \
  --model deepseek-ai/DeepSeek-Math-V2 \
  --base-url http://ip-26-0-171-230:30000/v1 \
  --max-tokens 124928 \
  --outfile-json-filename aops-v2-data-260123-missing-prompts-ds-math-v2-gen-raw-260123-2.jsonl \
  --output-dataset-name hf-imo-colab/aops-v2-data-260123-missing-prompts-ds-math-v2-gen-raw-260123-2 \
  --requests-per-prompt 1 \
  --enable-thinking