#!/bin/bash
#SBATCH --job-name=imobench-eval-vllm-serve
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# IMOBench Evaluation with vLLM Judge
# This script starts a vLLM OpenAI-compatible server for the judge model and runs IMOBench evaluation
# ZERO code changes needed - uses existing IMOBench API client!
# The judge model path is automatically extracted from the config file.
#
# Usage:
# sbatch slurm/launch_imobench_eval_vllm_serve.slurm \
#   --model-config qwen-32b-vllm \
#   --data-path outputs/model-output.jsonl \
#   --final-answer
#
# Examples:
# - Grade IMOBench results (answer benchmark):
#   sbatch slurm/launch_imobench_eval_vllm_serve.slurm --model-config qwen-32b-vllm --data-path outputs/gpt-5.jsonl --final-answer
#
# - Grade IMOProofBench results (proof grading):
#   sbatch slurm/launch_imobench_eval_vllm_serve.slurm --model-config qwen-32b-vllm --data-path outputs/gpt-5.jsonl
#
# - Grade ProofBench results (use ProofBench grading prompt):
#   sbatch slurm/launch_imobench_eval_vllm_serve.slurm --model-config qwen-32b-vllm --data-path outputs/gpt-5.jsonl --proofbench
#
# - With custom output path:
#   sbatch slurm/launch_imobench_eval_vllm_serve.slurm --model-config qwen-32b-vllm --data-path outputs/gpt-5.jsonl --output-path outputs/gpt-5-graded.jsonl --final-answer
#
# - With overwrite:
#   sbatch slurm/launch_imobench_eval_vllm_serve.slurm --model-config qwen-32b-vllm --data-path outputs/gpt-5.jsonl --final-answer --overwrite
#
# Arguments passed to eval.py (all are passed through):
#   --model-config CONFIG - Judge model config name (e.g., qwen-32b-vllm)
#   --data-path PATH     - Path to model outputs to grade
#   --output-path PATH   - (Optional) Where to save graded results
#   --final-answer       - (Optional) Grade IMOBench answer format
#   --proofbench         - (Optional) Grade ProofBench results
#   --overwrite          - (Optional) Ignore cached results

set -x -e

SCRIPT_START_TIME=$(date +%s)

# Parse arguments to extract model config for vLLM setup
MODEL_CONFIG=""
EVAL_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-config)
            MODEL_CONFIG="$2"
            EVAL_ARGS="$EVAL_ARGS $1 $2"
            shift 2
            ;;
        *)
            EVAL_ARGS="$EVAL_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$MODEL_CONFIG" ]; then
    echo "ERROR: --model-config is required"
    exit 1
fi

module load cuda/12.9
source ~/.bashrc
source .venv/bin/activate

# Extract model path from config file
CONFIG_FILE="configs/models/${MODEL_CONFIG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

MODEL_PATH=$(grep '^model:' "$CONFIG_FILE" | sed 's/model: *"\?\([^"]*\)"\?/\1/' | tr -d '"' | xargs)

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Could not extract 'model' field from $CONFIG_FILE"
    exit 1
fi

# Extract max_tokens from config and add margin for prompt tokens
MAX_TOKENS=$(grep '^max_tokens:' "$CONFIG_FILE" | sed 's/max_tokens: *//' | xargs)
if [ -z "$MAX_TOKENS" ] || [ "$MAX_TOKENS" -eq 0 ] 2>/dev/null; then
    echo "WARNING: Could not extract 'max_tokens' from config, using default 32768"
    MAX_TOKENS=32768
fi

# Extract prompt margin (how many tokens to reserve for prompts)
PROMPT_MARGIN=$(grep '^prompt_margin:' "$CONFIG_FILE" | sed 's/prompt_margin: *//' | xargs)
if [ -z "$PROMPT_MARGIN" ] || [ "$PROMPT_MARGIN" -eq 0 ] 2>/dev/null; then
    PROMPT_MARGIN=512
fi

# Ensure MAX_TOKENS/PROMPT_MARGIN are treated as integers
# Strip any non-numeric characters just in case
MAX_TOKENS=$(echo "$MAX_TOKENS" | grep -o '[0-9]*')
PROMPT_MARGIN=$(echo "$PROMPT_MARGIN" | grep -o '[0-9]*')
MAX_MODEL_LEN=$((MAX_TOKENS + PROMPT_MARGIN))

echo "================================================================"
echo "IMOBench vLLM Judge Evaluation started at: $(date)"
echo "Judge config: $MODEL_CONFIG"
echo "Judge model: $MODEL_PATH"
echo "Max tokens (generation): $MAX_TOKENS"
echo "Prompt margin: $PROMPT_MARGIN"
echo "Max model length (with prompt margin): $MAX_MODEL_LEN"
echo "Evaluation args: $EVAL_ARGS"
echo "================================================================"


# Set dummy API key for vLLM (doesn't actually require one, but IMOBench expects it)
export VLLM_API_KEY="dummy-key"

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Start vLLM server in background
echo "================================================================"
echo "STEP 1: Starting vLLM server for judge model on port 8000..."
echo "================================================================"

VLLM_LOG="/fsx/h4/logs/imobench-eval-vllm-serve-${SLURM_JOB_ID}-vllm.log"
echo "vLLM server logs will be written to: $VLLM_LOG"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.8 \
    --max-model-len $MAX_MODEL_LEN \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-num-seqs 16 \
    &> "$VLLM_LOG" &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready (max 10 minutes)
echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM server failed to start after 10 minutes"
        echo "Last 50 lines of $VLLM_LOG:"
        tail -50 "$VLLM_LOG"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "Attempt $i/60: Server not ready yet, waiting..."
    sleep 10
done

# Verify server is working
echo "Testing vLLM server..."
curl -s http://localhost:8000/v1/models | python -m json.tool || true
echo ""

# Run IMOBench evaluation
echo "================================================================"
echo "STEP 2: Running IMOBench grading with vLLM judge..."
echo "================================================================"

EVAL_START_TIME=$(date +%s)
uv run python scripts/eval.py $EVAL_ARGS

EVAL_END_TIME=$(date +%s)
EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))

echo "================================================================"
echo "STEP 2 completed in $((EVAL_DURATION / 60)) minutes"
echo "================================================================"

# Cleanup: Kill vLLM server
echo "Shutting down vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM server stopped"

# Calculate total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "================================================================"
echo "Grading completed at: $(date)"
echo "Evaluation time: $((EVAL_DURATION / 60)) minutes"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes"
echo "================================================================"
echo "Done!"
