#!/bin/bash
#SBATCH --job-name=imobench-vllm-serve
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# IMOBench with vLLM Server
# This script starts a vLLM OpenAI-compatible server and runs IMOBench against it
# ZERO code changes needed - uses existing IMOBench API client!
# The model path is automatically extracted from the config file.
#
# Usage:
# sbatch slurm/launch_imobench_vllm_serve.slurm \
#   --serve-config qwen-32b-vllm \
#   --output-path outputs/qwen32b.jsonl \
#   --final-answer
#
# Examples:
# - IMOBench (answer grading):
#   sbatch slurm/launch_imobench_vllm_serve.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --final-answer
#
# - IMOProofBench (proof grading):
#   sbatch slurm/launch_imobench_vllm_serve.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl
#
# - With agent scaffold:
#   sbatch slurm/launch_imobench_vllm_serve.slurm --serve-config qwen-32b-vllm --run-config agents/qwen-32b-vllm/deepseek_math --output-path outputs/qwen32b-agent.jsonl
#
# - ProofBench split 24_25:
#   sbatch slurm/launch_imobench_vllm_serve.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --data-path lm-provers/ProofBench --split 24_25
#
# - With overwrite:
#   sbatch slurm/launch_imobench_vllm_serve.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --final-answer --overwrite
#
# Arguments:
#   --serve-config SERVE_CONFIG - Model config name under IMOBench/configs/models/ (e.g., qwen-32b-vllm)
#                                 Used to extract `model:` and start vLLM.
#   --run-config RUN_CONFIG      - Optional run config (can be agents/<...>/<agent>); defaults to SERVE_CONFIG.
#   --output-path OUTPUT_PATH    - Required output path (.jsonl/.json).
#   $@: EXTRA_ARGS               - Additional args for run.py (e.g., --final-answer, --overwrite, --data-path, --split)

set -x -e

VLLM_EXTRA_ARGS=""
IMOBENCH_EXTRA_ARGS=""
MODEL_REVISION=""

SERVE_CONFIG=""
RUN_CONFIG=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --serve-config)
            SERVE_CONFIG="$2"
            shift 2
            ;;
        --run-config)
            RUN_CONFIG="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --revision)
            MODEL_REVISION="$2"
            VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --revision $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: sbatch ... launch_imobench_vllm_serve.slurm --serve-config SERVE_CONFIG [--run-config RUN_CONFIG] --output-path OUTPUT_PATH [EXTRA_ARGS...]"
            exit 0
            ;;
        *)
            # All other args go to IMOBench
            IMOBENCH_EXTRA_ARGS="$IMOBENCH_EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$SERVE_CONFIG" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: sbatch ... launch_imobench_vllm_serve.slurm --serve-config SERVE_CONFIG [--run-config RUN_CONFIG] --output-path OUTPUT_PATH [EXTRA_ARGS...]"
    exit 1
fi

if [ -z "$RUN_CONFIG" ]; then
    RUN_CONFIG="$SERVE_CONFIG"
fi

SCRIPT_START_TIME=$(date +%s)

module load cuda/12.9
source ~/.bashrc
source .venv/bin/activate

# Extract model path from config file
CONFIG_FILE="configs/models/${SERVE_CONFIG}.yaml"
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
echo "IMOBench vLLM Server Pipeline started at: $(date)"
echo "Serve config: $SERVE_CONFIG"
echo "Run config: $RUN_CONFIG"
echo "Model: $MODEL_PATH"
echo "Model revision: ${MODEL_REVISION:-main}"
echo "Output: $OUTPUT_PATH"
echo "Max tokens (generation): $MAX_TOKENS"
echo "Prompt margin: $PROMPT_MARGIN"
echo "Max model length (with prompt margin): $MAX_MODEL_LEN"
echo "vLLM args: $VLLM_EXTRA_ARGS"
echo "IMOBench args: $IMOBENCH_EXTRA_ARGS"
echo "================================================================"


# Set dummy API key for vLLM (doesn't actually require one, but IMOBench expects it)
export VLLM_API_KEY="dummy-key"

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Start vLLM server in background
echo "================================================================"
echo "STEP 1: Starting vLLM server on port 8000..."
echo "================================================================"

VLLM_LOG="/fsx/h4/logs/imobench-vllm-serve-${SLURM_JOB_ID}-vllm.log"
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
    --trust-remote-code \
    $VLLM_EXTRA_ARGS \
    &> "$VLLM_LOG" &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready (max 2 minutes)
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
echo "STEP 2: Running IMOBench evaluation..."
echo "================================================================"

RUN_START_TIME=$(date +%s)
uv run python scripts/run.py \
    --model-config "$RUN_CONFIG" \
    --output-path "$OUTPUT_PATH" \
    $IMOBENCH_EXTRA_ARGS

RUN_END_TIME=$(date +%s)
RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))

echo "================================================================"
echo "STEP 2 completed in $((RUN_DURATION / 60)) minutes"
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
echo "Pipeline completed at: $(date)"
echo "Evaluation time: $((RUN_DURATION / 60)) minutes"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes"
echo "================================================================"
echo "Results saved to: $OUTPUT_PATH"
echo "Done!"
