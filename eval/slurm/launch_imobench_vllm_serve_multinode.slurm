#!/bin/bash
#SBATCH --job-name=imobench-vllm-serve-multinode
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00

# IMOBench with Multi-node vLLM Server
# This script starts a distributed vLLM OpenAI-compatible server across multiple nodes
# and runs IMOBench against it. Uses Ray for distributed inference.
#
# Usage:
# sbatch slurm/launch_imobench_vllm_serve_multinode.slurm \
#   --serve-config qwen-32b-vllm \
#   --output-path outputs/qwen32b.jsonl \
#   --final-answer
#
# To change number of nodes, modify #SBATCH --nodes=2 above
#
# Examples:
# - IMOBench (answer grading):
#   sbatch slurm/launch_imobench_vllm_serve_multinode.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --final-answer
#
# - IMOProofBench (proof grading):
#   sbatch slurm/launch_imobench_vllm_serve_multinode.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl
#
# - With agent scaffold:
#   sbatch slurm/launch_imobench_vllm_serve_multinode.slurm --serve-config qwen-32b-vllm --run-config agents/qwen-32b-vllm/deepseek_math --output-path outputs/qwen32b-agent.jsonl
#
# - ProofBench split 24_25:
#   sbatch slurm/launch_imobench_vllm_serve_multinode.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --data-path lm-provers/ProofBench --split 24_25
#
# - With overwrite:
#   sbatch slurm/launch_imobench_vllm_serve_multinode.slurm --serve-config qwen-32b-vllm --output-path outputs/qwen32b.jsonl --final-answer --overwrite
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
            echo "Usage: sbatch ... launch_imobench_vllm_serve_multinode.slurm --serve-config SERVE_CONFIG [--run-config RUN_CONFIG] --output-path OUTPUT_PATH [EXTRA_ARGS...]"
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
    echo "Usage: sbatch ... launch_imobench_vllm_serve_multinode.slurm --serve-config SERVE_CONFIG [--run-config RUN_CONFIG] --output-path OUTPUT_PATH [EXTRA_ARGS...]"
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
MAX_TOKENS=$(echo "$MAX_TOKENS" | grep -o '[0-9]*')
PROMPT_MARGIN=$(echo "$PROMPT_MARGIN" | grep -o '[0-9]*')
MAX_MODEL_LEN=$((MAX_TOKENS + PROMPT_MARGIN))

# Get node information
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

# Get the list of nodes
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo $NODES | cut -d' ' -f1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)

echo "================================================================"
echo "IMOBench Multi-node vLLM Server Pipeline started at: $(date)"
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
echo "----------------------------------------------------------------"
echo "Cluster Configuration:"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Head node: $HEAD_NODE (IP: $HEAD_NODE_IP)"
echo "All nodes: $NODES"
echo "================================================================"

# Set dummy API key for vLLM
export VLLM_API_KEY="dummy-key"

# Ray configuration
RAY_PORT=6379
VLLM_PORT=8000

echo "================================================================"
echo "STEP 1: Starting Ray cluster..."
echo "================================================================"

# Resource detection
CPUS_PER_NODE=${SLURM_CPUS_PER_TASK:-$(nproc)}
echo "Resources per node: CPUs=$CPUS_PER_NODE, GPUs=$NUM_GPUS_PER_NODE"

# Clean up any existing Ray sessions
echo "Cleaning up existing Ray sessions..."
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES ray stop --force 2>/dev/null || true
sleep 3

# Start Ray head node with --block in background
echo "Starting Ray head node on $HEAD_NODE (IP: $HEAD_NODE_IP)..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head \
    --node-ip-address="$HEAD_NODE_IP" \
    --port=$RAY_PORT \
    --num-cpus=$CPUS_PER_NODE \
    --num-gpus=$NUM_GPUS_PER_NODE \
    --block &

echo "Waiting for Ray head to fully initialize..."
sleep 20

# Start Ray workers on other nodes with --block in background
WORKER_NUM=$((NUM_NODES - 1))
if [ $WORKER_NUM -gt 0 ]; then
    echo "Starting $WORKER_NUM Ray worker node(s)..."
    srun --nodes=$WORKER_NUM --ntasks=$WORKER_NUM --exclude="$HEAD_NODE" \
        ray start --address="$HEAD_NODE_IP:$RAY_PORT" \
        --num-cpus=$CPUS_PER_NODE \
        --num-gpus=$NUM_GPUS_PER_NODE \
        --block &

    echo "Waiting for workers to connect and stabilize..."
    sleep 30
fi

# Health check - wait for all GPUs to be visible
echo "Waiting for Ray cluster to register $TOTAL_GPUS GPUs..."
export RAY_ADDRESS="$HEAD_NODE_IP:$RAY_PORT"

for i in {1..60}; do
    # Ask Ray how many GPUs it sees
    CURRENT_GPUS=$(python3 -c "import ray; ray.init(); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo "0")

    echo "Ray cluster status: $CURRENT_GPUS / $TOTAL_GPUS GPUs ready"

    if [ "$CURRENT_GPUS" -ge "$TOTAL_GPUS" ]; then
        echo "✓ Ray cluster fully ready with $TOTAL_GPUS GPUs!"
        break
    fi

    if [ $i -eq 60 ]; then
        echo "ERROR: Timeout waiting for Ray cluster. Only $CURRENT_GPUS/$TOTAL_GPUS GPUs available."
        exit 1
    fi

    sleep 5
done

# Start vLLM server in background on head node
echo "================================================================"
echo "STEP 2: Starting vLLM server on port $VLLM_PORT..."
echo "================================================================"

VLLM_LOG="/fsx/h4/logs/imobench-vllm-serve-${SLURM_JOB_ID}-vllm.log"
echo "vLLM server logs will be written to: $VLLM_LOG"

# Run vLLM serve with Ray backend
echo "Starting vLLM server with Ray backend..."
export RAY_ADDRESS="$HEAD_NODE_IP:$RAY_PORT"

# Run vLLM directly (batch script is already on head node)
# Must explicitly set --distributed-executor-backend=ray for multi-node
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TOTAL_GPUS \
    --distributed-executor-backend ray \
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

# Wait for server to be ready (max 15 minutes)
echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
    if curl -s http://$HEAD_NODE_IP:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM server failed to start after 10 minutes"
        echo "Last 50 lines of $VLLM_LOG:"
        tail -50 "$VLLM_LOG"
        # Cleanup
        kill $VLLM_PID 2>/dev/null || true
        srun --nodes=$NUM_NODES --ntasks=$NUM_NODES ray stop || true
        exit 1
    fi
    echo "Attempt $i/60: Server not ready yet, waiting..."
    sleep 15
done

# Verify server is working
echo "Testing vLLM server..."
curl -s http://$HEAD_NODE_IP:$VLLM_PORT/v1/models | python -m json.tool || true
echo ""

# Run IMOBench evaluation from head node
echo "================================================================"
echo "STEP 3: Running IMOBench evaluation..."
echo "================================================================"

RUN_START_TIME=$(date +%s)

# Update config to point to head node
export OPENAI_BASE_URL="http://$HEAD_NODE_IP:$VLLM_PORT/v1"

# Run directly since we're already on an allocated node
uv run python scripts/run.py \
    --model-config "$RUN_CONFIG" \
    --output-path "$OUTPUT_PATH" \
    $IMOBENCH_EXTRA_ARGS

RUN_END_TIME=$(date +%s)
RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))

echo "================================================================"
echo "STEP 3 completed in $((RUN_DURATION / 60)) minutes"
echo "================================================================"

# Cleanup: Kill vLLM server and stop Ray
echo "Shutting down vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM server stopped"

echo "Stopping Ray cluster..."
# Kill the backgrounded srun processes (which will stop Ray)
pkill -P $$ srun 2>/dev/null || true
# Also explicitly stop Ray on all nodes
for NODE in $NODES; do
    ssh $NODE "ray stop --force" 2>/dev/null || true
done
echo "Ray cluster stopped"

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
