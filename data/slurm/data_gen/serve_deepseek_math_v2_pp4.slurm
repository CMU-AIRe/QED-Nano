#!/bin/bash
#SBATCH --job-name=deepseek-math-v2-serve-spec-pp4
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1

set -exo pipefail

MODEL_PATH="deepseek-ai/DeepSeek-Math-V2"
ROUTER_ADDRESS=""
SERVER_PORT=39877
DIST_PORT=45000

# TODO: Adjust these variables to your cluster configuration
export OUTLINES_CACHE_DIR=/scratch/serve_r1/ocache/
export TRITON_HOME=/scratch/serve_r1/triton/
export GLOO_SOCKET_IFNAME="enp71s0"
export NCCL_SOCKET_IFNAME="enp71s0"

# TODO: Environment setup, adjust to your cluster configuration
module load cuda/12.9
source ~/.bashrc
source IMOBench/.venv/bin/activate

FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
FIRST_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$FIRST_NODE" hostname --ip-address)

# Launch servers synchronously across all nodes
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash -c "export SGLANG_PP_LAYER_PARTITION=15,15,15,16 && python -m sglang.launch_server \
        --model-path '$MODEL_PATH' \
        --tp 8 \
        --ep 8 \
        --pp-size 4 \
        --dist-init-addr '$FIRST_NODE_IP:$DIST_PORT' \
        --nnodes 4 \
        --cuda-graph-max-bs 16 \
        --torch-compile-max-bs 16 \
        --node-rank \$SLURM_PROCID \
        --port '$SERVER_PORT' \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-running-requests 16 \
        --context-length 163840" &

# Wait for server with timeout
TIMEOUT=3600  # 1h, but model loading should take ~30min
START_TIME=$(date +%s)
echo "Waiting for SGLang server (http://$FIRST_NODE_IP:$SERVER_PORT)..."

while true; do
    if curl -s -o /dev/null -w "%{http_code}" "http://$FIRST_NODE_IP:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "Server is ready at http://$FIRST_NODE_IP:$SERVER_PORT"
        break
    fi

    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "Error: Server failed to start within $TIMEOUT seconds"
        exit 1
    fi

    echo "Still waiting... ($(($CURRENT_TIME - $START_TIME)) seconds elapsed)"
    sleep 60
done

# Register with router only if address was provided
if [ -n "$ROUTER_ADDRESS" ]; then
    echo "Registering with router at $ROUTER_ADDRESS..."
    curl -X POST "http://$ROUTER_ADDRESS/add_worker?url=http://$FIRST_NODE_IP:$SERVER_PORT" || true
    sleep 10
fi

echo "Checking available models..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"default\",
        \"prompt\": \"<|begin▁of▁sentence|><|User|>hi, how are you?<|Assistant|>\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# Keep the job running with health checks
while true; do
    if ! curl -s -o /dev/null "http://$FIRST_NODE_IP:$SERVER_PORT/health"; then
        echo "Error: Server health check failed"
        exit 1
    fi
    sleep 300
done