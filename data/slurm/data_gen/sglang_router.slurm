#!/bin/bash
#SBATCH --job-name=sglang_router
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

# TODO: Update --worker-urls with your actual server IPs from squeue/serve jobs
# For gRPC mode, use grpc://ip-address:port, otherwise http://ip-address:port

python -m sglang_router.launch_router \
    --worker-urls http://ip-26-0-164-18:39877 http://ip-26-0-163-220:39877 http://ip-26-0-160-40:39877 http://ip-26-0-166-125:39877 http://ip-26-0-167-177:39877 http://ip-26-0-167-245:39877 http://ip-26-0-166-36:39877 http://ip-26-0-161-103:39877 \
    --model-path deepseek-ai/DeepSeek-Math-V2 \
    --policy cache_aware \
    --retry-max-retries 5 \
    --retry-initial-backoff-ms 3000 \
    --retry-max-backoff-ms 30000 \
    --retry-backoff-multiplier 1.5 \
    --retry-jitter-factor 0.2 \
    --host 0.0.0.0 --port 30000