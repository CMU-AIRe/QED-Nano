# Training

This directory contains the training code for QED-Nano, built on top of [PipelineRL](https://github.com/ServiceNow/PipelineRL) — a scalable asynchronous RL framework with in-flight weight updates.

## CMU-AIRe Setup

Follow the conda environment and other setup instructions below.

- For multinode training, use `run_multi_slurm.sh`.
- For singlenode training, use `run_single_slurm.sh`.
- Configs live under `conf/`. The main config is `conf/base.yaml`, which you can override per-experiment (e.g. `conf/pope.yaml`). Training-specific configs are in `conf/finetune/base.yaml` and are imported automatically.

### Installation

Create the conda environment and install dependencies:

```bash
conda create -n pipeline-rl -y python=3.11
conda run --no-capture-output -n pipeline-rl pip install torch==2.6.0
conda run --no-capture-output -n pipeline-rl pip install -e . --no-build-isolation
```

By default, PipelineRL uses the filesystem to stream generated data between processes. This works on a single node, but files can grow large. To use Redis instead:

```bash
conda install redis-server==7.4.0 -c conda-forge
```

### Running experiments

Activate the environment first:

```bash
conda activate pipeline-rl
```

Single node with 8 H100 GPUs:

```bash
python -m pipelinerl.launch output_dir=results/base1
```

With only 4 H100 GPUs:

```bash
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1
```

Using Redis for data streaming:

```bash
python -m pipelinerl.launch streams=redis output_dir=results/base1
```

## Recipes

### Proof-verification pipeline

```bash
# Interactive run — use a timestamped job name to avoid collisions
timestamp=$(date +'%Y%m%d-%H%M%S')
python -m pipelinerl.launch --config-name=proof_qwen3-4b-instruct-8k \
  output_dir="tmp/results/proof_qwen3-4b-instruct-8k-${timestamp}"
```

> [!WARNING]
> Timestamp each run's `output_dir` to avoid WandB collisions in PipelineRL, which cause the whole job to crash.

For faster iteration during development, use the demo pipeline:

```bash
# 4B instruct model
timestamp=$(date +'%Y%m%d-%H%M%S')
python -m pipelinerl.launch --config-name=proof_demo-instruct \
  output_dir="tmp/results/proof_demo-instruct-${timestamp}"

# 4B thinking model
timestamp=$(date +'%Y%m%d-%H%M%S')
python -m pipelinerl.launch --config-name=proof_demo-thinking \
  output_dir="tmp/results/proof_demo-thinking-${timestamp}"
```

### LLM grader configuration

The proof-verification pipeline uses an external LLM grader. Configure it via `llm_grader.name` in your config:

```yaml
llm_grader:
  name: openai/gpt-oss-20b          # local grader server
  # name: gpt-oss-120b-twj          # deployed endpoint
```

The grader server starts automatically when you launch training. Key sampling parameters:

```yaml
llm_grader:
  name: openai/gpt-oss-20b
  vllm_kwargs:
    num_nodes: 1
    data-parallel-size: 8
    tensor-parallel-size: 1
  sampling_kwargs:
    temperature: 1.0
    max_output_tokens: 32768
    reasoning:
      effort: medium
  reasoning_delimiters: ["</think>"]
  prompt_name: v0
```

`prompt_name` selects an evaluator prompt from `conf/evaluator_prompts/`. `reasoning_delimiters` determines where to split the model's response to extract the final answer.

> [!NOTE]
> For the Responses API, `max_output_tokens` is the total token budget (prompt + output). Set it high enough to accommodate the full response.

For Slurm deployments, tune `llm_grader.vllm_kwargs`:

```yaml
llm_grader:
  name: openai/gpt-oss-20b
  vllm_kwargs:
    num_nodes: 2
    data-parallel-size: 16
    tensor-parallel-size: 1
    max-num-batched-tokens: 8192
    max-num-seqs: 16
    max-model-len: 32768
    gpu-memory-utilization: 0.85
```

> [!NOTE]
> `data-parallel-size × tensor-parallel-size` must equal the total number of GPUs allocated to the grader (e.g., 2 nodes × 8 GPUs = 16 GPUs total).

To debug the grader, allocate nodes and run the server manually:

```bash
salloc --nodes=1 --gres=gpu:8 --qos=high --time=02:00:00 --job-name=prl-grader --partition=hopper-prod
srun --nodes=1 --ntasks=1 --overlap bash run_grader.slurm --model Qwen/Qwen3-0.6B --data-parallel-size 8
```

## Architecture

PipelineRL tackles the classic trade-off between **inference throughput** (large batches on many GPUs) and **on-policy data freshness** by performing _in-flight weight updates_: after each optimizer step, updated weights are broadcast to inference servers without halting sampling. This keeps batch sizes large and data near on-policy, yielding fast and stable RL for large language models.

The framework uses a simplified GRPO algorithm — no value network, no trust-region clamping, no KL or entropy bonuses by default (KL support is available).

PipelineRL is organized as a Hydra-driven pipeline with six core components spanning three stages: **actor**, **verifier**, and **trainer**.

### 1. Orchestrator (`pipelinerl/launch.py`)

- Parses and validates the Hydra config; initializes directories, logging, and the streams backend.
- Builds a **WorldMap** (`pipelinerl/world.py`) for rank-aware GPU placement using `WORLD_SIZE`, `RANK`, and `MASTER_ADDR` environment variables.
- Allocates each node's GPUs into actor, preprocessor, and trainer pools based on `cfg.world.*_fraction`.
- Launches subprocesses for all roles: `actor_llm`, `preprocessor_llm`, `actor`, `preprocessor`, `verifier`, and `finetune`.

### 2. Inference servers

- **Reference LLMs**: serve reference log-probs via `vllm.entrypoints.openai.api_server`.
- **Actor LLMs** (`pipelinerl/entrypoints/llm.py` → `pipelinerl/run_llm.py`): subclass vLLM's `Worker` to add:
  - `init_actor_update_group(...)` for NCCL process-group setup.
  - `receive_weight_update(request)` to pause inference, broadcast new weights via NCCL, and reload model parameters.
  - HTTP endpoints: `POST /v1/chat/completion` for sampling and `POST /receive_weight_update` for weight updates.

### 3. Actor processes (`pipelinerl/entrypoints/actor.py`)

- Loads train/test datasets, waits for inference servers, and initializes `TrainerState`.
- `ActorLoop` creates a `problem_queue` and `result_queue`, then spawns worker processes running `rollout_maker_entrypoint`.
- Each worker runs a uvloop-based asyncio event loop, listens for weight-update broadcasts, and calls `schedule_rollouts(...)` to issue concurrent HTTP requests to Actor LLM servers.
- `ActorLoop.run` refills the problem queue, reads rollout batches, writes samples to the `actor` stream, and publishes sliding-window metrics to WandB.
- Backpressure is controlled via `cfg.finetune.max_lag` and `cfg.finetune.weight_update_interval`.

### 4. Preprocessor (`pipelinerl/entrypoints/preprocess.py`)

- A background thread reads raw actor traces from the input stream in chunks.
- A `ProcessPoolExecutor` tokenizes and preprocesses sequences and optionally attaches reference log-probs.
- Writes processed micro-batches to the output stream (`cfg.preprocess.output`).

### 5. Trainer (`pipelinerl/entrypoints/finetune.py`)

- Background threads read and collate preprocessed micro-batches into PyTorch tensors.
- Main training loop: pull batch → `rl_step(...)` (policy-gradient + optional KL) → `optimizer.step()` → `lr_scheduler.step()`.
- On rank 0, `WeightUpdateManager.send_weight_update(version)` gathers parameters, sends `WeightUpdateRequest` to Actor LLMs over HTTP, broadcasts tensors via NCCL, and writes a `WeightUpdateSuccess` message to the update stream.

### 6. Verifier (`pipelinerl/entrypoints/verifier.py`)

Serves a FastAPI app with:
- `POST /`: checks model outputs (math or countdown puzzles) via `math_verify` or `countdown_utils`.
- `GET /health`: readiness probe.

### Streams backend (`pipelinerl/streams.py`)

Implements `SingleStreamSpec` and `StreamRangeSpec` for file-system or Redis-backed queues. `write_to_streams(...)` and `read_stream(...)` provide a JSON-line protocol for inter-process messaging.

Key streams:

| Stream | Direction | Purpose |
|---|---|---|
| `actor` | Actor → Preprocessor | Raw rollout samples |
| `training_data` | Preprocessor → Trainer | Processed training micro-batches |
| `stats` | Actor → Monitoring | Sliding-window metrics for WandB |
| `actor_test` / `stats_test` | Actor → Monitoring | Evaluation samples and metrics |
