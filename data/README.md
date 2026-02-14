# Data Generation SLURM Scripts

This directory contains SLURM scripts for large-scale data generation using DeepSeek-Math-V2 (685B) for creating SFT training data.

## Overview

The data generation pipeline uses SGLang to serve DeepSeek-Math-V2 across multiple nodes and generate proof solutions at scale (128k context limit). This is used to create the distillation dataset for supervised fine-tuning.

## Scripts

### `serve_deepseek_math_v2_pp4.slurm`
Launches DeepSeek-Math-V2 server using SGLang with pipeline parallelism across 4 nodes.

**Configuration:**
- **Resources**: 4 nodes × 8 H100 GPUs (32 GPUs total)
- **Parallelism**: TP=8, EP=8, PP=4
- **Context length**: 163,840 tokens
- **Max running requests**: 16
- **Runtime**: Up to 7 days

**Key features:**
- Pipeline parallelism layer partition: 15,15,15,16
- Health check monitoring
- Optional router registration
- Sanity check on startup

**Usage:**
```bash
sbatch serve_deepseek_math_v2_pp4.slurm
```

### `sglang_router.slurm`
Launches an SGLang router to load-balance requests across multiple DeepSeek-Math-V2 server instances.

**Configuration:**
- **Resources**: 1 CPU-only node
- **Policy**: Cache-aware routing
- **Retry logic**: 5 max retries with exponential backoff
- **Port**: 30000

**Features:**
- Aggregates multiple worker URLs into a single endpoint
- Cache-aware request routing for efficiency
- Robust retry mechanism with jitter

**Usage:**
1. Update `--worker-urls` with your server IPs (from `serve_deepseek_math_v2_pp4.slurm` jobs)
2. Launch router:
```bash
sbatch sglang_router.slurm
```

### `sglang_generate.slurm`
Generates proof solutions from DeepSeek-Math-V2 for creating the SFT dataset using `scripts/generate_sglang.py`.

**Configuration:**
- **Resources**: 1 CPU-only node (no GPUs needed)
- **Max tokens**: 124,928 per generation
- **Runtime**: Up to 7 days
- **Concurrency**: 256 concurrent requests (configurable)

**Usage:**
```bash
sbatch sglang_generate.slurm
```

The SLURM script calls `scripts/generate_sglang.py` with the following key parameters:

**Required arguments:**
- `--dataset`: HuggingFace dataset ID with problems (e.g., `hf-imo-colab/aops-v2-data`)
- `--split`: Dataset split to use (e.g., `train`)
- `--text-column`: Column name containing the problem text (e.g., `problem`)
- `--model`: Model name to send to the API (e.g., `deepseek-ai/DeepSeek-Math-V2`)
- `--base-url`: OpenAI-compatible endpoint URL (router or server, e.g., `http://ip-26-0-171-230:30000/v1`)
- `--max-tokens`: Max tokens to generate per completion (e.g., `124928`)
- `--outfile-json-filename`: Local JSONL output file path

**Optional arguments:**
- `--output-dataset-name`: HuggingFace dataset ID to stream results to (e.g., `hf-imo-colab/output-dataset`)
- `--push-every`: Push to Hub after this many samples (default: 50)
- `--requests-per-prompt`: Number of independent generations per problem (default: 1)
- `--enable-thinking`: Enable thinking tokens for DeepSeek-Math-V2
- `--system-prompt`: Optional system prompt to prefix conversations
- `--temperature`: Sampling temperature (default: 1.0)
- `--concurrency`: Number of concurrent in-flight requests (default: 256)
- `--offset`: Start index within the dataset (default: 0)
- `--limit`: Maximum number of samples to process (default: all)

**Example direct usage:**
```bash
python scripts/generate_sglang.py \
  --dataset hf-imo-colab/olympiads-proof-schema-cleaned \
  --split train \
  --text-column problem \
  --model deepseek-ai/DeepSeek-Math-V2 \
  --base-url http://ip-26-0-160-100:30000/v1 \
  --max-tokens 124928 \
  --outfile-json-filename outputs.jsonl \
  --output-dataset-name hf-imo-colab/olympiads-completions \
  --requests-per-prompt 1 \
  --enable-thinking
```

## Typical Workflow

### Single Server Setup
1. Launch one server instance:
   ```bash
   sbatch serve_deepseek_math_v2_pp4.slurm
   ```
2. Get the server IP from logs and update `sglang_generate.slurm`
3. Run generation:
   ```bash
   sbatch sglang_generate.slurm
   ```

### Multi-Server Setup (Recommended)
For higher throughput (~3000 tokens/s aggregate):

1. Launch multiple server instances (e.g., 8 instances):
   ```bash
   for i in {1..8}; do sbatch serve_deepseek_math_v2_pp4.slurm; done
   ```

2. Wait for servers to start (~30 min for model loading)

3. Update `sglang_router.slurm` with all server IPs and launch router:
   ```bash
   sbatch sglang_router.slurm
   ```

4. Point generation script to router and run:
   ```bash
   # Update --base-url to router address in sglang_generate.slurm
   sbatch sglang_generate.slurm
   ```

## Performance Notes

- **Model loading time**: ~30 minutes per 4-node instance
- **Throughput**: ~3000 tokens/s with 8 parallel instances + router
- **Context length**: 128k tokens used during generation (163k max supported)
- **Total resources**: 8 instances × 4 nodes × 8 GPUs = 256 H100s for full pipeline

## Output

Generated data is saved as JSONL files and optionally pushed to HuggingFace Hub. The output contains:
- Original problem statements
- Generated proof solutions (up to 128k tokens)
- Metadata (tokens used, generation time, etc.)

This data is then graded with Gemini-3-Pro and used for supervised fine-tuning initialization before RL training.
