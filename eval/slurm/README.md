# IMOBench SLURM Scripts

This directory contains SLURM scripts for running IMOBench (International Mathematical Olympiad Benchmark) evaluations on HPC clusters.

## Overview

IMOBench evaluates AI models on mathematical problem-solving using two modes:
- **Answer Benchmark**: Models solve problems and provide final answers
- **Proof Grading Benchmark**: Models grade mathematical proofs/solutions, including ProofBench support

## Scripts

### Queue Script (Recommended)

**`queue_launch_imobench.sh`** - Main orchestration script that automatically:
- Creates model configs
- Launches vLLM inference server
- Runs model evaluation
- Grades results with a judge model
- Intelligently selects between API-based and vLLM-based judges

### Inference Scripts

**`launch_imobench.slurm`** - Run inference using external API models (OpenAI, Gemini, etc.)

**`launch_imobench_vllm_serve.slurm`** - Run inference using local vLLM server (single-node)

**`launch_imobench_vllm_serve_multinode.slurm`** - Run inference using local vLLM server (multi-node with Ray)

### Evaluation Scripts

**`launch_imobench_eval.slurm`** - Grade model outputs using external API judge models

**`launch_imobench_eval_vllm_serve.slurm`** - Grade model outputs using local vLLM judge model

**`launch_imobench_run_and_eval.slurm`** - Combined inference and evaluation (legacy)

### Utilities

**`infer_max_tokens.py`** - Python utility to automatically infer `max_tokens` from HuggingFace model configs
- Queries model's `config.json` for context length
- Chooses a prompt headroom (`prompt_margin`) and sets `max_tokens = context_length - prompt_margin`
- Used automatically by queue script
- The generated vLLM configs include `prompt_margin`, and the SLURM launch scripts use `max_tokens + prompt_margin` for vLLM `--max-model-len`

## Quick Start

### Basic Usage

```bash
# Note: API judge configs live under IMOBench/configs/models/<provider>/ (e.g., google/gemini-3-pro).
# queue_launch_imobench.sh expects the provider subdir path (google/gemini-3-pro)
# Run inference and evaluation with external API judge (Gemini)
bash slurm/queue_launch_imobench.sh \
  --model Qwen/Qwen3-32B \
  --judge google/gemini-3-pro \
  --final-answer

# Run with vLLM-based judge (auto-creates config if needed)
bash slurm/queue_launch_imobench.sh \
  --model Qwen/Qwen3-32B \
  --judge Qwen/Qwen2.5-72B-Instruct \
  --final-answer

# Run ProofBench (auto-selects ProofBench data and grading prompts)
bash slurm/queue_launch_imobench.sh \
  --model Qwen/Qwen3-32B \
  --judge google/gemini-3-pro \
  --proofbench \
  --split 24_25

# Run an agent scaffold against a vLLM-served model (derived config auto-created)
bash slurm/queue_launch_imobench.sh \
  --model lm-provers/Qwen3-4B-Thinking-2507-Proof \
  --model-revision v07.00-step-000200 \
  --agent deepseek_math \
  --judge google/gemini-3-pro
```

### Advanced Usage

```bash
# Multi-node deployment (large models)
bash slurm/queue_launch_imobench.sh \
  --model Qwen/Qwen3-235B \
  --num-nodes 4 \
  --num-gpus 8 \
  --final-answer

# Specific model revision
bash slurm/queue_launch_imobench.sh \
  --model Qwen/Qwen3-32B \
  --model-revision v1.0.0 \
  --final-answer

# Smaller model with fewer GPUs
bash slurm/queue_launch_imobench.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-gpus 1 \
  --judge google/gemini-3-pro \
  --final-answer
```

## Arguments Reference

### `queue_launch_imobench.sh`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | Required | - | HuggingFace model path (e.g., `Qwen/Qwen3-32B`) |
| `--model-revision` | Optional | `main` | Model revision/branch/tag to use |
| `--agent` | Optional | - | Agent scaffold config name under `IMOBench/configs/agents/` (e.g., `deepseek_math`) |
| `--judge` | Optional | `google/gemini-3-pro` | Judge model (config name or provider subdir path under `IMOBench/configs/models/`, or HF path) |
| `--final-answer` | Flag | - | Enable answer benchmark mode (vs proof grading) |
| `--proofbench` | Flag | - | Run ProofBench (cannot combine with `--final-answer`) |
| `--data-path` | Optional | Auto | Custom data source (defaults to IMOBench/ProofBench depending on mode) |
| `--split` | Optional | `train` | Dataset split to run (e.g., `24_25`, `other`) |
| `--num-nodes` | Optional | `1` | Number of nodes for inference |
| `--num-gpus` | Optional | `8` | GPUs per node for inference |

## Judge Model Types

The queue script automatically detects judge model type:

### External API Judges
Models starting with `gpt*`, `gemini*`, `claude*`, or `grok*` use external APIs:
- `google/gemini-3-pro` (configs/models/google/gemini-3-pro.yaml)
- `gpt-51` (configs/models/openai/gpt-51.yaml)
- `gpt-5-mini` (configs/models/openai/gpt-5-mini.yaml)
- `claude-*`
- `grok-41-fast` (configs/models/other/grok-41-fast.yaml)

**Requirements**: Appropriate API keys set in environment

### vLLM Judges
Any other model name or HuggingFace path uses local vLLM:
- `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- Existing vLLM configs (e.g., `vllm-qwen3-32b`)

**Requirements**: Model fits on 8 GPUs (judge evaluation always uses 8 GPUs)

## Workflow

1. **Config Creation**: Script creates vLLM config files in `configs/models/`
2. **Job 1 - Inference**:
   - Starts vLLM server with model
   - Runs inference on IMOBench problems
   - Outputs results to `outputs/<model-name>.jsonl`
3. **Job 2 - Evaluation** (runs after Job 1):
   - Starts vLLM server for judge (if vLLM judge) OR uses API
   - Grades model outputs
   - Outputs grades to `outputs/<model-name>-<judge-name>.jsonl`

## Output Files

### File Naming Convention

```
outputs/vllm-<org>-<model>[-<revision>].jsonl              # Inference results
outputs/vllm-<org>-<model>[-<revision>]-<judge>.jsonl      # Evaluation results
```

### Examples

```
outputs/vllm-qwen-qwen3-32b.jsonl                           # Inference (main branch)
outputs/vllm-qwen-qwen3-32b-v1.0.0.jsonl                    # Inference (v1.0.0)
outputs/vllm-qwen-qwen3-32b-google-gemini-3-pro.jsonl       # Graded by Gemini
outputs/vllm-qwen-qwen3-32b-vllm-qwen-qwen2.5-72b.jsonl    # Graded by Qwen vLLM
```
