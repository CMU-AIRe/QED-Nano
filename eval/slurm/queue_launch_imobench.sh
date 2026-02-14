#!/bin/bash
# Queue IMOBench pipeline: inference + evaluation
# Intelligently routes between API models and vLLM models for both inference and evaluation
#
# This script intelligently selects inference and evaluation methods:
# - External API models (gpt*, gemini*, claude*, grok*): Uses launch_imobench.slurm (CPU)
# - vLLM models (HuggingFace paths): Uses launch_imobench_vllm_serve*.slurm (GPU)
# - External API judges: Uses launch_imobench_eval.slurm (CPU)
# - vLLM judges: Uses launch_imobench_eval_vllm_serve.slurm (GPU)
# - Configs auto-created for HuggingFace models
#
# Usage: bash slurm/IMOBench/queue_launch_imobench.sh --model MODEL_NAME [OPTIONS]
#
# Examples:
#   # Note: API configs live under IMOBench/configs/models/<provider>/ (e.g., google/gemini-3-pro).
#   # queue_launch_imobench.sh expects the provider subdir path (google/gemini-3-pro)
#   # API model (Gemini) for proof generation with vLLM judge
#   bash slurm/IMOBench/queue_launch_imobench.sh --model google/gemini-3-pro --judge Qwen/Qwen2.5-72B-Instruct
#
#   # API model (OpenAI) + final-answer benchmark
#   bash slurm/IMOBench/queue_launch_imobench.sh --model openai/gpt-5-mini --judge google/gemini-3-pro --final-answer
#
#   # vLLM model with API judge
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge google/gemini-3-pro --final-answer
#
#   # vLLM model with agent scaffold
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge google/gemini-3-pro --agent deepseek_math
#
#   # vLLM model with vLLM judge (auto-creates config if needed)
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge Qwen/Qwen2.5-72B-Instruct --final-answer
#
#   # ProofBench (uses ProofBench data + proofbench grading)
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge google/gemini-3-pro --proofbench --split 24_25
#
#   # Multi-node inference
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen3-235B-A22B-Thinking-2507 --num-nodes 2 --num-gpus 8
#
#   # Smaller model with fewer GPUs
#   bash slurm/IMOBench/queue_launch_imobench.sh --model meta-llama/Llama-3.1-8B-Instruct --num-gpus 1
#
#   # Specific model revision
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --model-revision v1.0.0 --final-answer
#
#   # With reasoning efforts
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --gen-reasoning-effort high --judge Qwen/Qwen2.5-72B-Instruct --judge-reasoning-effort medium
#
#   # With custom sampling parameters
#   bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --temperature 0.6 --top-p 0.95
#
# Arguments:
#   --model MODEL_NAME             - (Required) Model id or config path under IMOBench/configs/models/
#                                   Examples:
#                                     - HuggingFace id (served via vLLM): Qwen/Qwen3-32B, openai/gpt-oss-20b
#                                     - External API config path: google/gemini-3-pro, openai/gpt-5-mini, openai/oss-20b-high
#   --model-revision REV           - (Optional) Model revision/branch (default: main)
#   --agent AGENT_NAME             - (Optional) Agent scaffold config under IMOBench/configs/agents/ (e.g., deepseek_math)
#   --judge JUDGE_NAME             - (Optional) Judge model (config name or provider path under IMOBench/configs/models/, or HF path)
#   --gen-reasoning-effort EFFORT  - (Optional) Generation reasoning effort: high, medium, or low
#   --judge-reasoning-effort EFFORT- (Optional) Judge reasoning effort: high, medium, or low
#   --final-answer                 - (Optional) Run IMOBench final-answer benchmark (passes --final-answer to run.py and eval.py)
#   --proofbench                   - (Optional) Use ProofBench data + grading prompts (mutually exclusive with --final-answer)
#   --data-path PATH               - (Optional) Custom data path for inference (defaults set by benchmark)
#   --split SPLIT                  - (Optional) Dataset split to run (e.g., train, 24_25, other)
#   --num-nodes NUM                - (Optional) Number of nodes to use for inference (default: 1)
#   --num-gpus NUM                 - (Optional) Number of GPUs per node for inference (default: 8)
#   --overwrite                    - (Optional) Overwrite existing output files from job 1
#   --temperature TEMP             - (Optional) Sampling temperature (e.g., 0.6)
#   --top-p VALUE                  - (Optional) Top-p (nucleus) sampling threshold (e.g., 0.95)
#   --top-k VALUE                  - (Optional) Top-k sampling value
#
# Notes:
#   - If num-nodes > 1, the multi-node vLLM serve script will be used automatically
#   - Judge evaluation always uses 8 GPUs when using vLLM judges

set -e

print_usage() {
    cat <<'EOF'
Usage: bash slurm/IMOBench/queue_launch_imobench.sh --model MODEL_NAME [OPTIONS]

Examples:
  # IMOBench (answer benchmark; final-answer mode)
  bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge google/gemini-3-pro --final-answer

  # IMOProofBench (proof grading; default mode)
  bash slurm/IMOBench/queue_launch_imobench.sh --model Qwen/Qwen3-32B --judge google/gemini-3-pro

Options (selected):
  --final-answer              Enable IMOBench final-answer benchmark mode
  --proofbench                Enable ProofBench mode (mutually exclusive with --final-answer)
  --data-path PATH            Override dataset path
  --split SPLIT               Dataset split (e.g., train, 24_25, other)
  --overwrite                 Overwrite existing inference output from Job 1
EOF
}

# Function to create vLLM config file for a model
# Arguments:
#   $1: model_path - HuggingFace model path (e.g., Qwen/Qwen3-32B)
#   $2: config_name - Config file name (e.g., vllm-qwen-qwen3-32b)
#   $3: reasoning_effort - (Optional) Reasoning effort: high, medium, or low
#   $4: temperature - (Optional) Sampling temperature
#   $5: top_p - (Optional) Top-p (nucleus) sampling
#   $6: top_k - (Optional) Top-k sampling
#   $7: model_revision - (Optional) Model revision/branch (e.g., v1.0.0)
# Returns: Creates config file at IMOBench/configs/models/vllm/${config_name}.yaml
create_vllm_config() {
    local model_path="$1"
    local config_name="$2"
    local reasoning_effort="$3"
    local temperature="$4"
    local top_p="$5"
    local top_k="$6"
    local model_revision="$7"
    local config_dir="configs/models/vllm"
    local config_file="${config_dir}/${config_name}.yaml"

    mkdir -p "$config_dir"
    echo "Creating vLLM config: $config_file"
    echo "  Model: $model_path"

    # Infer max_tokens for model
    local inferred
    inferred=$(python slurm/infer_max_tokens.py "$model_path" 2>/dev/null || echo "28672 512")
    local max_tokens prompt_margin
    read -r max_tokens prompt_margin <<< "$inferred"
    if [ -z "$max_tokens" ]; then max_tokens="28672"; fi
    if [ -z "$prompt_margin" ]; then prompt_margin="512"; fi
    echo "  Max tokens: $max_tokens (prompt margin: $prompt_margin)"

    # Create base vLLM config
    cat > "$config_file" <<EOF
model: "${model_path}"
EOF

    if [ -n "$model_revision" ]; then
        echo "model_revision: \"${model_revision}\"" >> "$config_file"
        echo "  Model revision: $model_revision"
    fi

    cat >> "$config_file" <<EOF
api: custom
api_key_env: VLLM_API_KEY
base_url: http://localhost:8000/v1
max_tokens: ${max_tokens}
prompt_margin: ${prompt_margin}
delimiters:
- </think>  # covers most reasoning models
- final<|message|> # gpt-oss models
read_cost: 0.0
write_cost: 0.0
concurrent_requests: 16
human_readable_id: ${model_path}
date: "$(date +%Y-%m-%d)"
EOF

    # Add reasoning_effort if provided
    if [ -n "$reasoning_effort" ]; then
        echo "reasoning_effort: ${reasoning_effort}" >> "$config_file"
        echo "  Reasoning effort: $reasoning_effort"
    fi

    # Add sampling parameters if provided
    if [ -n "$temperature" ]; then
        echo "temperature: ${temperature}" >> "$config_file"
        echo "  Temperature: $temperature"
    fi
    if [ -n "$top_p" ]; then
        echo "top_p: ${top_p}" >> "$config_file"
        echo "  Top-p: $top_p"
    fi
    if [ -n "$top_k" ]; then
        echo "top_k: ${top_k}" >> "$config_file"
        echo "  Top-k: $top_k"
    fi

    echo "  Config created: $config_file"
}

# List external API model configs (relative to IMOBench/configs/models, without .yaml)
list_external_api_configs() {
    find \
        IMOBench/configs/models/openai \
        IMOBench/configs/models/google \
        IMOBench/configs/models/other \
        -type f -name "*.yaml" 2>/dev/null \
        | sed 's|^IMOBench/configs/models/||' \
        | sed 's/\.yaml$//' \
        | sort
}

# Resolve an existing config path (relative to IMOBench/configs/models, without .yaml).
# Tries base, base_${effort}, base-${effort} (if effort is provided).
resolve_existing_model_config_path() {
    local base="$1"
    local effort="$2"
    local -a candidates=()
    candidates+=("$base")
    if [ -n "$effort" ]; then
        candidates+=("${base}_${effort}")
        candidates+=("${base}-${effort}")
    fi

    local cand
    for cand in "${candidates[@]}"; do
        if [ -f "IMOBench/configs/models/${cand}.yaml" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

# Default values
MODEL_NAME=""
MODEL_REVISION="main"
AGENT_NAME=""
JUDGE_NAME="google/gemini-3-pro"
GEN_REASONING_EFFORT=""
JUDGE_REASONING_EFFORT=""
FINAL_ANSWER_FLAG=""
PROOFBENCH=false
DATA_PATH=""
SPLIT=""
NUM_NODES=1
NUM_GPUS=8
OVERWRITE=false
TEMPERATURE=""
TOP_P=""
TOP_K=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_usage
            exit 0
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-revision)
            MODEL_REVISION="$2"
            shift 2
            ;;
        --agent)
            AGENT_NAME="$2"
            shift 2
            ;;
        --judge)
            JUDGE_NAME="$2"
            shift 2
            ;;
        --gen-reasoning-effort)
            GEN_REASONING_EFFORT="$2"
            shift 2
            ;;
        --judge-reasoning-effort)
            JUDGE_REASONING_EFFORT="$2"
            shift 2
            ;;
        --final-answer)
            FINAL_ANSWER_FLAG="--final-answer"
            shift
            ;;
        --proofbench)
            PROOFBENCH=true
            shift
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model is required"
    print_usage
    exit 1
fi

# Validate numeric arguments
if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || [ "$NUM_NODES" -lt 1 ]; then
    echo "ERROR: --num-nodes must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: --num-gpus must be a positive integer"
    exit 1
fi

# Validate reasoning effort arguments if provided
if [ -n "$GEN_REASONING_EFFORT" ] && [[ ! "$GEN_REASONING_EFFORT" =~ ^(high|medium|low)$ ]]; then
    echo "ERROR: --gen-reasoning-effort must be 'high', 'medium', or 'low'"
    exit 1
fi

if [ -n "$JUDGE_REASONING_EFFORT" ] && [[ ! "$JUDGE_REASONING_EFFORT" =~ ^(high|medium|low)$ ]]; then
    echo "ERROR: --judge-reasoning-effort must be 'high', 'medium', or 'low'"
    exit 1
fi

if [ "$PROOFBENCH" = true ] && [ -n "$FINAL_ANSWER_FLAG" ]; then
    echo "ERROR: --proofbench cannot be combined with --final-answer"
    exit 1
fi

if [ "$PROOFBENCH" = true ] && [ -z "$DATA_PATH" ]; then
    DATA_PATH="lm-provers/ProofBench"
fi

# Build args for inference (run.py)
IMOBENCH_ARGS=()
if [ -n "$FINAL_ANSWER_FLAG" ]; then
    IMOBENCH_ARGS+=(--final-answer)
fi
if [ -n "$DATA_PATH" ]; then
    IMOBENCH_ARGS+=(--data-path "$DATA_PATH")
fi
if [ -n "$SPLIT" ]; then
    IMOBENCH_ARGS+=(--split "$SPLIT")
fi

# Build args for evaluation (eval.py)
EVAL_FLAGS=()
if [ -n "$FINAL_ANSWER_FLAG" ]; then
    EVAL_FLAGS+=(--final-answer)
fi
if [ "$PROOFBENCH" = true ]; then
    EVAL_FLAGS+=(--proofbench)
fi

# Model reasoning effort (used for config selection / naming)
if [ -n "$GEN_REASONING_EFFORT" ]; then
    echo "Model reasoning effort: $GEN_REASONING_EFFORT"
fi

# Determine whether --model refers to an existing config (API or vLLM), otherwise treat it as a HF model id and use vLLM.
USE_API_MODEL=false
MODEL_CONFIG_NAME=""
MODEL_CONFIG_PATH=""
CONFIG_FILE=""
MODEL_CONFIG_IS_EXPLICIT=false

MODEL_NAME_STRIPPED="$MODEL_NAME"
if [[ "$MODEL_NAME_STRIPPED" == *.yaml ]]; then
    MODEL_NAME_STRIPPED="${MODEL_NAME_STRIPPED%.yaml}"
fi

RESOLVED_MODEL_CONFIG_PATH=""
if RESOLVED_MODEL_CONFIG_PATH=$(resolve_existing_model_config_path "$MODEL_NAME_STRIPPED" "$GEN_REASONING_EFFORT"); then
    MODEL_CONFIG_IS_EXPLICIT=true
    MODEL_CONFIG_PATH="$RESOLVED_MODEL_CONFIG_PATH"
    MODEL_CONFIG_NAME="$MODEL_CONFIG_PATH"
    CONFIG_FILE="IMOBench/configs/models/${MODEL_CONFIG_PATH}.yaml"

    # Treat as vLLM if config points at a local OpenAI-compatible server.
    if grep -q "api: custom" "$CONFIG_FILE" && grep -Eq "(localhost|127\\.0\\.0\\.1)" "$CONFIG_FILE"; then
        USE_API_MODEL=false
        echo "Model uses existing vLLM config: $MODEL_CONFIG_PATH"
    else
        USE_API_MODEL=true
        echo "Model uses existing API config: $MODEL_CONFIG_PATH"
    fi
else
    # vLLM model (HuggingFace path): generate config name from model id
    MODEL_CONFIG_NAME=$(echo "$MODEL_NAME" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    if [[ "$MODEL_NAME" =~ / ]]; then
        MODEL_CONFIG_NAME="vllm-${MODEL_CONFIG_NAME}"
    fi
    if [ -n "$GEN_REASONING_EFFORT" ]; then
        MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME}_${GEN_REASONING_EFFORT}"
    fi
    MODEL_CONFIG_PATH="vllm/${MODEL_CONFIG_NAME}"
    CONFIG_FILE="IMOBench/configs/models/${MODEL_CONFIG_PATH}.yaml"
    echo "Model is vLLM: $MODEL_NAME"
fi

# Validate agent scaffold if provided
if [ -n "$AGENT_NAME" ]; then
    if [ "$USE_API_MODEL" = true ]; then
        echo "ERROR: --agent requires a vLLM-served model (HuggingFace path). Got API model: $MODEL_NAME"
        exit 1
    fi
    AGENT_SCAFFOLD_FILE="configs/agents/${AGENT_NAME}.yaml"
    if [ ! -f "$AGENT_SCAFFOLD_FILE" ]; then
        echo "ERROR: Agent scaffold config not found: $AGENT_SCAFFOLD_FILE"
        echo "Available agent scaffolds:"
        ls -1 configs/agents/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//' | sed 's/^/  /' || true
        exit 1
    fi
fi

# Determine if judge uses external API or vLLM
# External APIs: gpt, gemini, claude, grok
# vLLM: everything else (HuggingFace models)

# First check if judge name is a config path (including provider subdir),
# otherwise treat it as a HuggingFace model path when it contains "/".
JUDGE_CONFIG_NAME="$JUDGE_NAME"
JUDGE_CONFIG_PATH="$JUDGE_NAME"
JUDGE_CONFIG_IS_VLLM=false
JUDGE_MODEL_PATH=""
JUDGE_OUTPUT_NAME=""
JUDGE_NAME_STRIPPED="$JUDGE_NAME"
if [[ "$JUDGE_NAME" == *.yaml ]]; then
    JUDGE_NAME_STRIPPED="${JUDGE_NAME%.yaml}"
fi
JUDGE_CONFIG_CANDIDATE="configs/models/${JUDGE_NAME_STRIPPED}.yaml"
JUDGE_VLLM_CONFIG_CANDIDATE="configs/models/vllm/${JUDGE_NAME_STRIPPED}.yaml"

if [ -f "$JUDGE_CONFIG_CANDIDATE" ]; then
    JUDGE_CONFIG_NAME="$JUDGE_NAME_STRIPPED"
    JUDGE_CONFIG_PATH="$JUDGE_CONFIG_NAME"
elif [ -f "$JUDGE_VLLM_CONFIG_CANDIDATE" ]; then
    JUDGE_CONFIG_NAME="$JUDGE_NAME_STRIPPED"
    JUDGE_CONFIG_IS_VLLM=true
else
    # If judge name contains "/" it's a HuggingFace path, create vLLM config name
    if [[ "$JUDGE_NAME" =~ / ]]; then
        JUDGE_MODEL_PATH="$JUDGE_NAME"
        JUDGE_CONFIG_NAME=$(echo "$JUDGE_NAME" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
        JUDGE_CONFIG_NAME="vllm-${JUDGE_CONFIG_NAME}"
        JUDGE_CONFIG_IS_VLLM=true
        echo "Judge is HuggingFace model: $JUDGE_MODEL_PATH"
        echo "Generated config name: $JUDGE_CONFIG_NAME"
    else
        # It's already a config name
        JUDGE_CONFIG_NAME="$JUDGE_NAME"
        JUDGE_CONFIG_PATH="$JUDGE_CONFIG_NAME"
    fi
fi

# Add reasoning effort suffix to judge config name if provided
if [ -n "$JUDGE_REASONING_EFFORT" ]; then
    JUDGE_CONFIG_NAME="${JUDGE_CONFIG_NAME}_${JUDGE_REASONING_EFFORT}"
    echo "Judge reasoning effort: $JUDGE_REASONING_EFFORT"
fi

if [ "$JUDGE_CONFIG_IS_VLLM" = true ]; then
    JUDGE_CONFIG_PATH="vllm/${JUDGE_CONFIG_NAME}"
else
    JUDGE_CONFIG_PATH="$JUDGE_CONFIG_NAME"
fi

JUDGE_CONFIG_FILE="configs/models/${JUDGE_CONFIG_PATH}.yaml"

# Safe name for filenames / job names (avoid slashes)
JUDGE_OUTPUT_NAME=$(echo "$JUDGE_CONFIG_NAME" | tr '/' '-')
USE_VLLM_JUDGE=false

# Check if judge config exists
if [ ! -f "$JUDGE_CONFIG_FILE" ]; then
    echo "Judge config not found: $JUDGE_CONFIG_FILE"

    # Check if judge name looks like it should use external API
    if [[ "$JUDGE_CONFIG_NAME" =~ ^(openai/gpt|google/gemini|other/grok) ]]; then
        echo "ERROR: External API judge config not found: $JUDGE_CONFIG_FILE"
        echo "Available external API configs:"
        list_external_api_configs | sed 's/^/  /' || true
        exit 1
    fi
    # Otherwise, treat as HuggingFace model and create vLLM config
    # If JUDGE_MODEL_PATH is not set, use JUDGE_NAME as the model path
    if [ -z "$JUDGE_MODEL_PATH" ]; then
        JUDGE_MODEL_PATH="$JUDGE_NAME"
    fi

    # Activate venv before calling the function (needs Python)
    source .venv/bin/activate

    # Use function to create config
    create_vllm_config "$JUDGE_MODEL_PATH" "$JUDGE_CONFIG_NAME" "$JUDGE_REASONING_EFFORT" "" "" ""
    USE_VLLM_JUDGE=true
else
    # Config exists, check if it uses vLLM (has custom API and localhost base_url)
    if grep -q "api: custom" "$JUDGE_CONFIG_FILE" && grep -Eq "(localhost|127\\.0\\.0\\.1)" "$JUDGE_CONFIG_FILE"; then
        USE_VLLM_JUDGE=true
    fi
fi

# Add revision to output paths if not "main"
if [ "$MODEL_REVISION" != "main" ]; then
    # Sanitize revision name for filesystem (replace / with -, remove special chars)
    REVISION_SUFFIX=$(echo "$MODEL_REVISION" | tr '/' '-' | tr -cd '[:alnum:]-_')
    OUTPUT_SUFFIX="-${REVISION_SUFFIX}"
else
    OUTPUT_SUFFIX=""
fi
JOB_NAME_SUFFIX="$OUTPUT_SUFFIX"

INFERENCE_SCRIPT="slurm/launch_imobench.slurm"

# Benchmark display info
if [ "$PROOFBENCH" = true ]; then
    BENCHMARK_MODE="ProofBench (proof grading)"
elif [ -n "$FINAL_ANSWER_FLAG" ]; then
    BENCHMARK_MODE="IMOBench Final Answer"
else
    BENCHMARK_MODE="IMOProofBench"
fi

if [ "$PROOFBENCH" = true ]; then
    BENCHMARK_SLUG="proofbench"
elif [ -n "$FINAL_ANSWER_FLAG" ]; then
    BENCHMARK_SLUG="imobench-final"
else
    BENCHMARK_SLUG="imoproofbench"
fi

if [ -n "$DATA_PATH" ]; then
    DATA_SOURCE="$DATA_PATH"
elif [ "$PROOFBENCH" = true ]; then
    DATA_SOURCE="lm-provers/ProofBench (default)"
elif [ -n "$FINAL_ANSWER_FLAG" ]; then
    DATA_SOURCE="lm-provers/IMOBench-FinalAnswer (default)"
else
    DATA_SOURCE="lm-provers/IMOBench-ProofBench (default)"
fi

# Set output paths (nest by benchmark for clarity)
timestamp=$(date +%Y%m%d-%H%M%S)
BENCHMARK_OUTPUT_DIR="outputs/${BENCHMARK_SLUG}"
AGENT_SUFFIX=""
if [ -n "$AGENT_NAME" ]; then
    AGENT_SLUG=$(echo "$AGENT_NAME" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-_')
    AGENT_SUFFIX="-agent-${AGENT_SLUG}"
fi
OUTPUT_FILENAME="${MODEL_CONFIG_NAME}${AGENT_SUFFIX}${OUTPUT_SUFFIX}_${timestamp}.jsonl"
EVAL_OUTPUT_FILENAME="${MODEL_CONFIG_NAME}${AGENT_SUFFIX}${OUTPUT_SUFFIX}-${JUDGE_OUTPUT_NAME}_${timestamp}.jsonl"
OUTPUT_PATH="${BENCHMARK_OUTPUT_DIR}/${OUTPUT_FILENAME}"
EVAL_OUTPUT_PATH="${BENCHMARK_OUTPUT_DIR}/${EVAL_OUTPUT_FILENAME}"

# Ensure benchmark-specific output directory exists
mkdir -p "${BENCHMARK_OUTPUT_DIR}"

# Default: run the same config we serve
RUN_MODEL_CONFIG_NAME="$MODEL_CONFIG_PATH"

# Create model config if using vLLM
if [ "$USE_API_MODEL" = false ]; then
    # Activate venv (needed for infer_max_tokens.py)
    source .venv/bin/activate

    # Create/refresh model config file only if needed (avoid clobbering hand-edited configs).
    # Refresh when explicit sampling parameters are provided, or when config doesn't exist.
    NEED_REFRESH_CONFIG=false
    if [ ! -f "$CONFIG_FILE" ]; then
        NEED_REFRESH_CONFIG=true
    fi
    if [ -n "$TEMPERATURE" ] || [ -n "$TOP_P" ] || [ -n "$TOP_K" ]; then
        NEED_REFRESH_CONFIG=true
    fi

    if [ "$MODEL_CONFIG_IS_EXPLICIT" = false ] && [ "$NEED_REFRESH_CONFIG" = true ]; then
        create_vllm_config "$MODEL_NAME" "$MODEL_CONFIG_NAME" "$GEN_REASONING_EFFORT" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MODEL_REVISION"
    else
        echo "Using existing vLLM config: $CONFIG_FILE"
    fi

    # Extract max_tokens from the created config for display
    MAX_TOKENS=$(grep '^max_tokens:' "$CONFIG_FILE" | sed 's/max_tokens: *//' | xargs)

    # If an agent scaffold is requested, generate a derived agent model config that points at this vLLM config.
    if [ -n "$AGENT_NAME" ]; then
        DERIVED_AGENT_CONFIG_FILE="configs/models/agents/${MODEL_CONFIG_NAME}/${AGENT_NAME}.yaml"
        mkdir -p "$(dirname "$DERIVED_AGENT_CONFIG_FILE")"
        cat > "$DERIVED_AGENT_CONFIG_FILE" <<EOF
type: agent
model_config: models/${MODEL_CONFIG_PATH}
scaffold_config: agents/${AGENT_NAME}
human_readable_id: "${MODEL_NAME} + ${AGENT_NAME} (${MODEL_REVISION})"
EOF
        RUN_MODEL_CONFIG_NAME="agents/${MODEL_CONFIG_NAME}/${AGENT_NAME}"
        echo "Created derived agent model config: $DERIVED_AGENT_CONFIG_FILE"
    fi

    # Determine which SLURM script to use
    if [ "$NUM_NODES" -gt 1 ]; then
        VLLM_SLURM_SCRIPT="slurm/launch_imobench_vllm_serve_multinode.slurm"
        DEPLOYMENT_TYPE="multi-node"
    else
        VLLM_SLURM_SCRIPT="slurm/launch_imobench_vllm_serve.slurm"
        DEPLOYMENT_TYPE="single-node"
    fi

    TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
    INFERENCE_SCRIPT="$VLLM_SLURM_SCRIPT"
else
    # For API models, check config exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: API model config not found: $CONFIG_FILE"
        echo "Available API configs:"
        list_external_api_configs | sed 's/^/  /' || true
        exit 1
    fi
    MAX_TOKENS=$(grep '^max_tokens:' "$CONFIG_FILE" | sed 's/max_tokens: *//' | xargs)
    DEPLOYMENT_TYPE="API (CPU)"
    TOTAL_GPUS=0
fi

echo "================================================================"
echo "Creating model config: $CONFIG_FILE"
if [ -n "$AGENT_NAME" ]; then
    echo "Agent scaffold: $AGENT_NAME"
    echo "Run model config: configs/models/${RUN_MODEL_CONFIG_NAME}.yaml"
fi
echo "Model: $MODEL_NAME"
echo "Model revision: $MODEL_REVISION"
echo "Max tokens: $MAX_TOKENS"
echo "Gen reasoning effort: $([ -n "$GEN_REASONING_EFFORT" ] && echo "$GEN_REASONING_EFFORT" || echo "not set")"
echo "Inference output: $OUTPUT_PATH"
echo "Evaluation output: $EVAL_OUTPUT_PATH"
echo "Judge: $JUDGE_NAME"
echo "Judge type: $([ "$USE_VLLM_JUDGE" = true ] && echo "vLLM (local)" || echo "External API")"
echo "Judge reasoning effort: $([ -n "$JUDGE_REASONING_EFFORT" ] && echo "$JUDGE_REASONING_EFFORT" || echo "not set")"
echo "Benchmark: $BENCHMARK_MODE"
echo "Data path: $DATA_SOURCE"
if [ -n "$SPLIT" ]; then
    echo "Dataset split: $SPLIT"
fi
echo "Final answer mode: $([ -n "$FINAL_ANSWER_FLAG" ] && echo "enabled" || echo "disabled")"
echo "Deployment: $DEPLOYMENT_TYPE (${NUM_NODES} node(s) × ${NUM_GPUS} GPU(s) = ${TOTAL_GPUS} total GPUs)"
echo "SLURM script: $INFERENCE_SCRIPT"
echo "================================================================"
echo ""

# Check if output file from job 1 already exists
SKIP_JOB1=false
if [ -f "$OUTPUT_PATH" ] && [ "$OVERWRITE" = false ]; then
    echo "================================================================"
    echo "OUTPUT FILE ALREADY EXISTS: $OUTPUT_PATH"
    echo "Skipping Job 1 (inference) and queuing Job 2 (evaluation) only"
    echo "Use --overwrite to regenerate inference output"
    echo "================================================================"
    echo ""
    SKIP_JOB1=true
fi
# replace / in model config name with - for job names and filenames
MODEL_CONFIG_NAME=$(echo "$MODEL_CONFIG_NAME" | tr '/' '-')

# Submit first job (inference)
if [ "$SKIP_JOB1" = false ]; then
echo "================================================================"
if [ "$USE_API_MODEL" = true ]; then
    echo "Submitting Job 1: API inference (CPU)"
    echo "================================================================"

    INFERENCE_JOB_NAME="imobench-${MODEL_CONFIG_NAME}${JOB_NAME_SUFFIX}"

    JOB1_ID=$(sbatch --parsable \
        --job-name="$INFERENCE_JOB_NAME" \
        slurm/IMOBench/launch_imobench.slurm \
        --model-config "$MODEL_CONFIG_PATH" \
        --output-path "$OUTPUT_PATH" \
        "${IMOBENCH_ARGS[@]}")
else
    echo "Submitting Job 1: vLLM server + inference ($DEPLOYMENT_TYPE)"
    echo "================================================================"

    INFERENCE_JOB_NAME="imobench-vllm-serve-${MODEL_CONFIG_NAME}${JOB_NAME_SUFFIX}"
    if [ -n "$AGENT_NAME" ]; then
        INFERENCE_JOB_NAME="${INFERENCE_JOB_NAME}-agent"
    fi

    # Prepare sbatch command with dynamic node/gpu configuration
    SBATCH_ARGS="--parsable --job-name=$INFERENCE_JOB_NAME --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS"

    if [ -n "$AGENT_NAME" ]; then
        JOB1_ID=$(sbatch $SBATCH_ARGS \
            "$VLLM_SLURM_SCRIPT" \
            --serve-config "$MODEL_CONFIG_PATH" \
            --run-config "$RUN_MODEL_CONFIG_NAME" \
            --output-path "$OUTPUT_PATH" \
            "${IMOBENCH_ARGS[@]}" \
            --revision "$MODEL_REVISION")
    else
        JOB1_ID=$(sbatch $SBATCH_ARGS \
            "$VLLM_SLURM_SCRIPT" \
            --serve-config "$MODEL_CONFIG_PATH" \
            --run-config "$RUN_MODEL_CONFIG_NAME" \
            --output-path "$OUTPUT_PATH" \
            "${IMOBENCH_ARGS[@]}" \
            --revision "$MODEL_REVISION")
    fi
fi

    echo "Job 1 submitted with ID: $JOB1_ID"
    echo ""
else
    echo "Job 1 skipped (output file already exists)"
    echo ""
    JOB1_ID=""
fi

# Submit second job (evaluation) with dependency on first job (if applicable)
echo "================================================================"
if [ "$SKIP_JOB1" = false ]; then
    echo "Submitting Job 2: Evaluation (depends on Job 1)"
else
    echo "Submitting Job 2: Evaluation (no dependency)"
fi
echo "================================================================"

EVAL_JOB_NAME="imobench-eval-${MODEL_CONFIG_NAME}${JOB_NAME_SUFFIX}-${JUDGE_OUTPUT_NAME}"

# Choose appropriate eval script based on judge type
if [ "$USE_VLLM_JUDGE" = true ]; then
    EVAL_SCRIPT="slurm/launch_imobench_eval_vllm_serve.slurm"
    echo "Using vLLM eval script (will start local vLLM server for judge)"
else
    EVAL_SCRIPT="slurm/launch_imobench_eval.slurm"
    echo "Using standard eval script (external API)"
fi

# Build sbatch command with optional dependency
if [ "$SKIP_JOB1" = false ]; then
    JOB2_ID=$(sbatch --parsable \
        --dependency=afterok:$JOB1_ID \
        --job-name="$EVAL_JOB_NAME" \
        "$EVAL_SCRIPT" \
        --model-config "$JUDGE_CONFIG_PATH" \
        --data-path "$OUTPUT_PATH" \
        --output-path "$EVAL_OUTPUT_PATH" \
        "${EVAL_FLAGS[@]}")
    echo "Job 2 submitted with ID: $JOB2_ID (will run after Job 1 completes)"
else
    JOB2_ID=$(sbatch --parsable \
        --job-name="$EVAL_JOB_NAME" \
        "$EVAL_SCRIPT" \
        --model-config "$JUDGE_CONFIG_PATH" \
        --data-path "$OUTPUT_PATH" \
        --output-path "$EVAL_OUTPUT_PATH" \
        "${EVAL_FLAGS[@]}")
    echo "Job 2 submitted with ID: $JOB2_ID (will run immediately)"
fi
echo ""

# Summary
echo "================================================================"
echo "Pipeline queued successfully!"
echo "================================================================"
echo "Model config: $CONFIG_FILE"
if [ -n "$AGENT_NAME" ]; then
    echo "Run model config: configs/models/${RUN_MODEL_CONFIG_NAME}.yaml"
fi
echo "Judge config: $JUDGE_CONFIG_FILE"
if [ "$SKIP_JOB1" = false ]; then
    echo "Job 1 (inference): $JOB1_ID (script: $INFERENCE_SCRIPT)"
    echo "Job 2 (evaluation): $JOB2_ID (script: $EVAL_SCRIPT, depends on $JOB1_ID)"
else
    echo "Job 1 (inference): SKIPPED (output exists: $OUTPUT_PATH)"
    echo "Job 2 (evaluation): $JOB2_ID (script: $EVAL_SCRIPT, no dependency)"
fi
echo "Inference output: $OUTPUT_PATH"
echo "Evaluation output: $EVAL_OUTPUT_PATH"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: /fsx/h4/logs/"
echo "================================================================"
