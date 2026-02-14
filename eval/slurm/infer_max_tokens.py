#!/usr/bin/env python3
"""
Infer max_tokens from HuggingFace model config
"""
import sys
import json
from huggingface_hub import hf_hub_download

def _find_context_length(config: dict) -> int | None:
    # Common field names for context length
    keys = [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "max_seq_len",
        "max_seq_length",
        "model_max_length",
    ]
    for key in keys:
        value = config.get(key)
        if isinstance(value, int):
            return value
    # Some configs nest under text_config
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        for key in keys:
            value = text_config.get(key)
            if isinstance(value, int):
                return value
    return None


def _choose_prompt_margin(max_ctx: int, min_margin: int = 4096, max_margin: int = 32768) -> int:
    # Reserve ~12.5% of the context for prompts (bounded), with a minimum margin.
    margin = max(min_margin, max_ctx // 8)
    margin = min(max_margin, margin)
    # Ensure we always leave at least 256 tokens for generation.
    margin = min(margin, max(0, max_ctx - 256))
    return max(0, margin)


def infer_max_tokens_and_margin(model_id: str, default_max_tokens: int = 28672):
    """
    Infer `max_tokens` (generation) and a `prompt_margin` from a model config.

    Returns:
        (max_tokens, prompt_margin)
    """
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        max_ctx = _find_context_length(config)
        if max_ctx is None:
            return default_max_tokens, 512

        prompt_margin = _choose_prompt_margin(max_ctx)
        max_tokens = max(0, max_ctx - prompt_margin)
        return max_tokens, prompt_margin

    except Exception as e:
        print(f"Warning: Could not fetch config for {model_id}: {e}", file=sys.stderr)
        print(f"Using default max_tokens: {default_max_tokens}", file=sys.stderr)
        return default_max_tokens, 512

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_max_tokens.py MODEL_ID", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]
    max_tokens, prompt_margin = infer_max_tokens_and_margin(model_id)
    print(f"{max_tokens} {prompt_margin}")
