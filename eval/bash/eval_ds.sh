export VLLM_API_KEY=EMPTY  # or any dummy value your client expects

uv run --active python scripts/run.py --model-config workflows/dsm --output-path outputs/dsm_eval.jsonl