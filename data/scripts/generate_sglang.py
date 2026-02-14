#!/usr/bin/env python3
"""
Load a Hugging Face dataset, hit an OpenAI-compatible chat endpoint asynchronously,
and save the results to a JSONL file.

Example:
python scripts/generate_sglang.py \
  --dataset hf-imo-colab/olympiads-proof-schema-cleaned \
  --split train \
  --text-column problem \
  --model deepseek-ai/DeepSeek-Math-V2 \
  --base-url http://ip-26-0-160-100:39877/v1 \
  --max-tokens 16 \
  --outfile-json-filename test-outputs.jsonl \
  --output-dataset-name edbeeching/olympiads-proof-schema-cleaned-completions \
  --push-every 10 \
  --requests-per-prompt 3

"""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from openai import AsyncOpenAI
from openai import OpenAIError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async chat completions over a HF dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name or path for datasets.load_dataset.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--text-column", default="text", help="Column name used as the user message.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt to prefix conversations.")
    parser.add_argument("--offset", type=int, default=0, help="Start index within the split.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("--model", required=True, help="Model name to send to the chat endpoint.")
    parser.add_argument("--outfile-json-filename", required=True, help="Where to write the JSONL results.")
    parser.add_argument("--base-url", required=True, default=None, help="OpenAI-compatible base URL (e.g., http://host:port/v1).")
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate per call.")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of concurrent in-flight requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample on transient failures.")
    parser.add_argument("--retry-initial-backoff", type=float, default=1.0, help="Seconds to back off after first failure.")
    parser.add_argument("--retry-backoff-multiplier", type=float, default=2.0, help="Backoff growth factor.")
    parser.add_argument("--request-timeout", type=float, default=7200.0, help="Per-request timeout in seconds.")
    parser.add_argument("--enable-thinking", action="store_true", help="Set chat_template_kwargs.enable_thinking=True.")
    parser.add_argument("--output-dataset-name", default=None, help="Hugging Face dataset repository to stream completions to.")
    parser.add_argument("--push-every", type=int, default=50, help="Push to the Hub after this many new samples.")
    parser.add_argument("--requests-per-prompt", type=int, default=1, help="How many independent requests to send per dataset row.")
    return parser.parse_args()


def build_messages(user_content: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


async def chat_with_retry(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    initial_backoff: float,
    backoff_multiplier: float,
    extra_body: Optional[Dict[str, Any]],
) -> Any:
    def _is_retryable(exc: Exception) -> bool:
        name = exc.__class__.__name__
        text = str(exc)
        retryable_names = {
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
            "RateLimitError",
        }
        retryable_phrases = [
            "timed out",
            "No available workers",
            "all circuits open",
            "temporarily overloaded",
        ]
        if name in retryable_names:
            return True
        return any(phrase.lower() in text.lower() for phrase in retryable_phrases)

    backoff = max(initial_backoff, 0.0)
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1 or not _is_retryable(exc):
                raise exc
            sleep_for = backoff if backoff > 0 else 0
            await asyncio.sleep(sleep_for)
            backoff = max(backoff * backoff_multiplier, 0.0) if backoff_multiplier else backoff


class StreamingJSONLWriter:
    """Append objects into a JSONL file safely with an async lock."""

    def __init__(self, path: str):
        self.path = path
        self.lock = asyncio.Lock()
        self._file = open(self.path, "w", encoding="utf-8")

    async def write(self, obj: Dict[str, Any]) -> None:
        async with self.lock:
            json.dump(obj, self._file, ensure_ascii=False)
            self._file.write("\n")
            self._file.flush()

    async def close(self) -> None:
        async with self.lock:
            self._file.close()


class StreamingHFDatasetWriter:
    """Push generated samples to a Hugging Face dataset repository incrementally."""

    def __init__(
        self,
        dataset_id: str,
        split: str,
        push_every: int,
    ):
        self.dataset_id = dataset_id
        self.config_name = "default"
        self.split = split
        self.private = True
        self.push_every = max(push_every, 1)
        self.lock = asyncio.Lock()
        self.push_lock = asyncio.Lock()
        self.rows: List[Dict[str, Any]] = []
        self._pending_since_push = 0
        self._last_pushed_len = 0

    async def write(self, obj: Dict[str, Any]) -> None:
        snapshot: Optional[List[Dict[str, Any]]] = None
        async with self.lock:
            self.rows.append(obj)
            self._pending_since_push += 1
            if self._pending_since_push >= self.push_every:
                snapshot = list(self.rows)
                self._pending_since_push = 0

        if snapshot is not None:
            await self._push_snapshot(snapshot)

    async def _push_snapshot(self, snapshot: List[Dict[str, Any]]) -> None:
        async with self.push_lock:
            try:
                dataset = Dataset.from_list(snapshot)
                commit_message = f"Update {self.split} with {len(snapshot)} samples"
                await asyncio.to_thread(
                    dataset.push_to_hub,
                    self.dataset_id,
                    config_name=self.config_name,
                    split=self.split,
                    private=self.private,
                    commit_message=commit_message,
                )
                self._last_pushed_len = len(snapshot)
                print(f"Pushed {len(snapshot)} samples to {self.dataset_id}:{self.split} ({self.config_name}).")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to push dataset to {self.dataset_id}:{self.split}: {exc}")

    async def close(self) -> None:
        snapshot: Optional[List[Dict[str, Any]]] = None
        async with self.lock:
            if len(self.rows) > self._last_pushed_len:
                snapshot = list(self.rows)
                self._pending_since_push = 0

        if snapshot is not None:
            await self._push_snapshot(snapshot)


class CompositeWriter:
    """Fan out writes to multiple writers."""

    def __init__(self, writers: List[Any]):
        self.writers = writers

    async def write(self, obj: Dict[str, Any]) -> None:
        await asyncio.gather(*(writer.write(obj) for writer in self.writers))

    async def close(self) -> None:
        await asyncio.gather(*(writer.close() for writer in self.writers))


async def process_sample(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    sample_idx: int,
    request_idx: int,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    text_column: str,
    writer: Any,
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if text_column not in sample:
        raise KeyError(f"Column '{text_column}' not found in dataset row {sample_idx}.")

    user_text = sample[text_column]
    messages = build_messages(user_text, args.system_prompt)

    async with semaphore:
        try:
            response = await chat_with_retry(
                client=client,
                messages=messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                initial_backoff=args.retry_initial_backoff,
                backoff_multiplier=args.retry_backoff_multiplier,
                extra_body=extra_body,
            )
            choice = response.choices[0].message
            result = {
                "index": sample_idx,
                "request_idx": request_idx,
                "prompt": user_text,
                "messages": messages,
                "completion": choice.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": response.model_dump(),
                "extra_body": extra_body,
            }
            await writer.write(result)
            return result
        except OpenAIError as exc:
            result = {
                "index": sample_idx,
                "request_idx": request_idx,
                "prompt": user_text,
                "messages": messages,
                "error": f"{exc.__class__.__name__}: {exc}",
                "extra_body": extra_body,
            }
            await writer.write(result)
            return result
        except Exception as exc:  # noqa: BLE001
            result = {
                "index": sample_idx,
                "request_idx": request_idx,
                "prompt": user_text,
                "messages": messages,
                "error": f"UnexpectedError: {exc}",
                "extra_body": extra_body,
            }
            await writer.write(result)
            return result


async def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, split=args.split)
    start = max(args.offset, 0)
    end = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    dataset_slice = dataset.select(range(start, end))

    client = AsyncOpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )
    semaphore = asyncio.Semaphore(max(args.concurrency, 1))
    writers: List[Any] = [StreamingJSONLWriter(args.outfile_json_filename)]
    if args.output_dataset_name:
        hf_writer = StreamingHFDatasetWriter(
            dataset_id=args.output_dataset_name,
            split=args.split,
            push_every=args.push_every,
        )
        writers.append(hf_writer)

    writer = CompositeWriter(writers) if len(writers) > 1 else writers[0]
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if args.enable_thinking else None

    per_prompt = max(args.requests_per_prompt, 1)
    tasks = [
        process_sample(
            semaphore,
            client,
            start + idx,
            request_idx,
            sample,
            args,
            args.text_column,
            writer,
            extra_body,
        )
        for idx, sample in enumerate(dataset_slice)
        for request_idx in range(per_prompt)
    ]

    print(f"Dispatching {len(tasks)} samples from {args.dataset}:{args.split} to {args.model} ...")
    try:
        await asyncio.gather(*tasks)
    finally:
        await writer.close()

    print(f"Wrote {len(tasks)} records to {args.outfile_json_filename}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
