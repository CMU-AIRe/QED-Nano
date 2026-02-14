import asyncio
import json
import logging
import math
import multiprocessing as mp
import os
import queue
from queue import Empty
import random
import time
from collections import defaultdict, deque
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import List, Dict, Any
from pipelinerl.utils import strip_chat_template_tokens
import aiohttp
import aiohttp.client_exceptions
import hydra
import numpy as np
import uvloop
import copy

# Transient errors that should not crash the entire actor when retries are exhausted.
# Instead, we re-queue the rollout to try again later.
TRANSIENT_EXCEPTIONS = (
    aiohttp.client_exceptions.ClientError,  # Base class for all aiohttp client errors
    aiohttp.client_exceptions.ClientPayloadError,  # Response payload incomplete
    aiohttp.client_exceptions.ClientOSError,  # Connection errors
    aiohttp.client_exceptions.ServerDisconnectedError,  # Server closed connection
    asyncio.TimeoutError,  # Request timeout
    ConnectionError,  # Base connection errors
    TimeoutError,  # Base timeout errors
)

# Maximum number of times to re-queue a rollout after transient errors before giving up
MAX_REQUEUE_ATTEMPTS = 10
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from tapeagents.llms import TrainableLLM

import wandb
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.shared_memory_array import SharedMemoryQueue
from pipelinerl.state import TrainerState
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamSpec,
    StreamWriter,
    set_streams_backend,
    write_to_streams,
)

from .utils import (
    always_or_never_success_stats,
    calculate_stats,
    setup_logging,
    wait_for_environments,
    wait_for_inference_servers,
)

from pipelinerl.rollouts import TrainingText

logger = logging.getLogger(__name__)


_WANDB_VERIFIER_TABLE = None
_WANDB_VERIFIER_TABLE_COLUMNS = ["group_index", "prompt", "reasoning", "output", "score"]


def _get_wandb_verifier_table():
    global _WANDB_VERIFIER_TABLE
    if getattr(wandb, "run", None) is None:
        return None
    if _WANDB_VERIFIER_TABLE is None:
        _WANDB_VERIFIER_TABLE = wandb.Table(columns=_WANDB_VERIFIER_TABLE_COLUMNS, log_mode="MUTABLE")
    return _WANDB_VERIFIER_TABLE


def _log_verifier_table_entry(entry: dict[str, str | int]):
    table = _get_wandb_verifier_table()
    if table is None:
        return
    table.add_data(
        entry.get("group_index", 0),
        entry.get("prompt", ""),
        entry.get("reasoning", ""),
        entry.get("output_text", ""),
        entry.get("score", 0),
    )
    wandb.log({"tables/rc_actor/verifier": table})


class VerifierTableBuffer:
    """
    A bounded ring-buffer for verifier table entries.

    Keeps only the last `k` groups' worth of rows. Each group can have multiple
    rows (one per rollout). When a new group is added and the buffer exceeds `k`
    groups, the oldest group is evicted.
    """

    def __init__(self, keep_last_k_groups: int = 32, log_every_n_groups: int = 32):
        self.keep_last_k_groups = max(0, int(keep_last_k_groups))
        self.log_every_n_groups = max(1, int(log_every_n_groups))
        self._groups: deque[list[dict[str, str | int]]] = deque()
        self._groups_added = 0

    def add_group(self, entries: list[dict[str, str | int]]) -> None:
        """Add a group of entries (rows) to the buffer."""
        if not entries:
            return
        self._groups.append(entries)
        self._groups_added += 1
        while len(self._groups) > self.keep_last_k_groups:
            self._groups.popleft()

    def should_log(self) -> bool:
        """Return True if we should log the table this group."""
        return self._groups_added % self.log_every_n_groups == 0

    def to_wandb_table(self) -> "wandb.Table":
        """Build a wandb.Table from all rows currently in the buffer."""
        table = wandb.Table(columns=_WANDB_VERIFIER_TABLE_COLUMNS)
        for group_entries in self._groups:
            for entry in group_entries:
                table.add_data(
                    entry.get("group_index", 0),
                    entry.get("prompt", ""),
                    entry.get("reasoning", ""),
                    entry.get("output_text", ""),
                    entry.get("score", 0),
                )
        return table

    def log_to_wandb(self) -> None:
        """Publish the current buffer as a table via wandb.log()."""
        if getattr(wandb, "run", None) is None:
            return
        table = self.to_wandb_table()
        wandb.log({"tables/rc_actor/verifier_last_k": table})


def _aggregate_group_verifier_metrics(rollout_results: list[list[RolloutResult]]) -> dict[str, float | int]:
    runtime_values: defaultdict[str, list[float]] = defaultdict(list)
    count_totals: defaultdict[str, int] = defaultdict(int)
    for attempt in rollout_results:
        for result in attempt:
            metrics = getattr(result, "verifier_metrics", {}) or {}
            for key, value in metrics.items():
                if key.startswith("verifier/failures/") or key.startswith("verifier/rollouts/"):
                    count_totals[key] += int(value)
                else:
                    runtime_values[key].append(float(value))
    aggregated: dict[str, float | int] = {}
    for key, values in runtime_values.items():
        if values:
            mean_value = sum(values) / len(values)
            aggregated[f"{key}_mean"] = mean_value
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
    aggregated.update(count_totals)

    total_rollouts = sum(len(attempt) for attempt in rollout_results)
    if total_rollouts:
        normalized_keys = [
            key
            for key in list(aggregated.keys())
            if key.startswith("verifier/failures/") or key.startswith("verifier/rollouts/")
        ]
        for count_key in normalized_keys:
            frac_key = f"{count_key}_frac"
            aggregated[frac_key] = aggregated[count_key] / total_rollouts
            del aggregated[count_key]

    return aggregated


def _log_group_verifier_metrics(metrics: dict[str, float | int], stats_writer: StreamWriter):
    if not metrics or getattr(wandb, "run", None) is None:
        return
    new_metrics = {}
    for k, v in metrics.items():
        new_metrics[f"rc_actor/{k}"] = v
    wandb.log(new_metrics)
    stats_writer.write(new_metrics)



class InferenceProblemState:
    """A helper class to track the inference progress for a single problem."""
    
    def __init__(
        self,
        problem_text: str,
        answer: str,
        dataset_name: str,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        problem_id: int,
        sample_id: int,
        starting_step: int,
        schema: str = None,
        use_think_tags: bool = False,
        model_class: str = "qwen",
        reasoning_prompt_style: str = "structured",
        summarization_style: str = "summ",
    ):
        self.problem_text = problem_text
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.problem_id = problem_id
        self.sample_id = sample_id
        self.starting_step = starting_step
        self.answer = answer
        self.dataset_name = dataset_name
        self.curr_summary = ""
        self.curr_reasoning = ""
        self.final_reward = None
        self.use_think_tags = use_think_tags
        self.model_class = model_class
        self.reasoning_prompt_style = reasoning_prompt_style
        self.summarization_style = summarization_style
        self.schema = schema
        self.reasoning_rollout_store = []
        self.summarization_rollout_store = []
        self.reasoning_string_store = []
        self.summarization_string_store = []
        self.reasoning_string_complete_store = []
        self.summarization_string_complete_store = []
        
        # Track turn numbers for metadata
        self.reasoning_turn_number = 0
        self.summarization_turn_number = 0
        self.overall_cycle_step = 0

    def update_reasoning(self, rollout: RolloutResult, response_string: str, model_version: int, rollout_index: int, group_id: str):
        # Increment reasoning turn counter
        self.reasoning_turn_number += 1
        
        # Add metadata to rollout result immediately
        rollout.model_version = model_version
        rollout.group_id = group_id
        
        for sample in rollout.training_texts:
            sample.metadata["model_version"] = model_version
            sample.metadata["rollout_index"] = rollout_index
            sample.metadata["cycle_step"] = self.overall_cycle_step
            sample.metadata["turn_type"] = "reasoning"
            sample.metadata["turn_number"] = self.reasoning_turn_number
            sample.metadata["problem_id"] = self.problem_id
            sample.metadata["sample_id"] = self.sample_id
            sample.metadata["answer"] = self.answer
            sample.metadata["dataset_name"] = self.dataset_name
            sample.metadata["schema"] = self.schema
            sample.metadata["original_problem"] = f"Generate a rigorous proof to the following question:\n\n{self.problem_text}"
            sample.group_id = group_id
        
        # Increment overall cycle step
        self.overall_cycle_step += 1
        
        # Store rollout and process response
        self.reasoning_rollout_store.append(rollout)
        self.reasoning_string_complete_store.append(response_string)
        processed_response_string = response_string.replace("<think>", "")
        if "</think>" in processed_response_string:
            processed_response_string = processed_response_string.split("</think>")[0]
        
        processed_response_string = strip_chat_template_tokens(processed_response_string)
        
        self.curr_reasoning = processed_response_string.strip()
        self.reasoning_string_store.append(self.curr_reasoning)
        
        logger.info(f"[R UPDATE] problem_id={self.problem_id}, sample_id={self.sample_id}, reward={rollout.metrics.reward}, "
                   f"turn={self.reasoning_turn_number}, "
                   f"n_tok={rollout.training_texts[0].output_tokens if rollout.training_texts else 0}")

    def update_summarization(self, rollout: RolloutResult, response_string: str, model_version: int, rollout_index: int, group_id: str):
        # Increment summarization turn counter
        self.summarization_turn_number += 1
        
        # Add metadata to rollout result immediately
        rollout.model_version = model_version
        rollout.group_id = group_id
        
        for sample in rollout.training_texts:
            sample.metadata["model_version"] = model_version
            sample.metadata["rollout_index"] = rollout_index
            sample.metadata["cycle_step"] = self.overall_cycle_step
            sample.metadata["turn_type"] = "summarization"
            sample.metadata["turn_number"] = self.summarization_turn_number
            sample.metadata["problem_id"] = self.problem_id
            sample.metadata["sample_id"] = self.sample_id
            sample.metadata["answer"] = self.answer
            sample.metadata["dataset_name"] = self.dataset_name
            sample.metadata["schema"] = self.schema
            sample.metadata["original_problem"] = f"Generate a rigorous proof to the following question:\n\n{self.problem_text}"
            sample.group_id = group_id
        
        # Increment overall cycle step
        self.overall_cycle_step += 1
        
        # Store rollout and process response
        self.summarization_rollout_store.append(rollout)
        self.summarization_string_complete_store.append(response_string)
        # Process response - handle both thinking and non-thinking models
        if "<think>" in response_string:
            processed_response_string = response_string.replace("<think>", "").replace("</think>", "").strip()
        else:
            processed_response_string = response_string.strip()

        processed_response_string = strip_chat_template_tokens(processed_response_string)
        
        # Update summary based on summarization style
        if self.summarization_style == "summ":
            self.curr_summary = processed_response_string.strip()
        else:  # sequential style
            self.curr_summary = f"{self.curr_summary}\n\n{processed_response_string.strip()}"
        self.summarization_string_store.append(self.curr_summary)
        
        # logger.info(f"[SUMMARIZATION UPDATE] problem_id={self.problem_id}, "
        #            f"turn={self.summarization_turn_number}, "
        #            f"summary_len={len(self.curr_summary)} chars, "
        #            f"output_tokens={rollout.training_texts[0].output_tokens if rollout.training_texts else 0}")

    def get_filled_reasoning_prompt(self, tokenizer=None) -> str:
        """
        Get the filled reasoning prompt. If using 'structured' style and a tokenizer is provided,
        applies chat template. Otherwise returns the filled template as-is.
        """
        if self.reasoning_prompt_style == "completion":
            # For completion style, just return the problem text with current summary
            prompt = self.reasoning_prompt_template.format(
                problem=self.problem_text,
                curr_summary=self.curr_summary,
            )
            # Add think tags if needed for completion style
            if self.use_think_tags and self.model_class != "gptoss" and "<think>" not in prompt:
                if self.curr_summary:
                    prompt = f"{prompt}\n\n{self.curr_summary}\n\n<think>"
                else:
                    prompt = f"{prompt}<think>"
            elif self.curr_summary:
                prompt = f"{prompt}\n\n{self.curr_summary}"
            return prompt
        else:
            # For structured style, format the template
            filled_prompt = self.reasoning_prompt_template.format(
                problem=self.problem_text,
                curr_summary=self.curr_summary,
            )
            
            # Apply chat template if tokenizer is provided
            if tokenizer is not None:
                if self.model_class == "gptoss":
                    templated_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": filled_prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        reasoning_effort="high",
                    )
                else:
                    templated_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": filled_prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=self.use_think_tags,
                    )
                
                # Add think tags if needed
                if self.use_think_tags and self.model_class != "gptoss" and "<think>" not in templated_prompt:
                    return f"{templated_prompt}<think>"
                return templated_prompt
            else:
                # No tokenizer, just return the filled prompt
                return filled_prompt

    def get_filled_summarization_prompt(self, tokenizer=None) -> str:
        """
        Get the filled summarization prompt. If tokenizer is provided, applies chat template.
        """
        # Extract current reasoning chunk
        if "<think>" in self.curr_reasoning:
            curr_chunk = self.curr_reasoning.split("<think>")[1]
            if "</think>" in curr_chunk:
                curr_chunk = curr_chunk.split("</think>")[0]
        else:
            curr_chunk = self.curr_reasoning
        
        filled_prompt = self.summarization_prompt_template.format(
            problem=self.problem_text,
            existing_summary=self.curr_summary, 
            reasoning=curr_chunk.strip()
        )
        
        # Apply chat template if tokenizer is provided
        if tokenizer is not None:
            if self.model_class == "gptoss":
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": filled_prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    reasoning_effort="medium",  # Use medium for summarization
                )
            else:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": filled_prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,  # Summarization doesn't use thinking tags
                )
        else:
            return filled_prompt

    def reset_stores(self):
        self.reasoning_rollout_store = []
        self.summarization_rollout_store = []
        self.reasoning_string_store = []
        self.summarization_string_store = []
    
    def __repr__(self) -> str:
        return f"InferenceProblemState(problem_id={self.problem_id}, problem_text={self.problem_text}, \
            sample_id={self.sample_id}, starting_step={self.starting_step}, answer={self.answer}, dataset_name={self.dataset_name}, \
            reasoning_turn={self.reasoning_turn_number}, summarization_turn={self.summarization_turn_number}, cycle_step={self.overall_cycle_step})"


def generate_dummy_training_text_from_prompt_and_old_training_text(prompt: str, old_training_text: TrainingText) -> TrainingText:
    new_training_text = copy.deepcopy(old_training_text)
    new_training_text.text = prompt + new_training_text.output_text
    new_training_text.metadata["turn_number"] += 1
    new_training_text.metadata["cycle_step"] += 2
    return new_training_text

class SlidingWindowData(BaseModel):
    prompt_tokens_window: list[list[int]] = Field(
        default_factory=list,
        description="Prompt token counts for each chunk in the window",
    )
    output_tokens_window: list[list[int]] = Field(
        default_factory=list,
        description="Output token counts for each chunk in the window",
    )
    timestamps: list[float] = Field(default_factory=list)


class SlidingWindowAggregator:
    def __init__(self, window_size: int):
        self.window_size = window_size
        # Maintain separate sliding windows for each turn type
        self.data_by_turn_type = defaultdict(SlidingWindowData)

    def update(self, prompt_tokens: list[int], output_tokens: list[int], turn_type: str = "unknown"):
        """Update statistics for a specific turn type."""
        data = self.data_by_turn_type[turn_type]
        data.prompt_tokens_window.append(prompt_tokens)
        data.output_tokens_window.append(output_tokens)
        data.timestamps.append(time.time())
        if len(data.prompt_tokens_window) > self.window_size:
            data.prompt_tokens_window.pop(0)
            data.output_tokens_window.pop(0)
            data.timestamps.pop(0)

    def get_stats(self, turn_type: str | None = None):
        """
        Get statistics for a specific turn type, or aggregated across all turn types.
        
        Args:
            turn_type: If specified, return stats for that turn type only.
                      If None, return aggregated stats across all turn types.
        """
        if turn_type is not None:
            # Return stats for specific turn type
            return self._compute_stats_for_data(self.data_by_turn_type[turn_type])
        else:
            # Return aggregated stats across all turn types
            return self._compute_aggregated_stats()
    
    def get_all_turn_type_stats(self) -> dict[str, dict]:
        """Get separate statistics for each turn type."""
        result = {}
        for turn_type, data in self.data_by_turn_type.items():
            stats = self._compute_stats_for_data(data)
            if stats is not None:
                result[turn_type] = stats
        return result

    def _compute_stats_for_data(self, data: SlidingWindowData):
        """Compute statistics for a single SlidingWindowData object."""
        if len(data.prompt_tokens_window) < self.window_size:
            return None

        null_stats = {
            "samples_per_second": 0,
            "output_tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "total_tokens_per_second": 0,
        }
        if not data.timestamps:
            return null_stats

        time_span = data.timestamps[-1] - data.timestamps[0]
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in data.prompt_tokens_window)
        total_output_tokens = sum(sum(tokens) for tokens in data.output_tokens_window)
        total_prompt_tokens = sum(sum(tokens) for tokens in data.prompt_tokens_window)

        return {
            "samples_per_second": num_samples / time_span,
            "output_tokens_per_second": total_output_tokens / time_span,
            "prompt_tokens_per_second": total_prompt_tokens / time_span,
            "total_tokens_per_second": (total_output_tokens + total_prompt_tokens) / time_span,
        }
    
    def _compute_aggregated_stats(self):
        """Compute aggregated statistics across all turn types."""
        # Collect all data points across turn types
        all_prompt_tokens = []
        all_output_tokens = []
        all_timestamps = []
        
        for data in self.data_by_turn_type.values():
            all_prompt_tokens.extend(data.prompt_tokens_window)
            all_output_tokens.extend(data.output_tokens_window)
            all_timestamps.extend(data.timestamps)
        
        if not all_timestamps:
            return None
        
        # Check if we have enough data
        total_samples = sum(len(tokens) for tokens in all_prompt_tokens)
        if total_samples < self.window_size:
            return None
        
        null_stats = {
            "samples_per_second": 0,
            "output_tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "total_tokens_per_second": 0,
        }
        
        time_span = max(all_timestamps) - min(all_timestamps)
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in all_prompt_tokens)
        total_output_tokens = sum(sum(tokens) for tokens in all_output_tokens)
        total_prompt_tokens = sum(sum(tokens) for tokens in all_prompt_tokens)

        return {
            "samples_per_second": num_samples / time_span,
            "output_tokens_per_second": total_output_tokens / time_span,
            "prompt_tokens_per_second": total_prompt_tokens / time_span,
            "total_tokens_per_second": (total_output_tokens + total_prompt_tokens) / time_span,
        }




def make_stats_dict() -> dict:
    return defaultdict(lambda: defaultdict(list))


async def schedule_rollouts(
    cfg: DictConfig,
    attempts: int,
    problem_queue: SharedMemoryQueue,
    result_queue: SharedMemoryQueue,
    trainer_state: TrainerState,
    llms: list[TrainableLLM],
    summarization_llms: list[TrainableLLM] | None,
    scheduler_name: str,
):
    """This routine schedules rollouts for a given problem queue and result queue.

    For online RC rollouts:
    - Takes InferenceProblemState from the problem queue
    - Does multiple reasoning/summarization cycles
    - Each cycle: generate reasoning, then summarize it
    - Collects all rollout results across cycles
    - Puts completed rollouts in result queue
    
    Key differences from standard rollouts:
    - Stateful: maintains InferenceProblemState across cycles
    - Iterative: multiple reasoning->summarization steps
    - Single attempt: each problem generates one trajectory
    
    Args:
        summarization_llms: Optional separate LLMs for summarization. If None, uses the same LLMs as solution generation.
    """
    loop = asyncio.get_running_loop()

    # Use separate summarization LLMs if provided, otherwise use the same LLMs
    actual_summarization_llms = summarization_llms if summarization_llms is not None else llms
    
    # Track active tasks per LLM
    active_rollouts = [0] * len(llms)
    active_summarization_rollouts = [0] * len(actual_summarization_llms)
    started_solution_rollouts = [0] * len(llms)
    started_summarization_rollouts = [0] * len(actual_summarization_llms)
    finished_solution_rollouts = [0] * len(llms)
    finished_summarization_rollouts = [0] * len(actual_summarization_llms)
    
    # Track rollouts per problem group with separate tracking for generation and summarization
    group_rollouts = {}  # Maps group_id -> list of RolloutResults (final ordered results)
    group_attempts_completed = {}  # Maps group_id -> number of attempts completed
    
    # Separate tracking for each turn's generation and summarization results
    # group_turn_results[group_id][turn_idx] = {"generation": RolloutResult, "summarization": RolloutResult}
    group_turn_results = {}  # Maps group_id -> dict[turn_idx -> dict with "generation" and "summarization"]
    
    solution_rollout_policy = hydra.utils.get_method(cfg.rc_actor.solution_rollout_policy)
    summarization_rollout_policy = hydra.utils.get_method(cfg.rc_actor.summarization_rollout_policy)
    logger.info(f"Use solution rollout policy: {solution_rollout_policy}")
    logger.info(f"Use summarization rollout policy: {summarization_rollout_policy}")

    max_retries = cfg.rc_actor.get("max_retries", 3)
    retry_base_delay = cfg.rc_actor.get("retry_base_delay", 1.0)

    
    # Queue for rollouts that failed with transient errors and need to be retried
    retry_queue: asyncio.Queue = asyncio.Queue()

    async def rollout_and_maybe_produce_result(
        problem_state: InferenceProblemState,
        group_id: int,
        rollout_index: int,
        llm_index: int,
        summarization_llm_index: int,
        session: aiohttp.ClientSession,
        requeue_count: int = 0,
    ):
        nonlocal started_solution_rollouts, started_summarization_rollouts, finished_solution_rollouts, finished_summarization_rollouts
        try:
            llm = llms[llm_index]
            summarization_llm = actual_summarization_llms[summarization_llm_index]
            model_version = trainer_state.propagated_weight_version
            assert model_version is not None

            # Create full group ID
            full_group_id = f"{scheduler_name}_{group_id}"

            # Get number of reasoning steps from config
            num_reasoning_steps = cfg.rc_actor.get("num_reasoning_steps", 3)
            
            # Online RC workflow: multiple reasoning/summarization cycles
            all_rollout_results = []
            
            for step_idx in range(num_reasoning_steps):
                # Retry loop for transient errors
                last_error = None
                for attempt in range(max_retries):
                    try:
                        # 1. Reasoning step: generate reasoning based on current summary
                        # Create a problem dict with the current state
                        reasoning_problem = {
                            "task": problem_state.get_filled_reasoning_prompt(),
                            "answer": problem_state.answer,
                            "dataset": problem_state.dataset_name,
                            "id": problem_state.problem_id,
                            "schema": problem_state.schema,
                            "original_problem": problem_state.problem_text,
                        }
                        # logger.info(f"Reasoning problem: {reasoning_problem}")
                        
                        started_solution_rollouts[llm_index] += 1
                        solution_rollout_result = await solution_rollout_policy(cfg, llm, reasoning_problem, session)
                        
                        # Extract the reasoning text from the rollout result
                        try:
                            reasoning_text = solution_rollout_result.training_texts[0].output_text
                        except Exception as e:
                            logger.error(f"Error in solution rollout step {step_idx} (attempt {attempt + 1}/{max_retries}), extracting reasoning text: {e}")
                            reasoning_text = ""

                        # Update reasoning with metadata
                        problem_state.update_reasoning(
                            solution_rollout_result, 
                            reasoning_text,
                            model_version,
                            rollout_index,
                            full_group_id
                        )
                    
                        # 2. Summarization step: summarize the reasoning (use summarization_llm)
                        summarization_problem = {
                            "task": problem_state.get_filled_summarization_prompt(),
                            "answer": problem_state.answer,
                            "dataset": problem_state.dataset_name,
                            "id": problem_state.problem_id,
                            "original_problem": problem_state.problem_text,
                        }
                        # logger.info(f"Summarization problem: {summarization_problem}")
                        
                        started_summarization_rollouts[summarization_llm_index] += 1
                        summarization_rollout_result = await summarization_rollout_policy(cfg, summarization_llm, summarization_problem, session)
                        
                        # Extract and update the summary
                        try:
                            summary_text = summarization_rollout_result.training_texts[0].output_text
                        except AttributeError as e:
                            logger.error(f"Error extracting summary text: {e}")
                            summary_text = ""
                        # Update summarization with metadata
                        problem_state.update_summarization(
                            summarization_rollout_result, 
                            summary_text,
                            model_version,
                            rollout_index,
                            full_group_id
                        )
                        
                        # Store both rollout results
                        all_rollout_results.append(solution_rollout_result)
                        all_rollout_results.append(summarization_rollout_result)

                        if len(all_rollout_results) == 2 * cfg.rc_actor.num_reasoning_steps:
                            # completed all reasoning and summarization turns
                            # now add a dummy rollout result for the last turn
                            # this is to ensure that the state at the end of the last turn is also considered 
                            # for sampling rollouts in the next stage by the actor.
                            last_but_one_reasoning_rollout_result = all_rollout_results[-2]
                            final_dummy_rollout_result = copy.deepcopy(last_but_one_reasoning_rollout_result)
                            final_dummy_rollout_result.training_texts[0] = generate_dummy_training_text_from_prompt_and_old_training_text(
                                problem_state.get_filled_reasoning_prompt(tokenizer=llm.tokenizer), last_but_one_reasoning_rollout_result.training_texts[0])
                            all_rollout_results.append(final_dummy_rollout_result)

                        finished_solution_rollouts[llm_index] += 1
                        finished_summarization_rollouts[summarization_llm_index] += 1
                        break
                        
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            delay = retry_base_delay * (2 ** attempt)
                            logger.warning(
                                f"Error in rollout step {step_idx} (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {delay:.1f}s: {type(e).__name__}: {e}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                f"Error in rollout step {step_idx} after {max_retries} attempts, giving up: "
                                f"{type(e).__name__}: {e}"
                            )
                            raise
                else:
                    # This shouldn't happen, but just in case
                    raise last_error
            
            # After all reasoning steps, package all rollout results
            # Each reasoning and summarization step is kept as a separate RolloutResult
            # Metadata was already set in update_reasoning and update_summarization
            if all_rollout_results:
                # Add all rollouts from this attempt to the group
                group_rollouts[group_id].append(all_rollout_results)
                group_attempts_completed[group_id] += 1
                
                # Check if we've completed all attempts for this group
                # For online RC with attempts=1, this means we've done 1 problem trajectory
                if group_attempts_completed[group_id] == attempts:
                    # All attempts complete - put all rollouts in result queue
                    random.shuffle(group_rollouts[group_id])
                    result_queue.put(group_rollouts[group_id])
                    del group_rollouts[group_id]
                    del group_attempts_completed[group_id]
        except TRANSIENT_EXCEPTIONS as e:
            # Transient errors (HTTP/connection issues) that exhausted retries.
            # Re-queue the rollout to try again later, up to MAX_REQUEUE_ATTEMPTS times.
            if requeue_count < MAX_REQUEUE_ATTEMPTS:
                logger.warning(
                    f"Transient error in rollout for group {group_id}, re-queuing "
                    f"(attempt {requeue_count + 1}/{MAX_REQUEUE_ATTEMPTS}): {type(e).__name__}: {e}"
                )
                await retry_queue.put((problem_state, group_id, rollout_index, requeue_count + 1))
            else:
                # Exhausted all re-queue attempts - this is a fatal error for the group
                logger.error(
                    f"Transient error in rollout for group {group_id} after {MAX_REQUEUE_ATTEMPTS} "
                    f"re-queue attempts, stopping actor: {type(e).__name__}: {e}"
                )
                current_task = asyncio.current_task(loop=loop)
                for task in asyncio.all_tasks(loop=loop):
                    if task != current_task:
                        task.cancel()
                result_queue.put(e)
                logger.error("Stopped all tasks and put exception in the result queue")
        except Exception as e:
            # Fatal error - cancel all tasks and stop the actor
            logger.error("Fatal exception in rollout, stop all other rollout tasks", exc_info=e)
            current_task = asyncio.current_task(loop=loop)
            for task in asyncio.all_tasks(loop=loop):
                if task != current_task:
                    task.cancel()
            result_queue.put(e)
            logger.error("Stopped all tasks and put exception in the result queue")
        finally:
            active_rollouts[llm_index] -= 1
            active_summarization_rollouts[summarization_llm_index] -= 1

    group_id = -1
    group_rollout_index = attempts
    problem_state = None

    last_logged = time.time()
    logger.info("Starting rollout scheduler")
    connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
    timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while True:
            if time.time() - last_logged > 10.0 and sum(active_rollouts):
                retry_queue_size = retry_queue.qsize()
                total_started_solution = sum(started_solution_rollouts)
                total_started_summarization = sum(started_summarization_rollouts)
                total_finished_solution = sum(finished_solution_rollouts)
                total_finished_summarization = sum(finished_summarization_rollouts)
                logger.info(
                    f"{scheduler_name}: "
                    f"rollouts in progress: {sum(active_rollouts)}, "
                    f"groups in progress: {len(group_rollouts)}, "
                    f"solution rollouts: {total_started_solution} started / {total_finished_solution} finished, "
                    f"summarization rollouts: {total_started_summarization} started / {total_finished_summarization} finished, "
                    f"max group size in bytes: {result_queue.max_actual_entry_size()}, "
                    + (f"retry queue size: {retry_queue_size}" if retry_queue_size > 0 else "")
                )
                last_logged = time.time()

            # First, check if there are any failed rollouts to retry
            retry_item = None
            try:
                retry_item = retry_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if retry_item is not None:
                # Re-queue a failed rollout
                retry_problem_state, retry_group_id, retry_rollout_index, requeue_count = retry_item
                next_llm = active_rollouts.index(min(active_rollouts))
                next_summarization_llm = active_summarization_rollouts.index(min(active_summarization_rollouts))
                if active_rollouts[next_llm] == cfg.rc_actor.llm_max_rollouts or active_summarization_rollouts[next_summarization_llm] == cfg.rc_actor.summarization_max_rollouts:
                    # All LLMs are busy, put item back and wait
                    await retry_queue.put(retry_item)
                    await asyncio.sleep(0.01)
                    continue
                active_rollouts[next_llm] += 1
                active_summarization_rollouts[next_summarization_llm] += 1
                loop.create_task(
                    rollout_and_maybe_produce_result(
                        problem_state=retry_problem_state,
                        group_id=retry_group_id,
                        rollout_index=retry_rollout_index,
                        llm_index=next_llm,
                        summarization_llm_index=next_summarization_llm,
                        session=session,
                        requeue_count=requeue_count,
                    )
                )
                logger.info(f"{scheduler_name}: Retrying rollout problem id {retry_problem_state.problem_id} with group id {retry_group_id} and actor llm {next_llm} and summarization llm {next_summarization_llm}")
                continue

            # Then, check if we need to start a new group
            if group_rollout_index == attempts:
                try:
                    problem_state = problem_queue.get(block=False)
                    logger.info(f"{scheduler_name}: Got a new problem {problem_state.problem_id}")
                    init_problem_state_copy = copy.deepcopy(problem_state)
                except Empty:
                    # give some quality time for other couroutines to work
                    await asyncio.sleep(0.01)
                    continue
                group_id += 1
                group_rollouts[group_id] = []
                group_attempts_completed[group_id] = 0
                group_rollout_index = 0
                
            next_llm = active_rollouts.index(min(active_rollouts))
            next_summarization_llm = active_summarization_rollouts.index(min(active_summarization_rollouts))
            if active_rollouts[next_llm] == cfg.rc_actor.llm_max_rollouts or active_summarization_rollouts[next_summarization_llm] == cfg.rc_actor.summarization_max_rollouts:
                # all llms are busy, wait for one to finish
                # logger.info(f"{scheduler_name}: All LLMs are busy, waiting for one to finish. Current active rollouts {active_rollouts}. Current active summarization rollouts {active_summarization_rollouts}.")
                await asyncio.sleep(1.0)
                continue
            active_rollouts[next_llm] += 1
            active_summarization_rollouts[next_summarization_llm] += 1
            logger.info(f"{scheduler_name}: Started a new rollout for problem id {problem_state.problem_id} with group id {group_id} and actor llm {next_llm} and summarization llm {next_summarization_llm}")
            assert problem_state is not None
            problem_state = copy.deepcopy(init_problem_state_copy)
            problem_state.sample_id = group_rollout_index
            loop.create_task(
                rollout_and_maybe_produce_result(
                    problem_state=problem_state,
                    group_id=group_id,
                    rollout_index=group_rollout_index,
                    llm_index=next_llm,
                    summarization_llm_index=next_summarization_llm,
                    session=session,
                )
            )
            group_rollout_index += 1
            


def rollout_maker_entrypoint(
    cfg: DictConfig,
    attempts: int,
    problem_queue: SharedMemoryQueue,
    result_queue: SharedMemoryQueue,
    llms: list[TrainableLLM],
    summarization_llms: list[TrainableLLM] | None,
    scheduler_name: str,
):
    trainer_state = TrainerState(Path(cfg.output_dir))
    eval_only_mode = cfg.get('eval_only', False)
    
    if cfg.debug.mode or eval_only_mode:
        # In debug or eval-only mode, don't listen for weight updates
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        schedule_rollouts(cfg, attempts, problem_queue, result_queue, trainer_state, llms, summarization_llms, scheduler_name)
    )
    loop.close()
    logger.info("Rollout maker loop closed")


def random_iter(problems: list):
    while True:
        yield random.sample(problems, 1)[0]


def sequential_iter(problems: list):
    for problem in problems:
        yield problem


class RCActorLoop:
    def __init__(
        self,
        cfg: DictConfig,
        llms: list[TrainableLLM],
        summarization_llms: list[TrainableLLM] | None,
        data_stream: StreamSpec,
        stats_stream: StreamSpec,
        trainer_state: TrainerState,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        tokenizer=None,
        is_training: bool = True,
        use_think_tags: bool = False,
        model_class: str = "qwen",
        reasoning_prompt_style: str = "structured",
        summarization_style: str = "summ",
    ) -> None:
        self.data_stream = data_stream
        self.trainer_state = trainer_state
        self.stats_stream = stats_stream
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.tokenizer = tokenizer
        self.use_think_tags = use_think_tags
        self.model_class = model_class
        self.reasoning_prompt_style = reasoning_prompt_style
        self.summarization_style = summarization_style
        self.sliding_aggregator = SlidingWindowAggregator(window_size=cfg.rc_actor.throughput_window_size)
        self.llms = llms
        self.summarization_llms = summarization_llms if summarization_llms is not None else llms
        self.loop_start_time = -1
        self.cfg = cfg
        self.is_training = is_training
        self.is_scheduling_paused = False
        self.debug_mode = bool(cfg.debug.mode)
        self.verifier_metrics_step = 0
        self._last_verifier_timestep: float | None = None
        llm_grader_cfg = cfg.get("llm_grader", None)
        wandb_table_cfg = llm_grader_cfg.get("wandb_table", None) if llm_grader_cfg is not None else None
        self.wandb_table_enabled = True
        keep_last_k_groups = 32
        log_every_n_groups = 32
        if wandb_table_cfg is not None:
            self.wandb_table_enabled = wandb_table_cfg.get("enabled", True)
            keep_last_k_groups = wandb_table_cfg.get("keep_last_k_groups", 32)
            log_every_n_groups = wandb_table_cfg.get("log_every_n_groups", 32)
        self.verifier_table_buffer = VerifierTableBuffer(
            keep_last_k_groups=keep_last_k_groups,
            log_every_n_groups=log_every_n_groups,
        )
        
        # Online rollout configuration
        self.num_reasoning_steps = cfg.rc_actor.get("num_reasoning_steps", 3)
        
        # Determine the number of processes to use
        num_processes = min(self.cfg.rc_actor.rollout_workers, len(self.llms))
        attempts = 1 if self.is_training else cfg.rc_actor.attempts # for online RC, always use 1 attempt
        logger.info(f"Using {attempts} attempts for {'train' if self.is_training else 'test'} RC")

        # Divide LLMs approximately equally across processes
        llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            llm_groups[i % num_processes].append((i, llm))
        
        # Divide summarization LLMs across processes (same pattern)
        summarization_llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.summarization_llms):
            summarization_llm_groups[i % num_processes].append((i, llm))

        self.smm = SharedMemoryManager()
        self.smm.start()

        
        # Use SharedMemoryQueue instead of separate problem_queue, result_queue, and io_buffer
        self.problem_queue = SharedMemoryQueue(self.smm, self.cfg.rc_actor.problem_queue_size, cfg.rc_actor.shared_memory_entry_size)
        self.result_queue = SharedMemoryQueue(self.smm, self.cfg.rc_actor.result_queue_size, cfg.rc_actor.shared_memory_entry_size)
        
        logger.info(f"Initialized {'train' if self.is_training else 'test'} RC actor loop")
        logger.info(f"Problem queue size: {self.problem_queue.max_size}, result queue size: {self.result_queue.max_size}")
        logger.info(f"Result queue buffer size: {self.result_queue.get_memory_size() / 2**30} Gb")

        # Create and start multiple rollout processes
        self.rollout_processes = []
        for llm_group, summarization_llm_group in zip(llm_groups, summarization_llm_groups):
            assert llm_group
            llm_idxs = [llm[0] for llm in llm_group]
            llms = [llm[1] for llm in llm_group]
            summarization_llms_for_process = [llm[1] for llm in summarization_llm_group]
            scheduler_name = (
                f"{'train' if self.is_training else 'test'} RC scheduler for llms {','.join([str(i) for i in llm_idxs])}"
            )
            process = mp.Process(
                target=rollout_maker_entrypoint,
                args=(self.cfg, attempts, self.problem_queue, self.result_queue, llms, summarization_llms_for_process, scheduler_name),
            )
            process.start()
            self.rollout_processes.append(process)

    def init_stats(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.latency_list = []
        self.model_versions_list = []
        self.sliding_stats = defaultdict(list)
    
    def compute_domain_agnostic_metrics(self, result: RolloutResult) -> dict[str, float]:
        metrics = {}
        
        metrics['overflow'] = all([not training_text.finished for training_text in result.training_texts ])
        metrics['num_turns'] = len(result.training_texts)
        metrics['prompt_tokens'] = [training_text.prompt_tokens for training_text in result.training_texts]
        metrics['output_tokens'] = [training_text.output_tokens for training_text in result.training_texts]
        
        return metrics

    def update_stats(self, rollout_results: list[list[RolloutResult]]):
        for attempt in rollout_results:
            for result in attempt:
                assert result.model_version is not None
                assert isinstance(result.metrics, BaseMetrics), "Metrics should be an instance of BaseMetrics"
                dataset_name = result.dataset_name
                group_id = result.group_id
                
                # Get turn_type and turn_number from the first training text's metadata
                # All training texts in a result should have the same turn_type/turn_number
                if result.training_texts:
                    turn_type = result.training_texts[0].metadata.get("turn_type", "unknown")
                    turn_number = result.training_texts[0].metadata.get("turn_number", 0)
                else:
                    turn_type = "unknown"
                    turn_number = 0
                metric_prefix = f"{turn_type}_turn_{turn_number}"
                
                self.latency_list.append(result.latency)
                self.model_versions_list.append(result.model_version)
                domain_agnostic_metrics = self.compute_domain_agnostic_metrics(result) 
                all_metrics = result.metrics.model_dump() | domain_agnostic_metrics
                for k, v in all_metrics.items():
                    # Add turn type and number prefix to all metric keys
                    metric_key = f"{metric_prefix}/{k}"
                    if isinstance(v, list):
                        self.stats[metric_key][dataset_name][group_id] += v
                    elif isinstance(v, float) | isinstance(v, bool) | isinstance(v, int):
                        self.stats[metric_key][dataset_name][group_id].append(v)
                    else:
                        raise ValueError(f"Unsupported metric type: {type(v)} for key {k}")
        
        # Update sliding window stats separately for each turn type
        # Group results by turn type
        results_by_turn_type = defaultdict(list)
        for attempt in rollout_results:
            for result in attempt:
                if result.training_texts:
                    turn_type = result.training_texts[0].metadata.get("turn_type", "unknown")
                else:
                    turn_type = "unknown"
                results_by_turn_type[turn_type].append(result)
        
        # Update aggregator for each turn type
        for turn_type, results in results_by_turn_type.items():
            prompt_length_tokens = [training_text.prompt_tokens for result in results for training_text in result.training_texts]
            output_length_tokens = [training_text.output_tokens for result in results for training_text in result.training_texts]
            if prompt_length_tokens:  # Only update if we have data
                self.sliding_aggregator.update(prompt_length_tokens, output_length_tokens, turn_type=turn_type)
        
        # Get stats for each turn type and log them
        turn_type_stats = self.sliding_aggregator.get_all_turn_type_stats()
        for turn_type, stats in turn_type_stats.items():
            for k, v in stats.items():
                metric_key = f"{turn_type}/{k}"
                self.sliding_stats[metric_key].append(v)
        
        # Also get aggregated stats across all turn types
        aggregated_stats = self.sliding_aggregator.get_stats()
        if aggregated_stats is not None:
            for k, v in aggregated_stats.items():
                self.sliding_stats[k].append(v)
        
    def _measure_verifier_group_runtime(self) -> float | None:
        """
        Track wall-clock seconds required to finish scoring a group of rollouts.
        """
        now = time.perf_counter()
        last = self._last_verifier_timestep
        self._last_verifier_timestep = now
        if last is None:
            return None
        return now - last

    def log_verifier_metrics_for_group(self, rollout_results: list[list[RolloutResult]]) -> None:
        if (
            not self.is_training
            or not self.cfg.wandb.use_wandb
            or not rollout_results
        ):
            return
        aggregated = _aggregate_group_verifier_metrics(rollout_results)
        sec_per_step = self._measure_verifier_group_runtime()
        if sec_per_step is not None:
            aggregated["verifier/runtime/sec_per_step"] = sec_per_step
        if not aggregated:
            return
        aggregated["verifier/group_size"] = len(rollout_results)
        success_frac = aggregated.get("verifier/rollouts/success_frac")
        if success_frac is not None:
            aggregated["verifier/group_size_eff"] = aggregated["verifier/group_size"] * success_frac
        self.verifier_metrics_step += 1
        aggregated["verifier/group_index"] = self.verifier_metrics_step
        with write_to_streams(self.stats_stream, "a") as stats_writer:
            _log_group_verifier_metrics(aggregated, stats_writer)
        return


    def create_rc_rollout_state(self, existing_state: InferenceProblemState, thinking_ind: int, summary_ind: int) -> InferenceProblemState:
        """
        Create a snapshot of the current state of the problem for RC.
        """

        thinking_strings = existing_state.reasoning_string_store
        summarization_strings = existing_state.summarization_string_store

        snapshot = {
            "problem_text": existing_state.problem_text,
            "original_problem": existing_state.problem_text,
            "problem_id": existing_state.problem_id,
            "sample_id": 0,
            "starting_step": existing_state.starting_step,
        }

        snapshot_state = InferenceProblemState(
            **snapshot,
            reasoning_prompt_template=self.reasoning_prompt_template,
            summarization_prompt_template=self.summarization_prompt_template,
            use_think_tags=self.use_think_tags,
            model_class=self.model_class,
            reasoning_prompt_style=self.reasoning_prompt_style,
            summarization_style=self.summarization_style,
        )

        if len(thinking_strings) > 0:
            snapshot_state.curr_reasoning = thinking_strings[thinking_ind - 1] if thinking_ind > 0 else ""
        if len(summarization_strings) > 0:
            snapshot_state.curr_summary = summarization_strings[summary_ind - 1] if summary_ind > 0 else ""
        return snapshot_state


    def init_rc_rollout_state(
        self,
        problem: dict
    ) -> InferenceProblemState:
        """
        Prepare init state for an RC rollout.
        """
        
        logger.info(f"[INIT PROBLEM] problem_id={problem['id']}, "
                   f"problem_text_len={len(problem['task'])} chars")
        
        state = InferenceProblemState(
            problem_text=problem['task'],
            problem_id=problem['id'],
            answer=problem['answer'],
            dataset_name=problem['dataset'],
            sample_id=0,
            reasoning_prompt_template=self.reasoning_prompt_template,
            summarization_prompt_template=self.summarization_prompt_template,
            use_think_tags=self.use_think_tags,
            model_class=self.model_class,
            reasoning_prompt_style=self.reasoning_prompt_style,
            summarization_style=self.summarization_style,
            starting_step=0,
            schema=problem['schema'] if 'schema' in problem else None
        )
        
        return state

    def broadcast_states(
        self, 
        active_states: List[InferenceProblemState], 
        n: int
    ) -> List[InferenceProblemState]:
        """
        Broadcast states to create n samples from each state.
        Similar to the reference implementation from verl-stable.
        """
        import copy
        broadcasted_states = []
        for i, state in enumerate(active_states):
            for j in range(n):
                curr_state = copy.deepcopy(state)
                curr_state.sample_id = j
                broadcasted_states.append(curr_state)
        return broadcasted_states

    def compute_online_rollout_metrics(
        self,
        online_rollout_states: List[InferenceProblemState],
        reward_function: Any,
    ) -> Dict[str, float]:
        """
        Compute metrics for online rollouts.
        Adapted from verl-stable reasoning cache implementation.
        """
        initial_scores = []
        final_scores = []
        problem_ids = [state.problem_id for state in online_rollout_states]
        
        for state in online_rollout_states:
            if len(state.reasoning_string_complete_store) == 0:
                continue
            initial_reasoning_string = state.reasoning_string_complete_store[0]
            final_reasoning_string = state.reasoning_string_complete_store[-1]
            initial_reasoning_score = reward_function(initial_reasoning_string, state.answer)
            final_reasoning_score = reward_function(final_reasoning_string, state.answer)
            initial_scores.append(initial_reasoning_score)
            final_scores.append(final_reasoning_score)

        problem_score_dict = defaultdict(list)
        for p_id, score in zip(problem_ids, final_scores):
            problem_score_dict[p_id].append(score)

        scores_by_problem = list(problem_score_dict.values())
        # Best-of-N: check if any sample succeeded
        bon_by_problem = [1 if any(x == 1 for x in score_list) else 0 for score_list in scores_by_problem]

        metrics = {
            "online_rollout_initial_score_mean": np.mean(initial_scores) if initial_scores else 0.0,
            "online_rollout_final_score_mean": np.mean(final_scores) if final_scores else 0.0,
            "online_rollout_final_score_bon_mean": np.mean(bon_by_problem) if bon_by_problem else 0.0,
        }
        return metrics



    def run(self, dataset: list[tuple[str, dict]], cfg: DictConfig):
        loop_start_time = time.time()
        self.init_stats()

        attempts = 1 # for online RC, always use 1 attempt
        published_samples = 0
        submitted_groups = 0
        finished_groups = 0
        expected_rollouts = -1 if self.is_training else len(dataset)
        if expected_rollouts > 0:
            logger.info(f"Will stop after {expected_rollouts} rollouts")
        trainer_version_to_publish = None

        # If training, we expect to sample infinitely
        # for train sample, sample random batches infinitely
        # for test samples, loop through the dataset once
        if self.is_training:
            problem_iter = random_iter(dataset)
        else:
            problem_iter = sequential_iter(dataset)
        assert self.trainer_state.propagated_weight_version is not None

        last_trainer_version = self.trainer_state.propagated_weight_version
        
        logger.info(f"Start {'train' if self.is_training else 'test'} actor loop")
        with (
            write_to_streams(self.data_stream, "a") as data_stream_writer,
            write_to_streams(self.stats_stream, "a") as stats_writer,
        ):
            while True:
                # the user function must do next(...) to run each iteration
                yield

                if self.trainer_state.propagated_weight_version > last_trainer_version:
                    # the weights have been updated, publish the stats of the previous trainer version
                    trainer_version_to_publish = last_trainer_version
                    last_trainer_version = self.trainer_state.propagated_weight_version

                # First, submit all problems you can until the problem queue is full
                if not self.is_scheduling_paused:
                    while True:
                        if not self.problem_queue.full():
                            try:
                                try:
                                    problem = next(problem_iter)
                                    init_problem_state = self.init_rc_rollout_state(
                                        problem=problem
                                    )
                                    self.problem_queue.put(init_problem_state, block=False)
                                    submitted_groups += 1
                                except queue.Full:            
                                    assert False, "Problem queue was not full just a moment ago, but now it is full"
                            except StopIteration:
                                break
                        else:
                            break

                # Second, try return a result
                try:
                    # Directly get the result from the SharedMemoryQueue
                    rollout_results = self.result_queue.get(block=False)
                except queue.Empty:
                    continue

                if isinstance(rollout_results, Exception):
                    logger.error("Stop actor loop due to error")
                    raise rollout_results

                try:
                    assert isinstance(rollout_results, list), f"rollout_results is not a list: {type(rollout_results)}" # each group is a list of size attempts. Every attempt is a list of rollout results, one for each cycle step.
                    assert isinstance(rollout_results[0], list), f"rollout_results[0] is not a list: {type(rollout_results[0])}" # each attempt is a list of rollout results
                    assert isinstance(rollout_results[0][0], RolloutResult), f"rollout_results[0][0] is not a RolloutResult: {type(rollout_results[0][0])}" # each cycle step is a RolloutResult
                except Exception as e:
                    logger.error(f"rollout_results: {rollout_results}")
                    logger.error(f"Error in rollout_results: {e}")
                    raise e
                
                group_samples = sum(len(attempt) for attempt in rollout_results) # number of cycle steps

                published_samples += group_samples
                samples_in_queue = self.result_queue.qsize() * attempts * cfg.rc_actor.num_reasoning_steps
                all_text_dumps = []
                for attempt in rollout_results:
                    for cycle_step in attempt:
                        for text in cycle_step.training_texts:
                            dump = text.model_dump()
                            # Explicitly include properties that aren't automatically dumped
                            dump['prompt_text'] = text.prompt_text
                            dump['output_text'] = text.output_text
                            all_text_dumps.append(dump)
                data_stream_writer.write(all_text_dumps)
                in_progress = submitted_groups - finished_groups
                logger.info(
                    f"Published {group_samples} {'train' if self.is_training else 'test'} samples"
                    f" to {self.data_stream}, total {published_samples} samples so far, {samples_in_queue} samples in the result queue,"
                    f" {in_progress} groups in progress"
                )

                if self.cfg.wandb.use_wandb and self.wandb_table_enabled:
                    group_index_value = finished_groups + 1
                    group_entries: list[dict[str, str | int]] = []
                    for result in rollout_results:
                        entry = getattr(result, "verifier_table_entry", None)
                        if entry:
                            entry_with_index = dict(entry)
                            entry_with_index["group_index"] = group_index_value
                            group_entries.append(entry_with_index)
                    if group_entries:
                        self.verifier_table_buffer.add_group(group_entries)
                        if self.verifier_table_buffer.should_log():
                            try:
                                self.verifier_table_buffer.log_to_wandb()
                            except Exception as e:
                                logger.error(f"Failed to log verifier table to wandb: {e}")

                
                self.update_stats(rollout_results=rollout_results)
                self.log_verifier_metrics_for_group(rollout_results)

                finished_groups += 1
                time_to_publish_train_stats = (
                    self.is_training
                    and trainer_version_to_publish is not None
                ) or self.debug_mode 
                time_to_publish_test_stats = finished_groups == expected_rollouts

                # Publish stats at every new model version or if all tapes are finished
                if time_to_publish_train_stats or time_to_publish_test_stats:
                    if self.is_training:
                        loop_stats = {
                            "published_samples": published_samples,
                            "problem_queue_size": self.problem_queue.qsize(),
                            "result_queue_size": self.result_queue.qsize(),
                            "finished_groups": finished_groups,
                            "trainer_model_version": trainer_version_to_publish, 
                            "time_since_start": time.time() - loop_start_time,
                        }
                        trainer_version_to_publish = None
                    else:
                        loop_stats = {
                            "trainer_model_version": last_trainer_version
                            }

                    self.publish_stats(
                        stats_writer=stats_writer,
                        loop_stats=loop_stats,
                    )


                if finished_groups == expected_rollouts:
                    logger.info(f"Finished {expected_rollouts} rollouts, stopping actor loop")
                    break

    def publish_stats(self, stats_writer: StreamWriter, loop_stats: dict):
        split_name = "test_" if not self.is_training else ""

        stats = defaultdict(float)
        for metric_name, dict_of_stats_per_metric in self.stats.items():
            for agg, group_stats in calculate_stats(dict_of_stats_per_metric).items():
                stats[f"{split_name}{metric_name}_{agg}"] = group_stats

            for dataset_name, list_of_stats_per_metric_and_dataset in self.stats[metric_name].items():
                for agg, sub_stats in calculate_stats(list_of_stats_per_metric_and_dataset).items():
                    stats[f"{dataset_name}/{metric_name}_{agg}"] = sub_stats

        stats |= (
            {
                f"{split_name}{k}": v
                for k, v in always_or_never_success_stats(self.stats["success"]).items()
            }
            | {
                f"{split_name}latency_" + k: v
                for k, v in calculate_stats(self.latency_list).items()
            }
            | {
                f"{split_name}model_version_" + k: v
                for k, v in calculate_stats(self.model_versions_list).items()
            }
        )

        stats |= loop_stats
        for k, v in self.sliding_stats.items():
            stats[k] = sum(v) / len(v) if v else 0
        if self.cfg.wandb.use_wandb:
            wandb.log({f"rc_actor/{k}": v for k, v in stats.items()})
        stats_writer.write(stats)
        self.init_stats()  # Reset stats for the next iteration


def run_actor_loop(cfg: DictConfig):


    set_streams_backend(**cfg.streams)

    # set seed for reproducibility (mostly intended for dataset loading)
    random.seed(cfg.seed)

    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path / "rc_actor", "rc_actor")
    logger.info(f"Current dir: {os.getcwd()}, experiment root dir: {cfg.output_dir}")
    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "rc_actor", flatten_dict_config(cfg))  # type: ignore
        if run is None:
            raise ValueError("Failed to initialize wandb run")
        wandb.define_metric("verifier/*", step_metric="verifier/group_index")
    llm_urls = str(cfg.me.llm_urls).split("+")
    
    # Check if separate summarization LLMs are configured
    summarization_llm_urls = None
    if hasattr(cfg.me, 'summarization_llm_urls') and cfg.me.summarization_llm_urls:
        summarization_llm_urls = str(cfg.me.summarization_llm_urls).split("+")
        logger.info(f"Using separate summarization LLMs: {summarization_llm_urls}")
    else:
        logger.info("Using the same LLMs for summarization as for solution generation")

    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="rc_stats")
    test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="rc_stats_test")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="rc_actor")
    test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="rc_actor_test")

    dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
    # Get dataset loader parameters if they exist in config, otherwise use empty dict
    dataset_loader_params = cfg.get('dataset_loader_params', {})
    # Use **dataset_loader_params to pass parameters only if they exist
    train_dataset = dataset_loader(cfg.train_dataset_names, **dataset_loader_params)
    test_dataset = dataset_loader(cfg.test_dataset_names, **dataset_loader_params)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin : cfg.train_subset.end]
    logger.info(f"Loaded {len(train_dataset)} training problems")
    logger.info(f"Loaded {len(test_dataset)} test problems")

    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
        actor_model_revision = None
    else:
        actor_model_path = cfg.model_path
        actor_model_revision = cfg.get("model_revision")
        if actor_model_path is None:
            raise ValueError("model_path must be defined")

    # Determine summarization model path
    if cfg.get('summarization_model_path') is not None:
        summarization_model_path = cfg.summarization_model_path
        summarization_model_revision = cfg.get("summarization_model_revision")
        if summarization_model_path is None:
            raise ValueError("summarization_model_path must be defined")
        logger.info(f"Using separate summarization model: {summarization_model_path}")
    else:
        summarization_model_path = actor_model_path
        summarization_model_revision = actor_model_revision
        logger.info("Using the same model for summarization as for solution generation")
    
    # Load tokenizer for chat template support
    from transformers import AutoTokenizer
    actor_tokenizer_path = cfg.get('tokenizer_path') or actor_model_path
    actor_tokenizer_revision = (
        actor_model_revision if cfg.get('tokenizer_path') is None else None
    )
    logger.info(f"Loading actor tokenizer from: {actor_tokenizer_path}")
    actor_tokenizer = AutoTokenizer.from_pretrained(
        actor_tokenizer_path, revision=actor_tokenizer_revision
    )
    summarization_tokenizer_path = cfg.get('summarization_tokenizer_path') or summarization_model_path
    summarization_tokenizer_revision = (
        summarization_model_revision if cfg.get('summarization_tokenizer_path') is None else None
    )
    logger.info(f"Loading summarization tokenizer from: {summarization_tokenizer_path}")
    summarization_tokenizer = AutoTokenizer.from_pretrained(
        summarization_tokenizer_path, revision=summarization_tokenizer_revision
    )
    
    # Get LLM parameters for summarization
    if cfg.get('summarization_llm') and cfg.summarization_llm.get('parameters'):
        summarization_llm_params = cfg.summarization_llm.parameters
    else:
        summarization_llm_params = cfg.llm.parameters
    
    # In eval-only mode, we don't need to collect logprobs since we're not creating training data
    eval_only_mode = cfg.get('eval_only', False)
    
    train_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_tokenizer_path),
            parameters=cfg.llm.parameters,
            use_cache=False,
            collect_logprobs=True,
            observe_llm_calls=False,
        )
        for url in llm_urls
    ]
    test_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_tokenizer_path),
            parameters=cfg.test_llm.parameters,
            use_cache=False,
            collect_logprobs=not eval_only_mode,  # Don't collect logprobs in eval-only mode
            observe_llm_calls=False,
        )
        for url in llm_urls
    ]
    
    # Initialize separate summarization LLMs if configured
    train_summarization_llms = None
    test_summarization_llms = None
    if summarization_llm_urls:
        train_summarization_llms = [
            TrainableLLM(
                base_url=url,
                model_name=str(summarization_model_path),
                tokenizer_name=str(summarization_tokenizer_path),
                parameters=summarization_llm_params,
                use_cache=False,
                collect_logprobs=True,
                observe_llm_calls=False,
            )
            for url in summarization_llm_urls
        ]
        test_summarization_llms = [
            TrainableLLM(
                base_url=url,
                model_name=str(summarization_model_path),
                tokenizer_name=str(summarization_tokenizer_path),
                parameters=summarization_llm_params,
                use_cache=False,
                collect_logprobs=not eval_only_mode,  # Don't collect logprobs in eval-only mode
                observe_llm_calls=False,
            )
            for url in summarization_llm_urls
        ]

    wait_for_inference_servers(llm_urls)
    if summarization_llm_urls:
        wait_for_inference_servers(summarization_llm_urls)
    wait_for_environments(cfg)
    
    eval_only_mode = cfg.get('eval_only', False)
    trainer_state = TrainerState(exp_path)
    
    if cfg.debug.mode or eval_only_mode:
        # In debug or eval-only mode, don't listen for weight updates
        logger.info("Debug or eval-only mode, not listening for weight updates")
        trainer_state.debug_mode_init()
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()

    # Load prompt templates from files
    def load_prompt_template(prompt_path: str, fallback: str) -> str:
        """Load prompt template from file, or use fallback if file doesn't exist"""
        if prompt_path:
            try:
                # Try relative to current working directory
                full_path = Path(prompt_path)
                if not full_path.exists():
                    # Try relative to experiment path
                    full_path = exp_path.parent / prompt_path
                
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        template = f.read()
                    logger.info(f"Loaded prompt template from: {full_path}")
                    return template
                else:
                    logger.warning(f"Prompt file not found: {prompt_path}, using fallback")
            except Exception as e:
                logger.warning(f"Error loading prompt file {prompt_path}: {e}, using fallback")
        return fallback
    
    # Get prompt templates from config files or use defaults
    reasoning_prompt_file = cfg.rc_actor.get("reasoning_prompt_file", None)
    summarization_prompt_file = cfg.rc_actor.get("summarization_prompt_file", None)
    
    reasoning_prompt_template = load_prompt_template(
        reasoning_prompt_file,
        fallback="Problem: {problem}\n\nCurrent summary: {curr_summary}\n\nContinue reasoning:"
    )
    summarization_prompt_template = load_prompt_template(
        summarization_prompt_file,
        fallback="Problem: {problem}\n\nExisting summary: {existing_summary}\n\nNew reasoning: {reasoning}\n\nProvide an updated summary:"
    )
    
    use_think_tags = cfg.rc_actor.get("use_think_tags", False)
    model_class = cfg.rc_actor.get("model_class", "qwen")
    reasoning_prompt_style = cfg.rc_actor.get("reasoning_prompt_style", "structured")
    summarization_style = cfg.rc_actor.get("summarization_style", "summ")
    
    logger.info(f"Model class: {model_class}")
    logger.info(f"Reasoning prompt style: {reasoning_prompt_style}")
    logger.info(f"Summarization style: {summarization_style}")
    logger.info(f"Use think tags: {use_think_tags}")

    # 0. If eval_only is True, set up the test loop and run it
    test_loop = RCActorLoop(
        data_stream=test_data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=test_stats_stream,
        llms=test_llms,
        summarization_llms=test_summarization_llms,
        reasoning_prompt_template=reasoning_prompt_template,
        summarization_prompt_template=summarization_prompt_template,
        tokenizer=actor_tokenizer,
        use_think_tags=use_think_tags,
        model_class=model_class,
        reasoning_prompt_style=reasoning_prompt_style,
        summarization_style=summarization_style,
        is_training=False,
    )
    if cfg.eval_only:
        logger.info("Create test loop")
        logger.info("Running test loop in eval-only mode")
        test_loop_run = test_loop.run(
            dataset=test_dataset,   
            cfg=cfg,
        )
        while True:
            if test_loop_run is not None:
                try:
                    _ = next(test_loop_run)
                except StopIteration:
                    logger.info("Test loop finished")
                    test_loop_run = None
    else:
        test_loop_run = None

        train_loop = RCActorLoop(
            data_stream=data_stream,
            cfg=cfg,
            trainer_state=trainer_state,
            stats_stream=stats_stream,
            llms=train_llms,
            summarization_llms=train_summarization_llms,
            reasoning_prompt_template=reasoning_prompt_template,
            summarization_prompt_template=summarization_prompt_template,
            tokenizer=actor_tokenizer,
            use_think_tags=use_think_tags,
            model_class=model_class,
            reasoning_prompt_style=reasoning_prompt_style,
            summarization_style=summarization_style,
            is_training=True,
        )
        train_loop_run = train_loop.run(
            dataset=train_dataset,
            cfg=cfg,
        )
        

        last_regular_eval = -1
        current_eval = -1
        skip_first_eval = cfg.get('skip_first_eval', False)
        logger.info(f"Skip first eval: {skip_first_eval}")
        while True:
            assert trainer_state.propagated_weight_version is not None
            # 1. Start a new test loop if needed
            next_regular_eval = (
                trainer_state.propagated_weight_version
                if last_regular_eval == -1
                else last_regular_eval + cfg.eval_every_n_versions
            )
            if (
                cfg.eval_every_n_versions
                and not cfg.debug.mode
                and trainer_state.propagated_weight_version >= next_regular_eval
                and test_dataset
                and test_loop_run is None
            ):
                logger.info("Create test loop")
                if skip_first_eval:
                    logger.info("Skipping first eval")
                    skip_first_eval = False
                    current_eval = next_regular_eval
                    last_regular_eval = current_eval
                    continue
                test_loop_run = test_loop.run(
                    dataset=test_dataset,   
                    cfg=cfg,
                )
                train_loop.is_scheduling_paused = True
                current_eval = next_regular_eval

            # 2. If there is an active test loop, keep it running
            if test_loop_run is not None:
                try:
                    _ = next(test_loop_run)
                except StopIteration:
                    # 2.1 If the test loop is finished, resume scheduling the training loop
                    test_loop_run = None
                    last_regular_eval = current_eval
                    train_loop.is_scheduling_paused = False
                    logger.info("Test loop finished")

            # 3. Keep running the training loop
            _ = next(train_loop_run)