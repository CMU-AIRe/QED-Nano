import asyncio
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
from pipelinerl.utils import strip_chat_template_tokens

import aiohttp
import aiohttp.client_exceptions
import hydra
import uvloop

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
    read_stream,
)

from .utils import (
    always_or_never_success_stats,
    calculate_stats,
    setup_logging,
    wait_for_environments,
    wait_for_inference_servers,
)

logger = logging.getLogger(__name__)

from pipelinerl.finetune.data import MASKED_TOKEN_ID



_WANDB_VERIFIER_TABLE_COLUMNS = ["group_index", "prompt", "reasoning", "output", "score"]
_WANDB_ROLLOUT_TABLE_COLUMNS = [
    "group_index",
    "prompt",
    "prompt_tokens",
    "reasoning",
    "reasoning_tokens",
    "output",
    "output_tokens",
    "total_tokens",
]


def split_reasoning_output(
    completion_text: str, reasoning_delimiters: list[str] | None
) -> tuple[str, str]:
    """
    Split completion text into reasoning and output parts using reasoning delimiters.
    """
    if reasoning_delimiters is None:
        return "", completion_text.strip()

    for delim in reasoning_delimiters:
        if not delim:
            continue
        if delim in completion_text:
            prefix, suffix = completion_text.rsplit(delim, 1)
            return prefix.strip(), suffix.strip()

    return completion_text.strip(), ""


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
        wandb.log({"tables/verifier_last_k": table})


class RolloutTableBuffer:
    """
    A bounded ring-buffer for actor rollout table entries.

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
        table = wandb.Table(columns=_WANDB_ROLLOUT_TABLE_COLUMNS)
        for group_entries in self._groups:
            for entry in group_entries:
                table.add_data(
                    entry.get("group_index", 0),
                    entry.get("prompt", ""),
                    entry.get("prompt_tokens", 0),
                    entry.get("reasoning", ""),
                    entry.get("reasoning_tokens", 0),
                    entry.get("output", ""),
                    entry.get("output_tokens", 0),
                    entry.get("total_tokens", 0),
                )
        return table

    def log_to_wandb(self) -> None:
        """Publish the current buffer as a table via wandb.log()."""
        if getattr(wandb, "run", None) is None:
            return
        table = self.to_wandb_table()
        wandb.log({"tables/rollouts_last_k": table})


def _aggregate_group_verifier_metrics(rollout_results: list[RolloutResult]) -> dict[str, float | int]:
    runtime_values: defaultdict[str, list[float]] = defaultdict(list)
    count_totals: defaultdict[str, int] = defaultdict(int)
    for result in rollout_results:
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

    total_rollouts = len(rollout_results)
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


def _log_group_verifier_metrics(metrics: dict[str, float | int]):
    if not metrics or getattr(wandb, "run", None) is None:
        return
    wandb.log(dict(metrics))


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
        self.data = SlidingWindowData()

    def update(self, prompt_tokens: list[int], output_tokens: list[int]):
        self.data.prompt_tokens_window.append(prompt_tokens)
        self.data.output_tokens_window.append(output_tokens)
        self.data.timestamps.append(time.time())
        if len(self.data.prompt_tokens_window) > self.window_size:
            self.data.prompt_tokens_window.pop(0)
            self.data.output_tokens_window.pop(0)
            self.data.timestamps.pop(0)

    def get_stats(self):
        if len(self.data.prompt_tokens_window) < self.window_size:
            return None

        # 1. How many samples do we produce per second?
        # 2. How many output tokens do we produce per second?
        # 3. How many prompt tokens do we produce per second?
        # 4. How many total tokens do we produce per second?
        null_stats = {
            "samples_per_second": 0,
            "output_tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "total_tokens_per_second": 0,
        }
        if not self.data.timestamps:
            return null_stats

        time_span = self.data.timestamps[-1] - self.data.timestamps[0]
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in self.data.prompt_tokens_window)
        total_output_tokens = sum(sum(tokens) for tokens in self.data.output_tokens_window)
        total_prompt_tokens = sum(sum(tokens) for tokens in self.data.prompt_tokens_window)

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
    scheduler_name: str,
):
    """This courotuine does the following.

    - It run asyncio loop for doing many rollouts in parallel using llm_async_generate
    - For each problem it does exactly `attempts` rollouts (let's call this a group)
    - It keeps track of how many rollout coroutines are running for each llms
    - it uses the LLM that has the least number of running coroutines for each new rollout
    - when all LLMs are busy it does nothing
    - It keeps track of how many rollouts are done for each group
    - When the group is done it puts the result in the result queue
    """
    loop = asyncio.get_running_loop()

    # Track active tasks per LLM
    active_rollouts = [0] * len(llms)
    started_rollouts = 0
    finished_rollouts = 0
    # Track rollouts per problem group
    group_rollouts = {}
    rollout_policy = hydra.utils.get_method(cfg.actor.rollout_policy)
    logger.info(f"Use rollout policy: {rollout_policy}")

    max_retries = cfg.actor.get("max_retries", 3)
    retry_base_delay = cfg.actor.get("retry_base_delay", 1.0)

    # Queue for rollouts that failed with transient errors and need to be retried
    # Each item is (problem, group_id, rollout_index, requeue_count)
    retry_queue: asyncio.Queue = asyncio.Queue()

    async def rollout_and_maybe_produce_result(
        problem: dict,
        group_id: int,
        rollout_index: int,
        llm_index: int,
        session: aiohttp.ClientSession,
        requeue_count: int = 0,
    ):
        nonlocal started_rollouts, finished_rollouts
        try:
            llm = llms[llm_index]
            model_version = trainer_state.propagated_weight_version
            assert model_version is not None

            # Retry loop for transient errors
            last_error = None
            for attempt in range(max_retries):
                try:
                    rollout_result = await rollout_policy(cfg, llm, problem, session)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = retry_base_delay * (2 ** attempt)
                        logger.warning(
                            f"Error in rollout (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.1f}s: {type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Error in rollout after {max_retries} attempts, giving up: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise
            else:
                # This shouldn't happen, but just in case
                raise last_error
            rollout_result.model_version = model_version
            # Make a group id that will be different from groups made by another rollout maker
            full_group_id = f"{scheduler_name}_{group_id}"
            rollout_result.group_id = full_group_id
            for step_index, sample in enumerate(rollout_result.training_texts):
                # Downstream in the pipeline we'll need these fields in every sample
                sample.metadata["model_version"] = model_version
                sample.metadata["rollout_index"] = rollout_index
                sample.metadata["step_index"] = step_index
                sample.group_id = full_group_id
            group_rollouts[group_id].append(rollout_result)
            if len(group_rollouts[group_id]) == attempts:
                # This is blocking call, but there's just one other thread reading from this queue.
                random.shuffle(group_rollouts[group_id])
                result_queue.put(group_rollouts[group_id])
                logger.info(f"Put group {group_id} of size {len(group_rollouts[group_id])} into result queue")
                del group_rollouts[group_id]
            finished_rollouts += 1
        except TRANSIENT_EXCEPTIONS as e:
            # Transient errors (HTTP/connection issues) that exhausted retries.
            # Re-queue the rollout to try again later, up to MAX_REQUEUE_ATTEMPTS times.
            if requeue_count < MAX_REQUEUE_ATTEMPTS:
                logger.warning(
                    f"Transient error in rollout for group {group_id}, re-queuing "
                    f"(attempt {requeue_count + 1}/{MAX_REQUEUE_ATTEMPTS}): {type(e).__name__}: {e}"
                )
                await retry_queue.put((problem, group_id, rollout_index, requeue_count + 1))
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

    group_id = -1
    group_rollout_index = attempts
    problem = None

    last_logged = time.time()
    logger.info("Starting rollout scheduler")
    connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
    timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while True:
            if time.time() - last_logged > 10.0 and sum(active_rollouts):
                retry_queue_size = retry_queue.qsize()
                logger.info(
                    f"{scheduler_name}: "
                    f"rollouts in progress: {sum(active_rollouts)}: {active_rollouts}, "
                    f"groups in progress: {len(group_rollouts)}, "
                    f"rollouts started so far: {started_rollouts}, "
                    f"rollouts finished so far: {finished_rollouts}, "
                    f"problem queue size: {problem_queue.qsize()}, "
                    f"result queue size: {result_queue.qsize()}, "
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
                retry_problem, retry_group_id, retry_rollout_index, requeue_count = retry_item
                next_llm = active_rollouts.index(min(active_rollouts))
                if active_rollouts[next_llm] == cfg.actor.llm_max_rollouts:
                    # All LLMs are busy, put item back and wait
                    await retry_queue.put(retry_item)
                    await asyncio.sleep(0.01)
                    continue
                active_rollouts[next_llm] += 1
                started_rollouts += 1
                loop.create_task(
                    rollout_and_maybe_produce_result(
                        problem=retry_problem,
                        group_id=retry_group_id,
                        rollout_index=retry_rollout_index,
                        llm_index=next_llm,
                        session=session,
                        requeue_count=requeue_count,
                    )
                )
                continue

            # Then, check if we need to start a new group
            if group_rollout_index == attempts:
                try:
                    problem = problem_queue.get(block=False)
                except Empty:
                    # give some quality time for other couroutines to work
                    await asyncio.sleep(0.01)
                    continue
                group_id += 1
                group_rollouts[group_id] = []
                group_rollout_index = 0

            next_llm = active_rollouts.index(min(active_rollouts))
            if active_rollouts[next_llm] == cfg.actor.llm_max_rollouts:
                # all llms are busy, wait for one to finish
                logger.info(f"{scheduler_name}: All LLMs are busy, waiting for one to finish. Current active rollouts {active_rollouts}.")
                await asyncio.sleep(2)
                continue
            active_rollouts[next_llm] += 1
            started_rollouts += 1
            assert problem is not None
            loop.create_task(
                rollout_and_maybe_produce_result(
                    problem=problem,
                    group_id=group_id,
                    rollout_index=group_rollout_index,
                    llm_index=next_llm,
                    session=session,
                )
            )
            group_rollout_index += 1
    logger.info("Rollout scheduler finished")


def rollout_maker_entrypoint(
    cfg: DictConfig,
    attempts: int,
    problem_queue: SharedMemoryQueue,
    result_queue: SharedMemoryQueue,
    llms: list[TrainableLLM],
    scheduler_name: str,
):
    trainer_state = TrainerState(Path(cfg.output_dir))
    if cfg.debug.mode:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        schedule_rollouts(cfg, attempts, problem_queue, result_queue, trainer_state, llms, scheduler_name)
    )
    loop.close()
    logger.info("Rollout maker loop closed")


def random_iter(problems: list):
    while True:
        yield random.sample(problems, 1)[0]


def sequential_iter(problems: list):
    for problem in problems:
        yield problem


def stream_iter(stream_reader, num_samples_per_batch: int = 3, is_training: bool = True):
    """
    Read from RC actor stream and yield problems from reasoning turns.
    
    Args:
        stream_reader: StreamReader instance to read from
        num_samples_per_batch: Number of samples to randomly select from each batch
        is_training: If True, sample randomly; if False, take all reasoning turns
    """
    for batch in stream_reader.read():
        # batch is a list of dicts (training text dumps)
        # Filter for reasoning turns only
        reasoning_samples = [
            sample for sample in batch 
            if sample.get("metadata", {}).get("turn_type") == "reasoning"
        ]
        
        if not reasoning_samples:
            continue
        
        # Subsample if training
        if is_training and len(reasoning_samples) > num_samples_per_batch:
            # Sample without replacement (random.sample already does this)
            reasoning_samples = random.sample(reasoning_samples, num_samples_per_batch)
        
        # Convert each sample to a problem dict that the actor can use
        for sample in reasoning_samples:
            # Extract the problem from the prompt_text or reconstruct it
            # The training text should have prompt_text and output_text
            # Strip chat template tokens from prompt_text to get the raw task
            raw_prompt = strip_chat_template_tokens(sample.get("prompt_text", ""))
            problem = {
                "task": raw_prompt,
                "answer": sample.get("metadata", {}).get("answer", ""),  
                "dataset": sample.get("metadata", {}).get("dataset_name", "unknown") + "_turn_" + str(sample.get("metadata", {}).get("turn_number", 0)),
                "id": sample.get("metadata", {}).get("problem_id", 0),
                "turn_number": sample.get("metadata", {}).get("turn_number", 0),
            }
            # Add schema and original problem if present (for LLM-based proof verification)
            if "schema" in sample.get("metadata", {}):
                problem["schema"] = sample["metadata"]["schema"]
            if "original_problem" in sample.get("metadata", {}):
                problem["original_problem"] = sample["metadata"]["original_problem"]
            yield problem


class ActorLoop:
    def __init__(
        self,
        cfg: DictConfig,
        llms: list[TrainableLLM],
        data_stream: StreamSpec,
        stats_stream: StreamSpec,
        trainer_state: TrainerState,
        is_training: bool = True,
    ) -> None:
        self.data_stream = data_stream
        self.trainer_state = trainer_state
        self.stats_stream = stats_stream
        self.sliding_aggregator = SlidingWindowAggregator(window_size=cfg.actor.throughput_window_size)
        self.llms = llms
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

        # Setup rollout table buffer for actor rollouts
        llm_cfg = cfg.get("llm", None)
        rollout_wandb_table_cfg = llm_cfg.get("wandb_table", None) if llm_cfg is not None else None
        self.rollout_table_enabled = True
        rollout_keep_last_k_groups = 32
        rollout_log_every_n_groups = 32
        if rollout_wandb_table_cfg is not None:
            self.rollout_table_enabled = rollout_wandb_table_cfg.get("enabled", True)
            rollout_keep_last_k_groups = rollout_wandb_table_cfg.get("keep_last_k_groups", 32)
            rollout_log_every_n_groups = rollout_wandb_table_cfg.get("log_every_n_groups", 32)
        self.rollout_table_buffer = RolloutTableBuffer(
            keep_last_k_groups=rollout_keep_last_k_groups,
            log_every_n_groups=rollout_log_every_n_groups,
        )

        # Get reasoning delimiters: primary from llm.reasoning_delimiters, fallback to llm_grader.reasoning_delimiters
        self.reasoning_delimiters: list[str] | None = None
        if llm_cfg is not None:
            delims = llm_cfg.get("reasoning_delimiters", None)
            if delims:
                self.reasoning_delimiters = list(delims)
        if self.reasoning_delimiters is None and llm_grader_cfg is not None:
            delims = llm_grader_cfg.get("reasoning_delimiters", None)
            if delims:
                self.reasoning_delimiters = list(delims)

        # Load tokenizer for counting reasoning/output tokens
        self._tokenizer = None
        self._tokenizer_load_attempted = False

        # Determine the number of processes to use
        num_processes = min(self.cfg.actor.rollout_workers, len(self.llms))
        attempts = self.cfg.attempts if is_training else 1

        # Divide LLMs approximately equally across processes
        llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            llm_groups[i % num_processes].append((i, llm))

        self.smm = SharedMemoryManager()
        self.smm.start()

        
        # Use SharedMemoryQueue instead of separate problem_queue, result_queue, and io_buffer
        self.problem_queue = SharedMemoryQueue(self.smm, self.cfg.actor.problem_queue_size, cfg.actor.shared_memory_entry_size)
        self.result_queue = SharedMemoryQueue(self.smm, self.cfg.actor.result_queue_size, cfg.actor.shared_memory_entry_size)
        
        logger.info(f"Initialized {'train' if self.is_training else 'test'} actor loop")
        logger.info(f"Problem queue size: {self.problem_queue.max_size}, result queue size: {self.result_queue.max_size}")
        logger.info(f"Result queue buffer size: {self.result_queue.get_memory_size() / 2**30} Gb")

        # Create and start multiple rollout processes
        self.rollout_processes = []
        for llm_group in llm_groups:
            assert llm_group
            llm_idxs = [llm[0] for llm in llm_group]
            llms = [llm[1] for llm in llm_group]
            scheduler_name = (
                f"{'train' if is_training else 'test'} scheduler for llms {','.join([str(i) for i in llm_idxs])}"
            )
            process = mp.Process(
                target=rollout_maker_entrypoint,
                args=(self.cfg, attempts, self.problem_queue, self.result_queue, llms, scheduler_name),
            )
            process.start()
            self.rollout_processes.append(process)

    def _maybe_load_rollout_tokenizer(self) -> None:
        """
        Lazily load a tokenizer for rollout table token counting.

        Important: this must happen after starting rollout worker processes to avoid
        forking/spawning with a loaded tokenizer (which can be non-pickleable or
        unsafe to fork depending on backend).
        """
        if self._tokenizer is not None or self._tokenizer_load_attempted:
            return
        self._tokenizer_load_attempted = True
        if not self.llms:
            return
        try:
            loaded = self.llms[0].load_tokenizer()
            self._tokenizer = loaded or getattr(self.llms[0], "tokenizer", None)
            if self._tokenizer is None:
                logger.warning(
                    "Tokenizer load succeeded but no tokenizer was set on the LLM; rollout token counts will be 0"
                )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for rollout table token counting: {e}")

    def _decode_completion_from_training_text(self, training_text) -> str:
        labels = getattr(training_text, "labels", None) or []
        if self._tokenizer is not None and labels:
            try:
                generated_token_ids = [tid for tid in labels if tid != MASKED_TOKEN_ID]
                if generated_token_ids:
                    return self._tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            except Exception as e:
                logger.warning(f"Failed to decode completion from token IDs: {e}")
        return getattr(training_text, "output_text", "") or ""

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

    def update_stats(self, rollout_results: list[RolloutResult]):
        for result in rollout_results:
            assert result.model_version is not None
            assert isinstance(result.metrics, BaseMetrics), "Metrics should be an instance of BaseMetrics"
            dataset_name = result.dataset_name
            group_id = result.group_id
            self.latency_list.append(result.latency)
            self.model_versions_list.append(result.model_version)
            domain_agnostic_metrics = self.compute_domain_agnostic_metrics(result) 
            all_metrics = result.metrics.model_dump() | domain_agnostic_metrics
            for k, v in all_metrics.items():
                if isinstance(v, list):
                    self.stats[k][dataset_name][group_id] += v
                elif isinstance(v, float) | isinstance(v, bool) | isinstance(v, int):
                    self.stats[k][dataset_name][group_id].append(v)
                else:
                    raise ValueError(f"Unsupported metric type: {type(v)} for key {k}")
        
        prompt_length_tokens = [training_text.prompt_tokens for result in rollout_results for training_text in result.training_texts]
        output_length_tokens = [training_text.output_tokens for result in rollout_results for training_text in result.training_texts]
        self.sliding_aggregator.update(prompt_length_tokens, output_length_tokens)
        sliding_window_stats = self.sliding_aggregator.get_stats()
        if sliding_window_stats is not None:
            for k, v in sliding_window_stats.items():
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

    def log_verifier_metrics_for_group(self, rollout_results: list[RolloutResult]) -> None:
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
        _log_group_verifier_metrics(aggregated)
        return



    def run(self, dataset):
        """
        Run the actor loop.
        
        Args:
            dataset: Either a list of problems or an iterator/generator yielding problems
        """
        loop_start_time = time.time()
        self.init_stats()

        attempts = self.cfg.attempts if self.is_training else 1
        published_samples = 0
        submitted_groups = 0
        finished_groups = 0
        
        # Check if dataset is an iterator/generator or a list
        # Simple check: lists have __len__, generators/iterators don't
        is_iterator = not isinstance(dataset, (list, tuple))
        
        if is_iterator:
            # Dataset is already an iterator (e.g., from stream_iter)
            expected_rollouts = -1  # Unknown length for iterators
            problem_iter = dataset
            logger.info("Using stream-based iterator for problems")
        else:
            # Dataset is a list (traditional behavior)
            expected_rollouts = -1 if self.is_training else len(dataset)
            if expected_rollouts > 0:
                logger.info(f"Will stop after {expected_rollouts} rollouts")
            
            # If training, we expect to sample infinitely
            # for train sample, sample random batches infinitely
            # for test samples, loop through the dataset once
            if self.is_training:
                problem_iter = random_iter(dataset)
            else:
                problem_iter = sequential_iter(dataset)
        
        assert self.trainer_state.propagated_weight_version is not None

        last_trainer_version = self.trainer_state.propagated_weight_version
        trainer_version_to_publish = self.trainer_state.propagated_weight_version

        logger.info(f"Trainer version to publish: {trainer_version_to_publish}, last trainer version: {last_trainer_version}")
        max_lag = self.cfg.finetune.max_lag if self.is_training else None
        if max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = (
                math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            )
            if total_batch_size % self.cfg.attempts != 0:
                logger.warning(
                    f"I'm trying to submit the exact right number of groups for this batch."
                    f" The attempt number  {self.cfg.attempts} ideally should divide"
                    f" total batch size {total_batch_size}"
                )
            groups_per_update = math.ceil(total_update_size / self.cfg.attempts)
            lag_groups = math.ceil(self.cfg.finetune.max_lag / self.cfg.attempts)
            logger.info(
                f"Sync RL mode on, can submit {groups_per_update} groups for each update,"
                f" that makes {groups_per_update * self.cfg.attempts} samples per update"
            )
            logger.info(
                f"Max lag is {self.cfg.finetune.max_lag} samples, that makes {lag_groups} additional starting chunks"
            )
            can_submit_before_update = lag_groups + groups_per_update
        else:
            groups_per_update = None
            can_submit_before_update = math.inf

        logger.info(f"Start {'train' if self.is_training else 'test'} actor loop")
        with (
            write_to_streams(self.data_stream, "a") as data_stream_writer,
            write_to_streams(self.stats_stream, "a") as stats_writer,
        ):
            while True:
                # the user function must do next(...) to run each iteration
                yield

                if self.trainer_state.propagated_weight_version > last_trainer_version:
                    if max_lag is not None:
                        assert groups_per_update is not None
                        can_submit_before_update += groups_per_update
                    # the weights have been updated, publish the stats of the previous trainer version
                    trainer_version_to_publish = last_trainer_version
                    last_trainer_version = self.trainer_state.propagated_weight_version
                    logger.info(f"Weights updated to version {trainer_version_to_publish}, last trainer version: {last_trainer_version}")

                # First, try to read a result (do this FIRST to avoid blocking on problem submission)
                result_consumed = False
                try:
                    # Directly get the result from the SharedMemoryQueue
                    rollout_results = self.result_queue.get(block=False)
                    result_consumed = True
                    logger.info(f"Got rollout results of size {len(rollout_results)}. Finished groups: {finished_groups}")
                except queue.Empty:
                    rollout_results = None
                
                # Second, try to submit problems if queue not full
                if not self.is_scheduling_paused:
                    while True:
                        blocked_by_lag = submitted_groups == can_submit_before_update and self.is_training
                        if not blocked_by_lag and not result_consumed and not self.problem_queue.full():
                            try:
                                try:
                                    problem = next(problem_iter)
                                    self.problem_queue.put(problem, block=False)
                                    submitted_groups += 1
                                    logger.info(f"Submitted problem: {problem['id']} turn {problem['turn_number'] if 'turn_number' in problem else '0'} Submitted groups: {submitted_groups}")
                                    break
                                except queue.Full:            
                                    assert False, "Problem queue was not full just a moment ago, but now it is full"
                            except StopIteration:
                                break
                        else:
                            break
                
                # Third, process the result if we got one
                if not result_consumed:
                    continue

                if isinstance(rollout_results, Exception):
                    logger.error("Stop actor loop due to error")
                    raise rollout_results

                assert isinstance(rollout_results, list)
                assert isinstance(rollout_results[0], RolloutResult)
                group_samples = sum(len(r.training_texts) for r in rollout_results)

                published_samples += group_samples
                samples_in_queue = self.result_queue.qsize() * attempts
                all_text_dumps = []
                for r in rollout_results:
                    for text in r.training_texts:
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

                # Log rollout table entries
                if self.cfg.wandb.use_wandb and self.rollout_table_enabled:
                    self._maybe_load_rollout_tokenizer()
                    group_index_value = finished_groups + 1
                    rollout_group_entries: list[dict[str, str | int]] = []
                    for result in rollout_results:
                        if not result.training_texts:
                            continue
                        # Use the last TrainingText in the rollout
                        training_text = result.training_texts[-1]
                        prompt = training_text.prompt_text
                        completion = self._decode_completion_from_training_text(training_text)
                        prompt_tokens = training_text.prompt_tokens
                        total_tokens = training_text.output_tokens

                        # Split completion into reasoning and output
                        reasoning, output = split_reasoning_output(
                            completion, self.reasoning_delimiters
                        )

                        # Count reasoning_tokens and output_tokens using tokenizer
                        reasoning_tokens = 0
                        output_tokens = 0
                        if self._tokenizer is not None:
                            try:
                                if reasoning:
                                    reasoning_tokens = len(
                                        self._tokenizer.encode(reasoning, add_special_tokens=False)
                                    )
                                if output:
                                    output_tokens = len(
                                        self._tokenizer.encode(output, add_special_tokens=False)
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to tokenize for rollout table: {e}")

                        rollout_entry: dict[str, str | int] = {
                            "group_index": group_index_value,
                            "prompt": prompt,
                            "prompt_tokens": prompt_tokens,
                            "reasoning": reasoning,
                            "reasoning_tokens": reasoning_tokens,
                            "output": output,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                        }
                        rollout_group_entries.append(rollout_entry)

                    if rollout_group_entries:
                        self.rollout_table_buffer.add_group(rollout_group_entries)
                        if self.rollout_table_buffer.should_log():
                            try:
                                self.rollout_table_buffer.log_to_wandb()
                            except Exception as e:
                                logger.error(f"Failed to log rollout table to wandb: {e}")

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
                    else:
                        loop_stats = {
                            "trainer_model_version": last_trainer_version
                            }

                    self.publish_stats(
                        stats_writer=stats_writer,
                        loop_stats=loop_stats,
                    )
                    trainer_version_to_publish = None


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
            wandb.log({f"actor/{k}": v for k, v in stats.items()})
        stats_writer.write(stats)
        self.init_stats()  # Reset stats for the next iteration


def run_actor_loop(cfg: DictConfig):
    set_streams_backend(**cfg.streams)

    # set seed for reproducibility (mostly intended for dataset loading)
    random.seed(cfg.seed)

    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path / "actor", "actor")
    logger.info(f"Current dir: {os.getcwd()}, experiment root dir: {cfg.output_dir}")
    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "actor", flatten_dict_config(cfg))  # type: ignore
        if run is None:
            raise ValueError("Failed to initialize wandb run")
        wandb.define_metric("verifier/*", step_metric="verifier/group_index")
    llm_urls = str(cfg.me.llm_urls).split("+")

    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
    test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats_test")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
    test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor_test")

    # Check if we should read from RC actor stream instead of dataset
    use_rc_stream = cfg.actor.get("use_rc_stream", False)
    
    # Initialize dataset loader (needed for test dataset in both cases)
    dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
    dataset_loader_params = cfg.get('dataset_loader_params', {})
    
    # Always load test dataset from files for consistent evaluation
    test_dataset = dataset_loader(cfg.test_dataset_names, **dataset_loader_params)
    logger.info(f"Loaded {len(test_dataset)} test problems")
    
    if use_rc_stream:
        # Read training data from RC actor stream
        rc_stream_topic = cfg.actor.get("rc_stream_topic", "actor")
        rc_train_stream = SingleStreamSpec(exp_path=exp_path, topic=rc_stream_topic)
        
        logger.info(f"Reading training data from RC actor stream: {rc_train_stream}")
        
        train_dataset = None
        train_stream_reader = read_stream(rc_train_stream)
    else:
        # Original behavior: load training dataset from files
        train_dataset = dataset_loader(cfg.train_dataset_names, **dataset_loader_params)
        if cfg.train_subset:
            train_dataset = train_dataset[cfg.train_subset.begin : cfg.train_subset.end]
        logger.info(f"Loaded {len(train_dataset)} training problems")
        
        train_stream_reader = None

    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path
        if actor_model_path is None:
            raise ValueError("model_path must be defined")
    
    train_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
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
            tokenizer_name=str(actor_model_path),
            parameters=cfg.test_llm.parameters,
            use_cache=False,
            collect_logprobs=True,
            observe_llm_calls=False,
        )
        for url in llm_urls
    ]

    wait_for_inference_servers(llm_urls)
    wait_for_environments(cfg)
    trainer_state = TrainerState(exp_path)
    if cfg.debug.mode:
        trainer_state.debug_mode_init()
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()

    # Prepare dataset or stream reader for training
    # Note: test_dataset is always loaded from files (above) for consistent evaluation
    if use_rc_stream:
        # Enter the stream readers context
        train_stream_reader = train_stream_reader.__enter__()
        
        # Create stream-based training dataset
        num_samples_per_batch = cfg.actor.get("rc_stream_samples_per_batch", 3)
        train_dataset_final = stream_iter(train_stream_reader, num_samples_per_batch, is_training=True)
        
        logger.info(f"Reading {num_samples_per_batch} reasoning samples per batch from RC stream")
    else:
        train_dataset_final = train_dataset
        train_stream_reader = None
        
    train_loop = ActorLoop(
        data_stream=data_stream, cfg=cfg, trainer_state=trainer_state, stats_stream=stats_stream, llms=train_llms
    )
    train_loop_run = train_loop.run(
        dataset=train_dataset_final,
    )
    test_loop = ActorLoop(
        data_stream=test_data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=test_stats_stream,
        llms=test_llms,
        is_training=False,
    )
    test_loop_run = None

    last_regular_eval = -1
    current_eval = -1
    skip_first_eval = cfg.get("skip_first_eval", False)
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
