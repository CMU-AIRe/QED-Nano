import logging
import os
from typing import Literal
from pydantic import BaseModel
from omegaconf import DictConfig
import torch
import re

logger = logging.getLogger(__name__)


def expand_nodelist(nodelist: str):
    s = nodelist.replace(" ", "")
    # If it's just a comma list with no brackets:
    if "[" not in s and "," in s:
        return [h for h in s.split(",") if h]
    # Bracketed form: prefix[spec]
    m = re.fullmatch(r"(.*)\[(.+)\]", s)
    if m:
        prefix, spec = m.groups()
        hosts = []
        for chunk in spec.split(","):
            # range like 07-12 (keep zero padding width)
            r = re.fullmatch(r"(\d+)-(\d+)", chunk)
            if r:
                a, b = r.groups()
                width = max(len(a), len(b))
                for i in range(int(a), int(b) + 1):
                    hosts.append(f"{prefix}{i:0{width}d}")
            else:
                # single index (could be zero-padded)
                hosts.append(f"{prefix}{chunk}")
        return hosts
    # Single host or simple comma list fallback
    return [s] if s else []


class Job(BaseModel):
    """Represent the decision to launch a replica of a particular worker (e.g. actor) at a particular rank"""
    # The job kind 
    kind: str
    # The global index of this job among all jobs
    idx: int 
    # The index of this job among jobs of the same kind
    replica_idx: int
    # The index of this job among similar jobs on the same node
    local_idx: int = 0
    # Where this job should run
    node_rank: int
    hostname: str 
    port: int | None = None
    # Which GPUs the job will use
    gpus: list[int] = []
    # The URL of the job
    url: str = ""


class WorldMap:
    def __init__(self, cfg: DictConfig, verbose: bool = False):
        self._log_info = logger.info if verbose else lambda x: None

        self.cfg = cfg
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.my_rank = int(os.environ.get("RANK", 0))
        self.address_map = {}
        if self.world_size > 1:
            nodelist = os.environ.get("ALL_ADDR", "")
            assert nodelist is not None, "ALL_ADDR is not set even though WORLD_SIZE > 1"
            nodelist = [x.strip() for x in nodelist.strip().split(",")]
            assert len(nodelist) == self.world_size, f"SLURM_NODELIST length {len(nodelist)} does not match WORLD_SIZE {self.world_size}"
            self.master_addr = nodelist[0]
            for rank in range(self.world_size):
                self.address_map[rank] = nodelist[rank]

            # nodelist = expand_nodelist(nodelist)
            # nodelist = nodelist.strip().split(",")
            # e.g.: dns-f6c9712f-4d9b-4c8d-a648-f8d94cf12113-0
            # for rank in range(self.world_size):
            #     basename = self.master_addr[: self.master_addr.rfind("-")]
            #     self.address_map[rank] = f"{basename}-{rank}"
        else:
            self.master_addr = "localhost"
            self.address_map[0] = "localhost"
        
        self.nodelist = [self.address_map[i] for i in range(self.world_size)]

        self._log_info(f"--- INITIALIZE WORLD MAP (this is rank {self.my_rank}) ---")

        # Calculate GPUs per LLM for each type based on TP/PP settings
        # RC Actor LLMs (use rc_actor_vllm_config if available, else vllm_config)
        rc_actor_vllm_cfg = self.cfg.get("rc_actor_vllm_config")
        if rc_actor_vllm_cfg is None:
            rc_actor_vllm_cfg = self.cfg.vllm_config
        rc_actor_llm_kwargs = rc_actor_vllm_cfg.vllm_kwargs
        rc_actor_tp = rc_actor_llm_kwargs.get("tensor-parallel-size", 1)
        rc_actor_pp = rc_actor_llm_kwargs.get("pipeline-parallel-size", 1)
        self.gpus_per_rc_actor_llm = rc_actor_tp * rc_actor_pp
        if cfg.world.get("rc_actor_fraction", 0) == 0:
            self.gpus_per_rc_actor_llm = 0
        
        # Regular Actor LLMs (use actor_vllm_config if available, else vllm_config)
        actor_llm_kwargs = self.cfg.get("actor_vllm_config", self.cfg.vllm_config).vllm_kwargs
        actor_tp = actor_llm_kwargs.get("tensor-parallel-size", 1)
        actor_pp = actor_llm_kwargs.get("pipeline-parallel-size", 1)
        self.gpus_per_actor_llm = actor_tp * actor_pp
        if cfg.world.get("actor_fraction", 0) == 0:
            self.gpus_per_actor_llm = 0

        # Summarization LLMs (use summarization_vllm_config if available, else vllm_config)
        summ_llm_kwargs = self.cfg.get("summarization_vllm_config", self.cfg.vllm_config).vllm_kwargs
        summ_tp = summ_llm_kwargs.get("tensor-parallel-size", 1)
        summ_pp = summ_llm_kwargs.get("pipeline-parallel-size", 1)
        self.gpus_per_summarization_llm = summ_tp * summ_pp
        if cfg.world.get("summarization_fraction", 0) == 0:
            self.gpus_per_summarization_llm = 0

        # Preprocessor LLMs (use vllm_config)
        self.gpus_per_preprocessor_llm = self.gpus_per_rc_actor_llm  # Same as RC actor by default
        
        # Legacy: keep gpus_per_llm for backward compatibility (use actor LLM value)
        self.gpus_per_llm = self.gpus_per_actor_llm
        
        self.node_size = 8 if self.world_size > 1 else torch.cuda.device_count()
        
        self._log_info(
            f"GPUs per LLM: RC Actor={self.gpus_per_rc_actor_llm}, "
            f"Actor={self.gpus_per_actor_llm}, "
            f"Summarization={self.gpus_per_summarization_llm}, "
            f"Preprocessor={self.gpus_per_preprocessor_llm}"
        )

        place_inference_jobs = not cfg.debug.mode or cfg.debug.place_inference_workers
        if place_inference_jobs:
            self._split_gpus_by_purpose(cfg)
        else:
            self.total_finetune_gpus = self.node_size * self.world_size
            # placeholder value, wont't be used
            self.weight_update_group_size = 1

        # Place jobs on nodes in a reverse order to make sure that last node has a finetuning job going on
        self.available_gpus = {i: set(range(self.node_size)) for i in reversed(range(self.world_size))}
        self.cpu_heavy_jobs = {i: 0 for i in range(self.world_size)} 
        self.job_map = {i: [] for i in range(self.world_size)}
        self.total_jobs = 0

        if place_inference_jobs:
            self._place_inference_jobs(cfg)
        self._place_pipeline_stages(cfg)
        if cfg.environment:
            self._place_environments(cfg)

        # Place the finetune workers on the remaining gpus, take all remaining GPUs
        current_finetune_rank = 0
        finetune_rank_node = {}
        for node, remaining_gpus in self.available_gpus.items():
            gpus = list(remaining_gpus)
            if gpus:
                self.add_job(node_rank=node, kind="finetune", replica_idx=node, gpus=gpus)
                for _ in remaining_gpus:
                    finetune_rank_node[current_finetune_rank] = node
                    current_finetune_rank += 1

        assert current_finetune_rank == self.total_finetune_gpus
        if self.total_finetune_gpus > 0 and self.total_finetune_gpus % cfg.finetune.seq_parallel != 0:
            raise ValueError(
                f"Total finetune GPUs {self.total_finetune_gpus} is not divisible by seq_parallel {cfg.finetune.seq_parallel}"
            )
        
        if self.total_finetune_gpus > 0:
            for leader_idx in range(0, current_finetune_rank, cfg.finetune.seq_parallel):
                # Check that all workers in the leader's group are on the same node
                leader_node = finetune_rank_node[leader_idx]
                for offset in range(cfg.finetune.seq_parallel):
                    if finetune_rank_node[leader_idx + offset] != leader_node:
                        raise ValueError(
                            f"Sequence parallel ranks {leader_idx} and {leader_idx + offset} are on different nodes: "
                            f"{finetune_rank_node[leader_idx]} and {finetune_rank_node[leader_idx + offset]}"
                        )


        # Pretty-log the world map
        self._log_info("--- WORLD MAP ---")
        for node, jobs in self.job_map.items():
            self._log_info(f"Node {node} has {len(jobs)} jobs:")
            for job in jobs:
                self._log_info(f"  {job.kind} {job.replica_idx} on gpus {job.gpus}, local idx {job.local_idx}")

    def add_job(self, node_rank: int, kind: str, replica_idx: int, local_idx: int = 0, port: int | None = None, gpus: list[int] | None = None, cpu_heavy: bool = False, url: str = "") -> Job:
        """Add a job to the world map."""
        if gpus is None:
            gpus = []
        job = Job(
            kind=kind,             
            idx=self.total_jobs,
            replica_idx=replica_idx,
            local_idx=local_idx, 
            node_rank=node_rank, 
            hostname=self.address_map[node_rank],
            port=port,
            gpus=gpus,
            url=url
        )       
        self.job_map[node_rank].append(job)
        self.total_jobs += 1
        if cpu_heavy:
            self.cpu_heavy_jobs[node_rank] += 1
        return job


    def _split_gpus_by_purpose(self, cfg):
        # Check if we're using RC actor mode
        rc_actor_fraction = cfg.world.get("rc_actor_fraction", 0)
        summarization_fraction = cfg.world.get("summarization_fraction", 0)
        
        fraction_sum = (
            rc_actor_fraction + cfg.world.actor_fraction + summarization_fraction +
            cfg.world.preprocessor_fraction + cfg.world.finetune_fraction
        )
        
        # TODO: support nodes with less than 8 GPUs available
        total_gpus = self.world_size * self.node_size
        
        # Calculate desired GPU shares (use appropriate GPUs per LLM for each type)
        desired_rc_actor_gpu_share = (
            max(int(total_gpus * rc_actor_fraction / fraction_sum), self.gpus_per_rc_actor_llm) if rc_actor_fraction else 0
        )
        desired_actor_gpu_share = max(int(total_gpus * cfg.world.actor_fraction / fraction_sum), self.gpus_per_actor_llm)
        desired_summarization_gpu_share = (
            max(int(total_gpus * summarization_fraction / fraction_sum), self.gpus_per_summarization_llm) if summarization_fraction else 0
        )
        desired_preprocessor_gpu_share = (
            max(int(total_gpus * cfg.world.preprocessor_fraction / fraction_sum), self.gpus_per_preprocessor_llm) if cfg.world.preprocessor_fraction else 0
        )
        desired_finetune_gpu_share = (
            total_gpus - desired_rc_actor_gpu_share - desired_actor_gpu_share - 
            desired_summarization_gpu_share - desired_preprocessor_gpu_share
        )
        
        self._log_info(
            f"Desired GPU share: {desired_rc_actor_gpu_share} for RC actors, "
            f"{desired_actor_gpu_share} for actors, {desired_summarization_gpu_share} for summarization, "
            f"{desired_preprocessor_gpu_share} for preprocessors, {desired_finetune_gpu_share} for finetune"
        )

        # Calculate GPUs per worker (use appropriate GPUs per LLM for each type)
        gpus_per_rc_actor = (
            int(desired_rc_actor_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 and rc_actor_fraction else 0
        )
        if self.gpus_per_rc_actor_llm > 0:
            gpus_per_rc_actor = gpus_per_rc_actor - (gpus_per_rc_actor % self.gpus_per_rc_actor_llm)
        
        gpus_per_actor = int(desired_actor_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 else 0
        if self.gpus_per_actor_llm > 0:
            gpus_per_actor = gpus_per_actor - (gpus_per_actor % self.gpus_per_actor_llm)
        
        gpus_per_summarization = (
            int(desired_summarization_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 and summarization_fraction else 0
        )
        if self.gpus_per_summarization_llm > 0:
            gpus_per_summarization = gpus_per_summarization - (gpus_per_summarization % self.gpus_per_summarization_llm)
        
        gpus_per_preprocessor = (
            int(desired_preprocessor_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 else 0
        )
        if self.gpus_per_preprocessor_llm > 0:
            gpus_per_preprocessor = gpus_per_preprocessor - (gpus_per_preprocessor % self.gpus_per_preprocessor_llm)
        
        # Calculate LLMs per worker (use appropriate GPUs per LLM for each type)
        self.llms_per_rc_actor = max(int(gpus_per_rc_actor / self.gpus_per_rc_actor_llm), 1) if gpus_per_rc_actor > 0 else 0
        self.total_rc_actor_llms = self.llms_per_rc_actor * cfg.world.replicas
        
        self.llms_per_actor = max(int(gpus_per_actor / self.gpus_per_actor_llm), 1) if gpus_per_actor > 0 else 0
        self.total_actor_llms = self.llms_per_actor * cfg.world.replicas
        
        self.llms_per_summarization = max(int(gpus_per_summarization / self.gpus_per_summarization_llm), 1) if gpus_per_summarization > 0 else 0
        self.total_summarization_llms = self.llms_per_summarization * cfg.world.replicas
        
        self.llms_per_preprocessor = (
            max(int(gpus_per_preprocessor / self.gpus_per_preprocessor_llm), 1) if gpus_per_preprocessor > 0 else 0
        )
        
        self.gpus_per_rc_actor = gpus_per_rc_actor
        self.gpus_per_actor = gpus_per_actor
        self.gpus_per_summarization = gpus_per_summarization
        self.gpus_per_preprocessor = gpus_per_preprocessor

        total_rc_actor_gpus = cfg.world.replicas * gpus_per_rc_actor
        total_actor_gpus = cfg.world.replicas * gpus_per_actor
        total_summarization_gpus = cfg.world.replicas * gpus_per_summarization
        total_preprocessor_gpus = cfg.world.replicas * gpus_per_preprocessor
        self.total_finetune_gpus = (
            total_gpus - total_rc_actor_gpus - total_actor_gpus - 
            total_summarization_gpus - total_preprocessor_gpus
        )
        
        self._log_info(
            f"The configuration required:\n"
            f"{desired_rc_actor_gpu_share} for RC actors ({self.gpus_per_rc_actor_llm} GPUs/LLM), "
            f"{desired_actor_gpu_share} for actors ({self.gpus_per_actor_llm} GPUs/LLM), "
            f"{desired_summarization_gpu_share} for summarization ({self.gpus_per_summarization_llm} GPUs/LLM), "
            f"{desired_preprocessor_gpu_share} for preprocessors ({self.gpus_per_preprocessor_llm} GPUs/LLM), "
            f"{self.total_finetune_gpus} for finetune,\n"
            f"with {cfg.world.replicas} workers.\n"
        )
        self._log_info("I have adjusted the GPU shares to accommodate these constraints.")
        self._log_info(
            f"Actual GPU share: {total_rc_actor_gpus} for RC actors, {total_actor_gpus} for actors, "
            f"{total_summarization_gpus} for summarization, "
            f"{total_preprocessor_gpus} for preprocessors, {self.total_finetune_gpus} for finetune"
        )
        if self.total_finetune_gpus < 0:
            raise ValueError("Not enough gpus to place all workers")
        if self.total_finetune_gpus == 0:
            logger.warning("No GPUs left for finetune workers. You can still debug other parts of the pipeline.")

        # Weight update group includes all actor LLMs (both RC and regular)
        # Each LLM contributes its TP*PP workers to the NCCL group
        rc_actor_nccl_ranks = self.total_rc_actor_llms * self.gpus_per_rc_actor_llm
        actor_nccl_ranks = self.total_actor_llms * self.gpus_per_actor_llm
        self.weight_update_group_size = 1 + rc_actor_nccl_ranks + actor_nccl_ranks  # +1 for finetune rank 0
        
        self._log_info(
            f"Weight update NCCL group size: {self.weight_update_group_size} "
            f"(1 finetune + {rc_actor_nccl_ranks} RC actor + {actor_nccl_ranks} actor ranks)"
        )

    def _place_pipeline_stages(self, cfg):
        for worker_idx in range(cfg.world.replicas):
            # node = self.get_least_busy_node()
            node = 0
            # Add RC actor if configured
            if cfg.world.get("rc_actor_fraction", 0) > 0:
                self.add_job(kind="rc_actor", replica_idx=worker_idx, node_rank=node, gpus=[], cpu_heavy=True)
            if cfg.world.get("actor_fraction", 0) > 0:
                self.add_job(kind="actor", replica_idx=worker_idx, node_rank=node, gpus=[], cpu_heavy=True)
            if cfg.world.get("finetune_fraction", 0) > 0:
                self.add_job(kind="preprocessor", replica_idx=worker_idx, node_rank=node, gpus=[], cpu_heavy=True)

    def _place_environments(self, cfg):
        for worker_idx in range(cfg.world.env_replicas):
            # node = self.get_least_busy_node()
            node = 0
            envs_at_node = len([job for job in self.job_map[node] if job.kind == "environment"])
            self.add_job(
                kind="environment",
                replica_idx=worker_idx,
                node_rank=node,
                port=cfg.world.environment_start_port + envs_at_node,
                gpus=[],
                cpu_heavy=True,
            )

    def _place_inference_jobs(self, cfg):
        # Place RC actor LLMs (port 8000+)
        for _ in range(cfg.world.replicas):
            for rc_actor_llm_idx in range(self.llms_per_rc_actor):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_rc_actor_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all RC actor LLMs")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_rc_actor_llm)]
                local_idx = min(gpus)
                llm_url = f"http://{self.address_map[node]}:{8000 + local_idx}"
                logger.info(f"Placing RC actor LLM {rc_actor_llm_idx} on node {node} at port {8000 + local_idx} with URL {llm_url}")
                self.add_job(
                    kind="rc_actor_llm",
                    replica_idx=rc_actor_llm_idx,
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    port=8000 + local_idx,
                    url=llm_url,
                )
        
        # Place regular actor LLMs (port 8080+)
        # Note: replica_idx must be offset by total_rc_actor_llms for NCCL group rank calculation
        for _ in range(cfg.world.replicas):
            for actor_llm_idx in range(self.llms_per_actor):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_actor_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all actor LLMs")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_actor_llm)]
                local_idx = min(gpus)
                llm_url = f"http://{self.address_map[node]}:{8080 + local_idx}"
                logger.info(f"Placing actor LLM {actor_llm_idx} on node {node} at port {8080 + local_idx} with URL {llm_url}")
                self.add_job(
                    kind="actor_llm",
                    replica_idx=self.total_rc_actor_llms + actor_llm_idx,  # Offset by RC actor LLMs
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    port=8080 + local_idx,
                    url=llm_url,
                )
        
        # Place summarization LLMs (port 8200+)
        for _ in range(cfg.world.replicas):
            for summarization_llm_idx in range(self.llms_per_summarization):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_summarization_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all summarization LLMs")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_summarization_llm)]
                local_idx = min(gpus)
                summ_url = f"http://{self.address_map[node]}:{8200 + local_idx}"
                logger.info(f"Placing summarization LLM {summarization_llm_idx} on node {node} at port {8200 + local_idx} with URL {summ_url}")
                self.add_job(
                    kind="summarization_llm",
                    replica_idx=summarization_llm_idx,
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    port=8200 + local_idx,
                    url=summ_url,
                )

        # Place preprocessor LLMs (port 8180+)
        for _ in range(cfg.world.replicas):
            for preprocessor_llm_idx in range(self.llms_per_preprocessor):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_preprocessor_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all preprocessor LLMs")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_preprocessor_llm)]
                logger.info(f"Placing preprocessor LLM {preprocessor_llm_idx} on node {node} at port {8180 + local_idx} with URL {ref_url}")
                local_idx = min(gpus)
                ref_url = f"http://{self.address_map[node]}:{8180 + local_idx}"
                self.add_job(
                    kind="preprocessor_llm",
                    replica_idx=preprocessor_llm_idx,
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    port=8180 + local_idx,
                    url=ref_url,
                )

    def get_least_busy_node(self):
        """Get the node with the least number of CPU-heavy jobs."""
        result = 0 
        for node, cpu_heavy_jobs in self.cpu_heavy_jobs.items():
            if cpu_heavy_jobs < self.cpu_heavy_jobs[result]:
                result = node
        return result

    def my_jobs(self) -> list[Job]:
        return self.job_map[self.my_rank]
    
    def get_jobs_on_rank(self, rank: int) -> list[Job]:
        return self.job_map[rank]

    def nodes_with_finetuning(self) -> list[int]:
        return [node for node, jobs in self.job_map.items() if any(job.kind == "finetune" for job in jobs)]

    def my_finetuning_rank(self) -> int:
        return self.nodes_with_finetuning().index(self.my_rank)

    def get_all_jobs(self):
        return [job for jobs in self.job_map.values() for job in jobs]

    def get_rc_actor_urls(self) -> list[str]:
        return [job.url for job in self.get_all_jobs() if job.kind == "rc_actor_llm"]
    
    def get_actor_urls(self) -> list[str]:
        return [job.url for job in self.get_all_jobs() if job.kind == "actor_llm"]
    
    def get_summarization_urls(self) -> list[str]:
        """Get URLs for summarization LLMs, returns empty list if none exist"""
        return [job.url for job in self.get_all_jobs() if job.kind == "summarization_llm"]

    def get_preprocessor_urls(self) -> list[str]:
        return [job.url for job in self.get_all_jobs() if job.kind == "preprocessor_llm"]
