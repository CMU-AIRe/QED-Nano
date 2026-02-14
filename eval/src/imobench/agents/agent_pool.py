"""This module defines the base Agent class for solving math problems."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override
import yaml
import os

from loguru import logger
from tqdm import tqdm

from imobench.agents import PureModelSolver, BaseSolver, SelfcheckAgent, ReasoningCacheAgent, MathAgent, StaticMathAgent, DeepSeekMathAgent, NomosAgent, RSAAgent
from imobench.utils import load_config

class AgentPool(BaseSolver):
    """
    A solver that manages a pool of agents to solve problems.
    """

    AGENT_CLASSES = {"selfcheck": SelfcheckAgent, 
                     "math_agent": MathAgent, "static_math_agent": StaticMathAgent, 
                     "deepseek_agent": DeepSeekMathAgent, "nomos": NomosAgent, 
                     "rc": ReasoningCacheAgent, "rsa": RSAAgent}  # Add other agent classes as needed

    def __init__(self, solver_config):
        super().__init__(solver_config)

        # AgentPool handles multithreading, individual agents use
        # model_config["concurrent_requests"] internally, but usually send 1 query at a time to APIClient.
        self.scaffold_config = self.solver_config
        self.n_threads = self.scaffold_config.get("n_threads", 1)
        self.AGENT_CLASS = AgentPool.AGENT_CLASSES[self.scaffold_config["scaffold_name"]]
        self.path = None

    @staticmethod
    def load_agent_from_path(solver_config_path, log=True):
        """
        Loads an agent from a solver configuration file.

        Args:
            solver_config_path (str): Path to the solver configuration file.
        Returns:
            AgentPool: An instance of the AgentPool class.
        """
        config = load_config(solver_config_path)
        agent_pool = AgentPool.load_agent_from_config(config, log=log)
        agent_pool.path = solver_config_path
        return agent_pool
    
    def set_path(self, path):
        self.path = path

    @staticmethod
    def load_agent_from_config(solver_config, log=True):
        """
        Loads an agent from a solver configuration dictionary.

        Args:
            solver_config (dict): A solver configuration dictionary.
        Returns:
            AgentPool: An instance of the AgentPool class.
        """
        if not solver_config.get("type") == "agent":
            agent = PureModelSolver(solver_config)
        else:
            scaffold_config_path = os.path.join("configs", solver_config["scaffold_config"] + ".yaml")
            scaffold_config = yaml.safe_load(open(scaffold_config_path, "r"))
            for param in solver_config:
                scaffold_config[param] = solver_config[param]
            agent = AgentPool(scaffold_config)
        agent.log = log
        return agent

    def _run_agent(self, batch_idx: int, problem_idx: int, run_idx: int, stmt: str):
        agent = self.AGENT_CLASS(
            batch_idx=batch_idx,
            problem_idx=problem_idx,
            run_idx=run_idx,
            solver_config=self.solver_config,
            log=self.log,
            solver_config_path=self.path
        )
        return agent.solve(stmt)

    @override
    def solve_batch(self, stmt_batch: list[tuple[str, Any]], 
                    batch_idx_to_problem_idx: dict[int, int], 
                    batch_idx_to_run_idx: dict[int, int], no_tqdm=False):
        """
        Solves a batch of problems. Handles multithreading, launching one Agent per problem.

        Args:
            stmt_batch (list[tuple[str, Any]]): A batch of problem statements as (text, image) pairs.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the batch_index, the conversation array, detailed cost, and history for each problem.
        """
        logger.info(f"Starting agents with n_threads={self.n_threads} for batch of size {len(stmt_batch)}.")
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for batch_idx, stmt in enumerate(stmt_batch):
                problem_idx = batch_idx_to_problem_idx[batch_idx]
                if isinstance(stmt, list):
                    full_stmt = ""
                    for part in stmt:
                        full_stmt += part["content"] + "\n"
                    stmt = full_stmt.strip()
                run_idx = batch_idx_to_run_idx[batch_idx]
                futures.append(executor.submit(self._run_agent, batch_idx, problem_idx, run_idx, stmt))
            iterator = as_completed(futures)
            if not no_tqdm:
                iterator = tqdm(iterator, total=len(futures))
            for future in iterator:
                solver_response = future.result()
                logger.info(f"[{solver_response.idx}] Agent completed solving problem.")
                yield solver_response