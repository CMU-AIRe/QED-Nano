import json
import threading
import time
from typing import Any
import os

from loguru import logger

from imobench.agents import SolverResponse, BaseSolver

class BaseAgent(BaseSolver):
    """
    An abstract agent that solves a single math problem instance by using one or more APIClients.

    batch_idx: the index of this problem in the batch handled by the AgentPool.
    solver_config: the full solver config, including model_config and scaffold_config.
    default_prompt_template: the prompt template
        (the "instruction" from the competition config + {problem} template)
    default_api_client_args: the kwargs for the APIClient constructor, i.e., all kwargs
        stated in model_config + "tools" and "max_tool_calls" from the competition config.
        Tools is a list of pairs (function, tool_spec) where function is None for responses API
        The agent can override these when creating its own APIClient(s).
    """

    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        self.batch_idx = batch_idx
        self.problem_idx = problem_idx
        self.run_idx = run_idx
        self.bi = batch_idx  # short alias
        self.solver_config = solver_config
        self._lock = threading.Lock()
        self.has_finished = False
        self.RUN_ID_FULL = None
        self.log = log
        self.solver_config_path = solver_config_path
        
        self.checkpoint_base_dir = "agent_checkpoints"
        self._model_name = self._resolve_model_name()
        self._model_revision = self._resolve_model_revision()
        self._checkpoint_run_id = None

    @staticmethod
    def load_agent_from_config(solver_config, log=True):
        from imobench.agents.agent_pool import AgentPool
        return AgentPool.load_agent_from_config(solver_config, log=log)

    @staticmethod
    def load_agent_from_path(solver_config_path, log=True):
        from imobench.agents.agent_pool import AgentPool
        return AgentPool.load_agent_from_path(solver_config_path, log=log)

    def _start_run(self, stmt: str):
        """
        Starts the run:
            - Resets the state to be returned.
            - Sets the 1st of 2 entries in the final conversation.
        """
        self.stmt = stmt
        logger.debug(f"[{self.bi}] Starting agent run for problem: {stmt[:50]}...")
        self.conversation = [{"role": "user", "content": stmt}, {"role": "assistant", "content": "TODO"}]
        self.detailed_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
        self.history = []

    def _query(self, client: BaseSolver, query: list[dict[str, Any]], ignore_tool_calls: bool = False):
        """
        A wrapper that runs a single query (conversation) via the given APIClient and updates the cost state.
        Queries should add user/developer messages in clean format or reuse message blocks from same client.
        """
        start_time = time.time()
        ret = list(client.solve_batch([query], {0: 0}, {0: 0}, no_tqdm=True))[0]
        conversation = ret.conversation
        detailed_cost = ret.detailed_cost
        with self._lock:
            self.detailed_cost["cost"] += detailed_cost["cost"]
            self.detailed_cost["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost["output_tokens"] += detailed_cost["output_tokens"]
            self.detailed_cost["time"] += time.time() - start_time
        return conversation
    
    def _query_multiple(self, client: BaseSolver, queries: list[list[dict[str, Any]]], ignore_tool_calls: bool = False):
        """
        A wrapper that runs multiple queries (conversations) via the given APIClient and updates the cost state.
        Queries should add user/developer messages in clean format or reuse message blocks from same client.
        """
        start_time = time.time()
        ret = client.solve_batch(queries, {i: i for i in range(len(queries))}, 
                                 {i: 0 for i in range(len(queries))}, no_tqdm=True)
        conversations = [None] * len(queries)
        with self._lock:
            for r in ret:
                conversations[r.idx] = r.conversation
                self.detailed_cost["cost"] += r.detailed_cost["cost"]
                self.detailed_cost["input_tokens"] += r.detailed_cost["input_tokens"]
                self.detailed_cost["output_tokens"] += r.detailed_cost["output_tokens"]
            self.detailed_cost["time"] += time.time() - start_time
        return conversations

    def _add_history(self, step: str, timestep: int, conversation: Any, **kwargs):
        """
        Adds an entry to the history.
        """
        entry = {
            "step": step,
            "timestep": timestep,
            "messages": conversation,
        }
        entry.update(kwargs)
        with self._lock:
            self.history.append(entry)

    def _end_run(self, final_response) -> SolverResponse:
        """
        Ends the run:
            - Sets the 2nd of 2 entries in the final conversation.
            - Returns a SolverResponse object with batch index.
        """
        logger.debug(f"[{self.bi}] Ending agent run for problem: {self.stmt[:50]}...")
        if isinstance(final_response, str):
            self.conversation[1]["content"] = final_response
        else:
            self.conversation = final_response

        if len(self.conversation[-1]["content"]) == 0 or self.conversation[-1]["role"] != "assistant":
            logger.warning(f"[{self.bi}] Final conversation appears broken.")
            self.conversation.append({"role": "assistant", "content": "The model did not return an answer."})

        self.has_finished = True
        self._save_checkpoint()
        return SolverResponse(
            idx=self.batch_idx, conversation=self.conversation, detailed_cost=self.detailed_cost, history=self.history
        )

    def _resolve_model_revision(self):
        if not isinstance(self.solver_config, dict):
            return None
        revision = self.solver_config.get("model_revision")
        model_config = self.solver_config.get("model_config")
        if not revision and isinstance(model_config, dict):
            revision = model_config.get("model_revision")
        if not revision and isinstance(model_config, str):
            try:
                from imobench.utils import load_config
                config = load_config(model_config)
                revision = config.get("model_revision")
            except Exception:
                return None
        if not revision or revision == "main":
            return None
        return revision

    def _resolve_model_name(self):
        if not isinstance(self.solver_config, dict):
            return None
        model = self.solver_config.get("model")
        model_config = self.solver_config.get("model_config")
        if not model and isinstance(model_config, dict):
            model = model_config.get("model")
        if not model and isinstance(model_config, str):
            try:
                from imobench.utils import load_config
                config = load_config(model_config)
                model = config.get("model")
            except Exception:
                return None
        return model

    def _sanitize_revision(self, revision: str) -> str:
        revision = revision.replace("/", "-")
        return "".join(ch for ch in revision if ch.isalnum() or ch in "-_.")

    def _sanitize_model_name(self, model: str) -> str:
        model = model.replace("/", "--")
        return "".join(ch for ch in model if ch.isalnum() or ch in "-_.")

    def _build_checkpoint_run_id(self, include_model: bool, include_revision: bool):
        if self.RUN_ID is None:
            return None
        model = self._model_name if include_model else None
        revision = self._model_revision if include_revision else None
        if model:
            model = self._sanitize_model_name(model)
            if not model:
                model = None
        if revision:
            revision = self._sanitize_revision(revision)
            if not revision:
                revision = None
        if not model and not revision:
            return self.RUN_ID
        insert = ""
        if model:
            insert += f"_{model}"
        if revision:
            insert += f"--{revision}"
        problem_token = f"_{self.problem_idx}_"
        if problem_token in self.RUN_ID:
            return self.RUN_ID.replace(problem_token, f"{insert}{problem_token}", 1)
        path_parts = self.RUN_ID.split("/")
        last_part = path_parts[-1]
        if "_" in last_part:
            prefix, rest = last_part.split("_", 1)
            path_parts[-1] = f"{prefix}{insert}_{rest}"
            return "/".join(path_parts)
        return f"{self.RUN_ID}{insert}"

    def _get_checkpoint_run_id(self):
        if self._checkpoint_run_id is not None:
            return self._checkpoint_run_id
        if self.RUN_ID is None:
            return None
        self._checkpoint_run_id = self._build_checkpoint_run_id(
            include_model=True, include_revision=True
        )
        return self._checkpoint_run_id

    def _find_full_run_id(self):
        if self.RUN_ID is None:
            self.RUN_ID = f"agent_b{self.batch_idx}_p{self.problem_idx}"
        if self.RUN_ID_FULL is not None:
            return
        
        base_run_id = self._get_checkpoint_run_id()
        indices_to_go = self.run_idx + 1
        current_index = 0
        while indices_to_go > 0:
            run_id = base_run_id + f"_r{current_index}"
            checkpoint_path = f"{self.checkpoint_base_dir}/{run_id}.json"
            if not os.path.exists(checkpoint_path):
                indices_to_go -= 1
            else:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                    if not checkpoint.get("has_finished", True):
                        indices_to_go -= 1
            current_index += 1
        self.RUN_ID_FULL = base_run_id + f"_r{current_index - 1}"

    def _load_checkpoint_if_exists(self) -> None:
        """
        Loads a checkpoint of the current detailed cost and history from a file.
        """
        if not self.log:
            return
        if self.RUN_ID_FULL is None:
            self._find_full_run_id()

        checkpoint_path = f"{self.checkpoint_base_dir}/{self.RUN_ID_FULL}.json"
        os.makedirs(self.checkpoint_base_dir, exist_ok=True)
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                self.detailed_cost = checkpoint["detailed_cost"]
                self.history = checkpoint["history"]
                log = f"[{self.bi}] Loaded checkpoint from {checkpoint_path}! Will skip the following steps:\n"
                for entry in self.history:
                    log += f"    - Step {entry['step']} at timestep {entry['timestep']}\n"
                logger.info(log)
        except FileNotFoundError:
            logger.info(f"[{self.bi}] No checkpoint found at {checkpoint_path}, starting fresh.")

    def _save_checkpoint(self) -> None:
        """
        Saves a checkpoint of the current detailed cost and history to a file.
        """
        if not self.log:
            return
        if self.RUN_ID_FULL is None:
            self._find_full_run_id()
        checkpoint_path = f"{self.checkpoint_base_dir}/{self.RUN_ID_FULL}.json"
        logger.info(f"[{self.bi}] Saving checkpoint to {checkpoint_path}.")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "detailed_cost": self.detailed_cost,
                    "history": self.history,
                    "has_finished": self.has_finished
                },
                f,
                indent=4,
            )

    def _history_has_step(self, step: str) -> bool:
        """
        Checks if a step exists in history.
        """

        for entry in self.history:
            if entry["step"] == step:
                return True
        return False

    def get_history_step(self, step: str) -> dict[str, Any]:
        """
        Retrieves a history entry by step name.
        """

        for entry in self.history:
            if entry["step"] == step:
                return entry
        raise ValueError(f"No history entry found for step {step}.")

    def _get_convo_from_history(self, step: str) -> dict[str, Any]:
        """
        Retrieves a conversation from history by step name.
        """

        for entry in self.history:
            if entry["step"] == step:
                return entry["messages"]
        raise ValueError(f"No history entry found for step {step}.")

    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object containing:
             - index: set to 0 here
             - conversation: the conversation array, for agents it should just have 2 blocks: "user" and "assistant"
             - detailed_cost: agent must report detailed cost info: cost, in/out tokens, time
             - history: a list of steps, where each step corresponds to one conversation:
                 - "step": unique string id
                 - "timestep": the time at which this step happened (for visualization)
                 - "messages": the full conversation in this step
                 - any extra debug keys
                 The agent decides what to report as the final response in the conversation.
        """
        raise NotImplementedError("Subclasses should implement this method.")
