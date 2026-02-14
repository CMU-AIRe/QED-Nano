from typing import Any, override

from loguru import logger

from imobench.agents import BaseAgent, SolverResponse
from imobench.utils import load_config
from copy import deepcopy

class ReasoningCacheAgent(BaseAgent):
    """
    An agent that solves problems through a series of
    self-correction and verification steps.
    """

    # TODO: Try with a thinking model to make sure CoT flows make sense.
    # (api_client should always split CoT into a separate message, which will end up in history but is
    #  never sent as a request to the model)

    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.model_config = load_config(solver_config["model_config"])
        self.scaffold_config = solver_config
        if self.scaffold_config.get("summarizer_model") is not None:
            self.summary_config = load_config(self.scaffold_config["summarizer_model"])
        else:
            self.summary_config = load_config(solver_config["model_config"])

        stringify_params = str(self.model_config) + str(self.scaffold_config)
        # time independent hash
        parameter_hash = str(hash(stringify_params.encode('utf-8')) % 10 ** 12)
        self.RUN_ID = self.scaffold_config["run_idx"].format(
            problem_id=self.problem_idx,
            run_idx=run_idx,
            parameter_hash=parameter_hash,
            solver_path=self.solver_config_path if self.solver_config_path is not None else "default"
        )
        # Clients
        self.model = self.load_agent_from_config(self.model_config, log=False)
        self.summarizer = self.load_agent_from_config(self.summary_config, log=False)
        self.n_iterations = self.scaffold_config.get("n_iterations", 3)
        self.summarization_prompt = self.scaffold_config.get("summarization_prompt", "")
        self.solve_prompt = self.scaffold_config.get("solve_prompt", "")

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object (see BaseAgent._end_run).
        """
        # Start run and find the first solution with self improvement.
        self._start_run(stmt)
        self._load_checkpoint_if_exists()
        current_summary = "[No previous summary]"
        current_reasoning = ""
        final_answer_content = ""

        for i in range(self.n_iterations):
            logger.info(f"[Run {self.run_idx}][Problem {self.problem_idx}] RC Iteration {i+1}/{self.n_iterations}.")

            if self._history_has_step(f"reasoning_{i}"):
                # Load from history
                reasoning_entry = self.get_history_step(f"reasoning_{i}")
                response = reasoning_entry["messages"]
            else:
                solve_prompt_filled = self.solve_prompt.format(
                    curr_summary=current_summary,
                    problem=self.stmt,
                )
                response = self._query(self.model, [{"role": "user", "content": solve_prompt_filled}])
                self._add_history(
                    step=f"reasoning_{i}",
                    timestep=i,
                    conversation=deepcopy(response)
                )
                self._save_checkpoint()

            if response[-1].get("type") != "cot" and response[-2].get("type") == "cot": # there is CoT
                current_reasoning = response[-2]["content"]
            elif response[-1].get("type") == "cot": # last message is CoT
                current_reasoning = response[-1]["content"]
            else: # there is no CoT
                current_reasoning = response[-1]["content"]

            # Track the latest non-CoT assistant content (may be empty).
            for msg in reversed(response):
                if msg.get("role") == "assistant" and msg.get("type") != "cot" and "content" in msg:
                    final_answer_content = msg["content"]
                    break
            
            if i == self.n_iterations - 1:
                # Final iteration, we are done
                break
            if self._history_has_step(f"summarization_{i}"):
                # Load from history
                summary_entry = self.get_history_step(f"summarization_{i}")
                response = summary_entry["messages"]
            else:
                summary_prompt_filled = self.summarization_prompt.format(
                    problem=self.stmt,
                    existing_summary=current_summary,
                    reasoning=current_reasoning,
                )

                response = self._query(self.summarizer, [{"role": "user", "content": summary_prompt_filled}])
                self._add_history(
                    step=f"summarization_{i}",
                    timestep=i,
                    conversation=deepcopy(response)
                )
                self._save_checkpoint()
            current_summary = response[-1]["content"]
        
        return self._end_run(final_answer_content)
