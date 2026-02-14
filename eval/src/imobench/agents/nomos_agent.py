import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override
from copy import deepcopy

import regex
from loguru import logger

from imobench.api_client import APIClient
from imobench.agents import BaseAgent, SolverResponse
from imobench.utils import load_config
import numpy as np

class NomosAgent(BaseAgent):
    """
    An agent for Nomos strategy:
    """
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.model_config = load_config(solver_config["model_config"])
        self.scaffold_config = solver_config

        # create a hash that is unique to this model + relevant scaffold params
        stringify_params = str(self.model_config) + str(self.scaffold_config)
        # time independent hash
        parameter_hash = str(hash(stringify_params.encode('utf-8')) % 10 ** 12)

        self.RUN_ID = self.scaffold_config["run_idx"].format(
            problem_id=self.problem_idx,
            parameter_hash=parameter_hash,
            solver_path=self.solver_config_path if self.solver_config_path is not None else "default"
        )

        # Simple client with no tools
        simple_client_args = copy.deepcopy(self.model_config)
        if "human_readable_id" in simple_client_args:
            del simple_client_args["human_readable_id"]
        if "date" in simple_client_args:
            del simple_client_args["date"]
        self.client = self.load_agent_from_config(simple_client_args, log=False)

        self.n_solutions_to_generate = self.scaffold_config.get("n_solutions_to_generate", 8)
        self.max_kept_solutions = self.scaffold_config.get("max_kept_solutions", 8)
        self.consolidation_prompt = self.scaffold_config.get("consolidation_prompt", "")
        self.pairwise_prompt = self.scaffold_config.get("pairwise_prompt", "")
        self.score_prompt = self.scaffold_config.get("score_prompt", "")
        self.target_perfect_scores = self.scaffold_config.get("target_perfect_scores", 4)
        self.solutions = []

    def solve_stage(self, stmt: str):
        prompt = [{"role": "user", "content": stmt}]
        conversations = self._query_multiple(self.client, [prompt] * self.n_solutions_to_generate)
        for conversation in conversations:
            assistant_msg = conversation[-1]["content"]
            if len(assistant_msg.strip()) == 0:
                assistant_msg = "The model returned an empty response. This proof is therefore invalid and should be given a score of 0."
            self.solutions.append({
                "solution": assistant_msg,
            })

    def _extract_score(self, judge_response: str):
        """Extract score from judge response."""
        # Look for boxed{3.0} or boxed{2} or ... patterns
        match = re.search(r'boxed\{\s*(\d+(\.\d+)?)\s*\}', judge_response)
        if match:
            return float(match.group(1))
        return None

    def _extract_keep_list(self, consolidation_response: str):
        """Extract keep list from consolidation response."""
        match = re.search(r'<keep>\s*\[([\d,\s]+)\]\s*</keep>', consolidation_response)
        if match:
            nums = match.group(1).split(',')
            return [int(n.strip()) for n in nums if n.strip().isdigit()]
        return [i for i in range(1, len(self.solutions)+1)]  # Keep all if not found
    
    def _extract_verdict(self, pairwise_response: str):
        """Extract verdict from pairwise response."""
        match = re.search(r'<verdict>\s*(1|2|tie)\s*</verdict>', pairwise_response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        last_possible = pairwise_response.lower().split("verdict")[-1]
        if "1" in last_possible and not "2" in last_possible:
            return '1'
        if "2" in last_possible and not "1" in last_possible:
            return '2'
        return "tie"

    def score_stage(self, stmt: str):
        prompts = [
           [{"role": "user", "content": self.score_prompt.format(
                problem=stmt,
                answer=solution["solution"]
            )}] for solution in self.solutions
        ]
        conversations = self._query_multiple(self.client, prompts)
        for i, conversation in enumerate(conversations):
            assistant_msg = conversation[-1]["content"]
            score = self._extract_score(assistant_msg)
            self.solutions[i]["score"] = score if score is not None else 0
            self.solutions[i]["judge_feedback"] = assistant_msg
            self.solutions[i]["is_perfect"] = (score == 7)

    def consolidate_stage(self, stmt: str):
        submissions_text = ""
        for i, sub in enumerate(self.solutions, 1):
            submissions_text += f"\n### Submission {i}\n\n{sub['solution']}\n"
            if sub["judge_feedback"]:
                submissions_text += f"\n**Judge Feedback:** {sub['judge_feedback']}\n"
        prompt = [{"role": "user", "content": self.consolidation_prompt.format(
            problem=stmt,
            submissions=submissions_text
        )}]
        
        conversation = self._query(self.client, prompt)
        assistant_msg = conversation[-1]["content"]
        keep_list = self._extract_keep_list(assistant_msg)
        self.solutions = [self.solutions[i-1] for i in keep_list if 1 <= i <= len(self.solutions)]
        for sol in self.solutions:
            sol["consolidation_feedback"] = assistant_msg
    
    def filter_stage(self):
        # Keep only top scored solutions up to max_kept_solutions
        perfect_solutions = [s for s in self.solutions if s["is_perfect"]]
        if len(perfect_solutions) >= self.target_perfect_scores:
            self.solutions = perfect_solutions[:self.max_kept_solutions]
        else:
            self.solutions.sort(key=lambda x: x["score"], reverse=True)
            self.solutions = self.solutions[:self.max_kept_solutions]

    def tournament_stage(self, stmt: str):
        # randomly sort solutions for pairwise comparisons
        while len(self.solutions) > 1:
            np.random.shuffle(self.solutions)
            next_round = []
            for i in range(0, len(self.solutions), 2):
                if i + 1 >= len(self.solutions):
                    continue
                prompt = [{"role": "user", "content": self.pairwise_prompt.format(
                    problem=stmt,
                    submission1=self.solutions[i]["solution"],
                    submission2=self.solutions[i+1]["solution"]
                )}]
                next_round.append((i, i+1, prompt))
            conversations = self._query_multiple(self.client, [item[2] for item in next_round])
            kept_solutions = [self.solutions[-1]] if len(self.solutions) % 2 == 1 else []
            for idx, conversation in enumerate(conversations):
                assistant_msg = conversation[-1]["content"]
                verdict = self._extract_verdict(assistant_msg)
                i, j, _ = next_round[idx]
                if verdict == "tie":
                    random_choice = np.random.choice([0, 1])
                    if random_choice == 0:
                        keep_index = i
                    else:
                        keep_index = j
                if verdict == '1':
                    keep_index = i
                elif verdict == '2':
                    keep_index = j
                kept_solutions.append(self.solutions[keep_index])
                if "pairwise_feedback" not in self.solutions[keep_index]:
                    self.solutions[keep_index]["pairwise_feedback"] = []
                self.solutions[keep_index]["pairwise_feedback"].append({
                    "opponent_index": j if keep_index == i else i,
                    "verdict": verdict,
                    "judge_response": assistant_msg
                })
            self.solutions = kept_solutions[:]


    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object (see BaseAgent._end_run).
        """
        self._start_run(stmt)
        self._load_checkpoint_if_exists()  # Will prefill history with some steps, those we can skip
        self.solutions = []

        logger.info(f"[{self.bi}] Starting NomosAgent run for problem: {self.stmt[:50]}...")
        if self._history_has_step("solve_stage"):
            logger.info(f"[{self.bi}] Resuming from checkpoint at solve_stage.")
            self.solutions = self.get_history_step("solve_stage").get("solutions", [])
        else:
            self.solve_stage(stmt)
            self._add_history(
                step="solve_stage",
                timestep=0,
                conversation=[],
                solutions=deepcopy(self.solutions)
            )
            self._save_checkpoint()
        logger.info(f"[{self.bi}] Generated {len(self.solutions)} solutions.")
        if self._history_has_step("score_stage"):
            logger.info(f"[{self.bi}] Resuming from checkpoint at score_stage.")
            self.solutions = self.get_history_step("score_stage").get("solutions", [])
        else:
            self.score_stage(stmt)
            self._add_history(
                step="score_stage",
                timestep=1,
                conversation=[],
                solutions=deepcopy(self.solutions)
            )
            self._save_checkpoint()
        logger.info(f"[{self.bi}] Scored all solutions.")
        if self._history_has_step("consolidate_stage"):
            logger.info(f"[{self.bi}] Resuming from checkpoint at consolidate_stage.")
            self.solutions = self.get_history_step("consolidate_stage").get("solutions", [])
        else:
            self.filter_stage()
            self.consolidate_stage(stmt)
            self._add_history(
                step="consolidate_stage",
                timestep=2,
                conversation=[],
                solutions=deepcopy(self.solutions)
            )
            self._save_checkpoint()
        logger.info(f"[{self.bi}] Consolidated solutions, {len(self.solutions)} remain after consolidation.")
        if self._history_has_step("tournament_stage"):
            logger.info(f"[{self.bi}] Resuming from checkpoint at tournament_stage.")
            self.solutions = self.get_history_step("tournament_stage").get("solutions", [])
        else:
            self.tournament_stage(stmt)
            self._add_history(
                step="tournament_stage",
                timestep=3,
                conversation=[],
                solutions=deepcopy(self.solutions)
            )
            self._save_checkpoint()
        logger.info(f"[{self.bi}] Tournament complete, {len(self.solutions)} winning solutions.")

        return self._end_run([
            {
                "role": "user",
                "content": stmt
            },
            {
                "role": "assistant",
                "content": self.solutions[0]["solution"] if self.solutions else "No solution generated."
            }
        ])
