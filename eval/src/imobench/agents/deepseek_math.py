import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override

import regex
from loguru import logger

from imobench.api_client import APIClient
from imobench.agents import BaseAgent, SolverResponse
from imobench.utils import get_substring
from copy import deepcopy
import numpy as np


class DeepSeekMathAgent(BaseAgent):
    """
    An agent for Project Euler problems.
    """


    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.scaffold_config = solver_config
        self.model_config = solver_config["model_config"]

        # create a hash that is unique to this model + relevant scaffold params
        stringify_params = str(self.model_config) + str(self.scaffold_config)
        # time independent hash
        parameter_hash = str(hash(stringify_params.encode('utf-8')) % 10 ** 12)

        self.RUN_ID = self.scaffold_config["run_idx"].format(
            problem_id=self.problem_idx,
            parameter_hash=parameter_hash,
            solver_path=self.solver_config_path if self.solver_config_path is not None else "default",
        )

        # Simple client with no tools
        self.simple_client = self.load_agent_from_path(self.model_config, log=False)
        if self.scaffold_config.get("verifier_model") is not None:
            self.verification_client = self.load_agent_from_path(self.scaffold_config["verifier_model"], log=False)
        else:
            self.verification_client = self.load_agent_from_path(self.model_config, log=False)
        if self.scaffold_config.get("improver_model") is not None:
            self.improver_client = self.load_agent_from_path(self.scaffold_config["improver_model"], log=False)
        else:
            self.improver_client = self.load_agent_from_path(self.model_config, log=False)

        self.pool_size = self.scaffold_config.get("pool_size", 4)
        self.verification_pool_size = self.scaffold_config.get("verification_pool_size", 4)
        self.verifications_per_refinement = self.scaffold_config.get("verifications_per_refinement", 2)
        self.max_iterations = self.scaffold_config.get("max_iterations", 3)
        self.generation_prompt = self.scaffold_config.get("generation_prompt", None)
        self.verifier_prompt = self.scaffold_config.get("verifier_prompt", None)
        self.refinement_prompt = self.scaffold_config.get("refinement_prompt", None)

    def extract_proof(self, text):
        start_pattern = re.compile(r"##\s*solution", re.IGNORECASE)
        end_pattern_strict = re.compile(r"##\s*self\s*evaluation", re.IGNORECASE)
        end_pattern_soft = re.compile(r"self\s*evaluation", re.IGNORECASE)

        start_match = start_pattern.search(text)
        end_match = end_pattern_strict.search(text)

        extracted_proof = ""
        if not end_match:
            end_match = end_pattern_soft.search(text)

        if start_match and end_match:
            if end_match.start() > start_match.end():
                extracted_proof = text[start_match.end():end_match.start()].strip()
            else:
                extracted_proof = text[start_match.end():].strip() 
        elif start_match:
            extracted_proof = text[start_match.end():].strip()
        elif end_match:
            extracted_proof = text[:end_match.start()].strip()
        else:
            extracted_proof = text.strip()
        if len(extracted_proof) == 0:
            extracted_proof = "[Empty proof -> model error should be given 0 score]"
        return extracted_proof
        
    def extract_score(self, text):
        # extract the score in the format \boxed{...}
        score_pattern = re.compile(r"boxed\{\s*(0|0\.5|1)\s*\}")
        score_match = score_pattern.search(text)
        if score_match:
            return float(score_match.group(1))
        else:
            return 0.0

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

        solution_pool = []

        def run_generator():
            """Run a single generator and return its conversation."""
            prompt = self.generation_prompt.format(
                question=stmt,
            )
            convo = [
                {"role": "user", "content": prompt},
            ]
            convo = self._query(self.simple_client, convo)
            return self.extract_proof(convo[-1]["content"]), convo
        
        def run_verifier(proof):
            """Run a single verifier and return its conversation."""
            prompt = self.verifier_prompt.format(
                question=stmt,
                proof=proof,
            )
            convo = [
                {"role": "user", "content": prompt},
            ]
            convo = self._query(self.verification_client, convo)
            try:
                return (convo[-1]["content"], self.extract_score(convo[-1]["content"]), convo)
            except Exception as e:
                logger.error(f"[{self.bi}] Error extracting score: {convo[-1]}")
                raise e
        
        def run_refiner(proof, proof_analyses):
            """Run a single refiner and return its conversation."""
            proof_analysis = ""
            for index, pa in enumerate(proof_analyses):
                proof_analysis += "\n\n### Verification " + str(index + 1) + " ###\n" + pa
            generation_prompt = self.generation_prompt.format(
                question=stmt,
            )
            prompt = self.refinement_prompt.format(
                proof_generation_prompt=generation_prompt,
                proof=proof,
                proof_analysis=proof_analysis,
            )
            convo = [
                {"role": "user", "content": prompt},
            ]
            convo = self._query(self.improver_client, convo)
            return self.extract_proof(convo[-1]["content"]), convo
        
        if self._history_has_step(f"init"):
            logger.debug(f"[{self.bi}] Skipping init step from checkpoint.")
            start = self.get_history_step(f"init")
            solution_pool = start["solutions"]
        else:
            logger.debug(f"[{self.bi}] Starting the initial generation step.")
            # Initial generation
            gen_convos = []
            with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
                futures = [executor.submit(run_generator) for _ in range(self.pool_size)]
                for future in as_completed(futures):
                    gen_convos.append(future.result())
            solution_pool.extend([proof for (proof, _) in gen_convos])

            for idx, convo in enumerate(gen_convos):
                self._add_history(step=f"init_proof_{idx}", timestep=1, conversation=deepcopy(convo[1]))

            self._add_history(step="init", timestep=1, conversation=[], solutions=deepcopy(solution_pool))
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Initial generation step finished.")

        verification_pool = []

        if self._history_has_step(f"initial_verification"):
            logger.debug(f"[{self.bi}] Skipping initial_verification step from checkpoint.")
            start = self.get_history_step(f"initial_verification")
            verification_pool = start["verifications"]
        else:
            logger.debug(f"[{self.bi}] Starting the initial verification step.")
            # Initial verification
            verif_convos = [[] for _ in range(len(solution_pool))]
            with ThreadPoolExecutor(max_workers=self.verification_pool_size) as executor:
                futures = []
                for idx, proof in enumerate(solution_pool):
                    for _ in range(self.verification_pool_size):
                        futures.append((idx, executor.submit(run_verifier, proof)))
                for idx, future in futures:
                    verif_convos[idx].append(future.result())

            for idx, verifications in enumerate(verif_convos):
                    for idx2, convo in enumerate(verifications):
                        self._add_history(step=f"initial_verification_{idx}_{idx2}",
                                        timestep=2, conversation=deepcopy(convo[2]))
            verification_pool.extend([
                [(idx0, idx1) for (idx0, idx1, _) in verif_convos[idx]]
                for idx in range(len(solution_pool))
            ])
            self._add_history(step="initial_verification", timestep=2, conversation=[], 
                              verifications=deepcopy(verification_pool))
            self._save_checkpoint()
            logger.debug(f"[{self.bi}] Initial verification step finished.")
        
        for iteration in range(self.max_iterations):
            # select the pool_size best proofs based on verifications
            scores = [
                np.mean([score for (_, score) in verifs]) if len(verifs) > 0 else 0.0
                for verifs in verification_pool
            ]
            # if any proof has score 1.0, we can stop early
            if max(scores) >= 1.0:
                logger.info(f"[{self.bi}] Stopping early at iteration {iteration} due to perfect proof found.")
                break
            if self._history_has_step(f"iteration_{iteration}_refinement"):
                logger.debug(f"[{self.bi}] Skipping iteration_{iteration}_refinement step from checkpoint.")
                start = self.get_history_step(f"iteration_{iteration}_refinement")
                solution_pool = start["solutions"]
            else:
                best_indices = np.argsort(scores)[-self.pool_size:]
                selected_proofs = [solution_pool[idx] for idx in best_indices]
                selected_analyses = []
                # select the lowest scoring verifications for each selected proof
                for idx in best_indices:
                    verifs = verification_pool[idx]
                    verifs_sorted = sorted(verifs, key=lambda x: x[1])
                    analyses = [verif for (verif, _) in verifs_sorted[:self.verifications_per_refinement]]
                    selected_analyses.append(analyses)
                logger.debug(f"[{self.bi}] Starting iteration {iteration} refinement step.")
                # Refinement
                ref_convos = []
                with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
                    futures = [executor.submit(run_refiner, proof, analyses) 
                               for proof, analyses in zip(selected_proofs, selected_analyses)]
                    for future in as_completed(futures):
                        ref_convos.append(future.result())
                solution_pool.extend([proof for (proof, _) in ref_convos])

                for idx, convo in enumerate(ref_convos):
                    self._add_history(step=f"iteration_{iteration}_refinement_proof_{idx}",
                                        timestep=3 + iteration, conversation=deepcopy(convo[1]))

                self._add_history(step=f"iteration_{iteration}_refinement", timestep=3 + iteration, conversation=[], 
                                  solutions=deepcopy(solution_pool))
                self._save_checkpoint()
                logger.debug(f"[{self.bi}] Iteration {iteration} refinement step finished.")
            
            if self._history_has_step(f"iteration_{iteration}_verification"):
                logger.debug(f"[{self.bi}] Skipping iteration_{iteration}_verification step from checkpoint.")
                start = self.get_history_step(f"iteration_{iteration}_verification")
                verification_pool = start["verifications"]
            else:
                logger.debug(f"[{self.bi}] Starting iteration {iteration} verification step.")
                # Verification, only for the newly added proofs
                new_verif_convos = [[] for _ in range(len(solution_pool) - len(verification_pool))]
                with ThreadPoolExecutor(max_workers=self.verification_pool_size) as executor:
                    futures = []
                    for idx in range(len(verification_pool), len(solution_pool)):
                        proof = solution_pool[idx]
                        for _ in range(self.verification_pool_size):
                            futures.append((idx - len(verification_pool), executor.submit(run_verifier, proof)))
                    for idx, future in futures:
                        new_verif_convos[idx].append(future.result())

                for idx, verifications in enumerate(new_verif_convos):
                    for idx2, convo in enumerate(verifications):
                        self._add_history(step=f"iteration_{iteration}_verification_proof_{idx}_{idx2}",
                                        timestep=4 + iteration, conversation=deepcopy(convo[2]))

                verification_pool.extend([
                    [(idx0, idx1) for (idx0, idx1, _) in new_verif_convos[idx]
                    ] for idx in range(len(new_verif_convos))
                ])
                self._add_history(step=f"iteration_{iteration}_verification", timestep=4 + iteration, conversation=[], 
                                  verifications=deepcopy(verification_pool))
                self._save_checkpoint()
                logger.debug(f"[{self.bi}] Iteration {iteration} verification step finished.")

        # Select best proof overall
        final_scores = [
            np.mean([score for (_, score) in verifs]) if len(verifs) > 0 else 0.0
            for verifs in verification_pool
        ]
        best_index = np.argmax(final_scores)
        best_proof = solution_pool[best_index]
        final_convo = [
            {"role": "user", "content": self.generation_prompt.format(question=stmt)},
            {"role": "assistant", "content": best_proof}
        ]
        logger.info(f"[{self.bi}] Best proof selected with score {final_scores[best_index]:.2f}.")
        return self._end_run(final_convo)