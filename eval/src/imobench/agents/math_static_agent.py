import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Any, override

import time
from loguru import logger
import yaml

from imobench.agents import BaseAgent, SolverResponse
from imobench.utils import load_config
from imobench.agents.math_core import *
from copy import deepcopy



class StaticMathAgent(BaseAgent):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.model_config = load_config(solver_config["model_config"])
        self.scaffold_config = solver_config
        self.core_components = yaml.safe_load(open(self.scaffold_config["core_components"], 'r'))
        self.log_index = f"StaticMathAgent-P{self.problem_idx}-R{self.run_idx}"

        # create a hash that is unique to this model + relevant scaffold params
        stringify_params = str(self.model_config) + str(self.scaffold_config) + str(self.core_components)
        # time independent hash
        parameter_hash = str(hash(stringify_params.encode('utf-8')) % 10 ** 12)

        self.RUN_ID = self.scaffold_config["run_idx"].format(
            problem_id=self.problem_idx,
            parameter_hash=parameter_hash,
            run_idx=run_idx,
            solver_path=self.solver_config_path if self.solver_config_path is not None else "default"
        )
        set_prompts(self.core_components)
        self.id_manager = IDManager()


        args = copy.deepcopy(self.model_config)
        self.clients = [
            self.load_agent_from_config(args, log=False) for _ in range(self.scaffold_config["max_concurrent_calls"])
        ]
        self.current_step = 0
    
    def _caller(self, component, index=0):
        query = component.prepare_query(self.problem_statement)
        response = self._query(self.clients[index], query)
        component.process_response(response, self.id_manager)
        self._add_history(
            step=str(component),
            timestep=self.current_step,
            conversation=deepcopy(response)
        )
        logger.debug(f"[{self.log_index}] Completed component {component} at step {self.current_step}")
        self._save_checkpoint()
        self.current_step += 1
    
    @override
    def solve(self, stmt: str) -> SolverResponse:
        self.problem_statement = stmt
        self._start_run(stmt)
        logger.debug(f"[{self.log_index}] MathAgent starting solve for problem_idx={self.problem_idx}, run_idx={self.run_idx}")

        node = self.execute(self.create_agent())

        return self._end_run(node.response_text if node.response_text is not None else "No solution submitted by the agent.")
    
    def execute(self, node: CoreComponent):
        max_workers = self.scaffold_config["max_concurrent_calls"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}          # future -> (node, index)
            in_flight_nodes = set()
            completed = set()
            busy_indices = set()

            while True:
                if node.is_ready() and not futures:
                    break

                all_necessary_nodes = self.get_necessary_nodes(node)
                ready_nodes = self.get_ready_nodes(all_necessary_nodes)

                nodes_to_process = ready_nodes - completed - in_flight_nodes

                while nodes_to_process and len(busy_indices) < max_workers:
                    free_indices = set(range(max_workers)) - busy_indices
                    index = free_indices.pop()

                    n = nodes_to_process.pop()
                    busy_indices.add(index)

                    future = executor.submit(self._caller, n, index)
                    futures[future] = (n, index)
                    in_flight_nodes.add(n)

                if not futures:
                    if node.is_ready():
                        break
                    raise RuntimeError(
                        "No runnable tasks and root node not ready. Possible dependency deadlock."
                    )

                # Wait for at least one task to complete
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    n, index = futures.pop(future)
                    busy_indices.remove(index)
                    in_flight_nodes.remove(n)

                    try:
                        future.result()
                        completed.add(n)
                    except Exception as e:
                        logger.error(f"Error processing node {n}: {e}")
                        # Optional: cancel remaining futures to stop background work
                        for f in futures:
                            f.cancel()
                        raise

        return node

    def get_necessary_nodes(self, node : CoreComponent):
        all_nodes = {node}
        for input_node in node.get_input_nodes():
            necessary_nodes = self.get_necessary_nodes(input_node)
            all_nodes.update(necessary_nodes)
        return all_nodes
    
    def get_ready_nodes(self, nodes : set[CoreComponent]):
        ready_nodes = set()
        for node in nodes:
            if node.can_start():
                ready_nodes.add(node)
        return ready_nodes

    def create_agent(self):
        initial_approaches = []
        if self.scaffold_config["n_approach_solutions"] > 0:
            approaches = DetermineApproaches(self.scaffold_config["n_approach_solutions"])
            initial_approaches += list(approaches)
        if self.scaffold_config["use_no_approach"]:
            initial_approaches += [None] * self.scaffold_config["n_no_approach_solutions"]

        initial_solutions = []
        for approach in initial_approaches:
            solutions_approach = []
            for _ in range(self.scaffold_config["n_solutions_per_approach"]):
                if approach is not None:  
                    solution = ApproachSolver(approach)
                else:
                    solution = Solver()
                for _ in range(self.scaffold_config["n_verifications_per_solution"]):
                    verifier = VerifySolution(solution)
                    solution = ImproveSolution(solution, verifier)
                solutions_approach.append(solution)

            if len(solutions_approach) == 1:
                initial_solutions.append(solutions_approach[0])
            elif self.scaffold_config["merge_solutions"]:
                merged_solution = Merger(solutions_approach)
                initial_solutions.append(merged_solution)
            else:
                selector = Selector(solutions_approach)
                initial_solutions.append(selector)      
        if len(initial_solutions) == 1:
            return initial_solutions[0]
        else:
            return Selector(initial_solutions)