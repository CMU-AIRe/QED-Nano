import copy
import re
from typing import Any, override

import regex
from loguru import logger
import yaml

from imobench.agents import BaseAgent, SolverResponse
from imobench.utils import load_config
from imobench.agents.math_core import *


class MathAgent(BaseAgent):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.model_config = load_config(solver_config["model_config"])
        self.scaffold_config = solver_config
        self.core_components = yaml.safe_load(open(self.scaffold_config["core_components"], 'r'))
        self.total_calls_left = self.scaffold_config.get("max_calls", 10)
        self.max_calls = self.scaffold_config.get("max_calls", 10)
        self.log_index = f"MathAgent-P{self.problem_idx}-R{self.run_idx}"

        stringify_params = str(self.model_config) + str(self.scaffold_config) + str(self.core_components)
        parameter_hash = str(hash(stringify_params.encode('utf-8')) % 10 ** 12)
        self.RUN_ID = self.scaffold_config["run_idx"].format(
            problem_id=self.problem_idx,
            parameter_hash=parameter_hash,
            run_idx=run_idx,
            solver_path=self.solver_config_path if self.solver_config_path is not None else "default"
        )

        self.all_names = {
            "submit": self.submit_solution,
            "verifier": self.verify_solution,
            "improver": self.improve_solution,
            "selector": self.select_best_solution,
            "merger": self.merge_solutions,
            "solver": self.solve_problem,
            "approach_solver": self.solve_problem,
            "determine_approaches": self.determine_approaches,
        }
        set_prompts(self.core_components)

        self.id_manager = IDManager()

        self.core_components["submit"] = self.scaffold_config["submit"]

        args = copy.deepcopy(self.model_config)
        tool_specs = self.generate_tool_specs()
        args["max_tool_calls"] = {
            self.core_components[name]["name"]: self.scaffold_config.get("max_calls", 10) * 10 
            for name in self.all_names
        }
        args["include_max_tool_calls"] = False
        
        args["tools"] = [
            (self.all_names[name], tool_specs[self.core_components[name]["name"]]) for name in self.all_names
        ]
        self.client = self.load_agent_from_config(args, log=False)
        args_no_tools = copy.deepcopy(self.model_config)
        if "tools" in args_no_tools:
            del args_no_tools["tools"]
        self.client_no_tools = self.load_agent_from_config(args_no_tools, log=False)

        self.solutions = dict()
        self.verifications = dict()
        self.approaches = dict()
        self.solution_id_to_approach_id = dict()
        self.problem_statement = None
        self.output = None

    def generate_tool_specs(self):
        tool_specs = {}
        for name in self.all_names:
            tool_name = self.core_components[name].get("name")
            parameters = {
                "properties": self.core_components[name].get("properties", {}),
                "required": list(self.core_components[name].get("properties", {}).keys()),
                "type": "object",
            }
            tool_specs[tool_name] = {
                "function": {
                    "description": self.core_components[name].get("description", ""),
                    "parameters": parameters,
                    "name": tool_name,
                },
                "type": "function"
            }
        return tool_specs
    
    def _caller(self, component):
        tool_calls_left, message = self.check_tool_calls_left()
        if not tool_calls_left:
            return message
        query = component.prepare_query(self.problem_statement)
        response = self._query(self.client_no_tools, query)
        component.process_response(response, self.id_manager)
        self._add_history(
            step=str(component),
            timestep=self.max_calls - self.total_calls_left + 1,
            conversation=response
        )
        self.total_calls_left -= 1
        return component.format_response(calls_left=self.total_calls_left)
    
    @override
    def solve(self, stmt: str) -> SolverResponse:
        self.problem_statement = stmt
        self._start_run(stmt)
        logger.debug(f"[{self.log_index}] MathAgent starting solve for problem_idx={self.problem_idx}, run_idx={self.run_idx}")

        query = [
            {
                "role": "developer",
                "content": self.scaffold_config["main"]["sysprompt"]
            }, 
            {
                "role": "user",
                "content": self.scaffold_config["main"]["prompt"].format(
                    problem_statement=stmt,
                    max_calls=self.max_calls
                )
            }
        ]
        response = self._query(self.client, query)
        
        if self.output is None:
            logger.debug(f"[{self.log_index}] No solution submitted by the agent, reminding to submit.")
            response += [
                {
                    "role": "user",
                    "content": self.scaffold_config["main"]["reminder_submit"]
                }
            ]
            response = self._query(self.client, response)

        self._add_history(
            step="Main Solve Loop",
            timestep=0,
            conversation=response
        )

        return self._end_run(self.output if self.output is not None else "No solution submitted by the agent.")
    
    def check_tool_calls_left(self):
        if self.total_calls_left <= 0:
            return False, self.scaffold_config.get("no_tools_message", "No tool calls left.")
        return True, ""

    def submit_solution(self, solution_id: str):
        logger.debug(f"Submitting solution with ID: {solution_id}")
        if solution_id not in self.solutions and not solution_id.startswith("solution_"):
            solution_id = f"solution_{solution_id}"
        if solution_id not in self.solutions:
            return f"Error: SolutionID {solution_id} not found."
        self.output = self.solutions[solution_id].response_text
        return self.scaffold_config["submit"]["output_format"].format(
            solution_id=solution_id
        )
    
    def determine_approaches(self, number_of_approaches: int = 2):
        logger.debug(f"[{self.log_index}] Determining {number_of_approaches} approaches.")
        component = DetermineApproaches(number_of_approaches)
        response = self._caller(component)
        for resp in component.responses:
            self.approaches[resp.response_id] = resp
        return response

    def solve_problem(self, approach_id: str = None):
        logger.debug(f"[{self.log_index}] Solving problem.")
        if approach_id is None:
            component = Solver()
        else:
            if approach_id not in self.approaches and not approach_id.startswith("approach_"):
                approach_id = f"approach_{approach_id}"
            if approach_id not in self.approaches:
                return f"Error: ApproachID {approach_id} not found."
            component = ApproachSolver(self.approaches[approach_id])
        
        response = self._caller(component)
        self.solutions[component.response_id] = component
        self.solution_id_to_approach_id[component.response_id] = approach_id
        return response

    def verify_solution(self, solution_id: str):
        logger.debug(f"[{self.log_index}] Verifying solution {solution_id}.")
        if solution_id not in self.solutions and not solution_id.startswith("solution_"):
            solution_id = f"solution_{solution_id}"
        if solution_id not in self.solutions:
            return f"Error: SolutionID {solution_id} not found."
        
        component = VerifySolution(self.solutions[solution_id])
        response = self._caller(component)
        self.verifications[component.response_id] = component
        return response

    def improve_solution(self, solution_id: str, verification_id: str):
        logger.debug(f"[{self.log_index}] Improving solution {solution_id} based on verification {verification_id}.")
        
        if solution_id not in self.solutions and not solution_id.startswith("solution_"):
            solution_id = f"solution_{solution_id}"
        if solution_id not in self.solutions:
            return f"Error: SolutionID {solution_id} not found."
        if verification_id not in self.verifications and not verification_id.startswith("verification_"):
            verification_id = f"verification_{verification_id}"
        if verification_id not in self.verifications:
            return f"Error: VerificationID {verification_id} not found."
        
        component = ImproveSolution(
            self.solutions[solution_id],
            self.verifications[verification_id]
        )
        response = self._caller(component)
        new_solution_id = component.response_id
        self.solutions[new_solution_id] = component
        return response

    def select_best_solution(self, solution_ids: list[str]):
        logger.debug(f"[{self.log_index}] Selecting best solution among {solution_ids}.")
        
        valid_solution_ids = []
        for solution_id in solution_ids:
            if solution_id not in self.solutions and not solution_id.startswith("solution_"):
                solution_id = f"solution_{solution_id}"
            if solution_id in self.solutions:
                valid_solution_ids.append(solution_id)
            else:
                return f"Error: SolutionID {solution_id} not found."
        
        response_list_solutions = [self.solutions[sol_id] for sol_id in valid_solution_ids]
        component = Selector(response_list_solutions)
        response = self._caller(component)
        return response

    def merge_solutions(self, solution_ids: list[str]):
        logger.debug(f"[{self.log_index}] Merging solutions {solution_ids}.")
        
        valid_solution_ids = []
        for solution_id in solution_ids:
            if solution_id not in self.solutions and not solution_id.startswith("solution_"):
                solution_id = f"solution_{solution_id}"
            if solution_id in self.solutions:
                valid_solution_ids.append(solution_id)
            else:
                return f"Error: SolutionID {solution_id} not found."
        
        response_list_solutions = [self.solutions[sol_id] for sol_id in valid_solution_ids]
        component = Merger(response_list_solutions)
        response = self._caller(component)
        merged_solution_id = component.response_id
        self.solutions[merged_solution_id] = component
        return response