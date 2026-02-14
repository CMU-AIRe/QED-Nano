import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Any, override
from hashlib import md5

import time
from loguru import logger
import yaml

from imobench.agents import BaseAgent, SolverResponse, StaticMathAgent
from imobench.utils import load_config
from imobench.agents.math_core import *
from copy import deepcopy
import random



class RSAAgent(StaticMathAgent):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, log=True, solver_config_path=None):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, log=log, solver_config_path=solver_config_path)
        self.log_index = f"RSAAgent-P{self.problem_idx}-R{self.run_idx}"

    def sample_blocks(self, items, K, seed=None):
        N = len(items)

        rng = random.Random(seed)
        perm = items[:]           # randomize labels (optional but nice)
        rng.shuffle(perm)

        offsets = rng.sample(range(N), K)  # K distinct offsets
        return [[perm[(s + o) % N] for o in offsets] for s in range(N)]

    def create_agent(self):
        solutions = [Solver() for _ in range(self.scaffold_config["n_solutions"])]

        for i in range(self.scaffold_config["n_rounds"]):
            new_solutions = []
            blocks = self.sample_blocks(solutions, self.scaffold_config["block_size"], seed=i)
            for block in blocks:
                merged_solution = Merger(block)
                new_solutions.append(merged_solution)
            solutions = new_solutions[:]
        
        if len(solutions) == 1:
            return solutions[0]

        if self.scaffold_config["select_pairwise"]:
            while len(solutions) > 1:
                new_solutions = []
                for i in range(0, len(solutions), 2):
                    if i + 1 < len(solutions):
                        pair = [solutions[i], solutions[i + 1]]
                        selected_solution = Selector(pair)
                    else:
                        selected_solution = solutions[i]
                    new_solutions.append(selected_solution)
                solutions = new_solutions[:]
                random.shuffle(solutions)
            return solutions[0]
        else:
            # final merge if multiple remain
            final_merger = Selector(solutions)
            return final_merger