"""This module defines a Chain-of-Thought (CoT) solver for math problems."""

from typing import Any, override

from imobench.api_client import APIClient
from imobench.agents import BaseSolver, SolverResponse


class PureModelSolver(BaseSolver):
    """
    A solver that wraps a pure model, prompting it once with the problem statement.
    """

    def __init__(self, solver_config):
        """
        Initializes the solver.
        """
        super().__init__(solver_config)
        self.client = APIClient(**solver_config)

    @override
    def solve_batch(self, stmt_batch: list[tuple[str, Any]], 
                    batch_idx_to_problem_idx: dict[int, int], 
                    batch_idx_to_run_idx: dict[int, int], no_tqdm=False):
        """
        Solves a batch of problems.

        Args:
            stmt_batch (list[tuple[str, Any]]): A batch of problem statements as (text, image) pairs.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the batch_index, the conversation array, detailed cost, and history for each problem.
        """

        queries = []
        for text in stmt_batch:
            if isinstance(text, str):
                queries.append([{"role": "user", "content": text}])
            else:
                # text is a (str, image) pair
                queries.append(text)
        for idx, conversation, detailed_cost in self.client.run_queries(queries, no_tqdm=no_tqdm):
            # History is None for pure model solver
            yield SolverResponse(idx, conversation, detailed_cost, history=conversation[:-1])
