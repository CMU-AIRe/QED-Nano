"""Generic solver to inherit from (for base_model or agent solvers)."""

from typing import Any

class BaseSolver:
    """
    An abstract solver. It wraps an APIClient and uses it to solve problems.
    Subclassed by PureModelSolver and AgentPool.
    """

    def __init__(self, solver_config):
        """
        Initializes the solver.
        """
        self.solver_config = solver_config
        self.log = True
        self.path = None

    def set_path(self, path: str) -> None:
        """
        Stores the config path for logging/debugging.
        """
        self.path = path

    def solve_batch(self, stmt_batch: list[tuple[str, Any]], batch_idx_to_problem_idx: dict[int, int], batch_idx_to_run_idx: dict[int, int]):
        """
        Solves a batch of problems.

        Args:
            stmt_batch (list[tuple[str, Any]]): A list of problem statements (text, image) to be solved.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the index, conversation, detailed cost, and history for each problem.
        """
        raise NotImplementedError("Subclasses should implement solver.solve")
