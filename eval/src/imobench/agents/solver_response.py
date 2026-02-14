class SolverResponse:
    def __init__(self, idx: int, conversation: list[dict], detailed_cost: dict, history: list):
        """The response return from a solver for a specific problem.

        Args:
            idx (int): The index of the problem in the input batch.
            conversation (list[dict]): The array of message blocks representing the only (pure model) or the last interaction (agents). This is in the API format; needs to be converted before saving.
            detailed_cost (dict): The detailed cost dict.
            history (list): The array of interactions, each of format {"step": "step name", "messages": [...]}. The last one is the same as 'messages'. It is None for pure model solvers.
        """
        self.idx = idx
        self.conversation = conversation
        self.detailed_cost = detailed_cost
        self.history = history
        for convo in self.conversation:
            if "content" not in convo: # gemini does this, of course.
                convo["content"] = ""
