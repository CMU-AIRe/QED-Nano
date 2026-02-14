import re
from loguru import logger



class IDManager:
    def __init__(self):
        self.all_ids = set()

    def get_next_id(self, start_string=""):
        max_id = 1
        for existing_id in self.all_ids:
            if existing_id.startswith(start_string):
                numeric_part = int(existing_id[len(start_string) + 1:])
                if numeric_part >= max_id:
                    max_id = numeric_part + 1
        new_id = f"{start_string}_{max_id}"
        self.all_ids.add(new_id)
        return new_id
    
class Response:
    def __init__(self, response_id=None, response_text=None, origin=None):
        self.response_id = response_id
        self.response_text = response_text
        self.origin = origin

    def is_ready(self):
        return self.response_id is not None and self.response_text is not None
    
    def get_input_nodes(self):
        return [self.origin]

class CoreComponent(Response):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        super().__init__(response_id=None, response_text=None, origin=self)

    def can_start(self):
        if len(self.input_dict) == 0:
            return True
        for value in self.input_dict.values():
            if isinstance(value, Response):
                if not value.is_ready():
                    return False
            elif isinstance(value, list):
                for item in value:
                    if not item.is_ready():
                        return False
        return True
    
    def get_input_nodes(self):
        input_nodes = set()
        for value in self.input_dict.values():
            if isinstance(value, CoreComponent):
                input_nodes.add(value)
            elif isinstance(value, Response):
                input_nodes.add(value.origin)
            elif isinstance(value, list):
                for item in value:
                    input_nodes.add(item)
        return input_nodes
    
    def prepare_query(self, problem_statement):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def process_response(self, response_data):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def format_response(self, calls_left=""):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __str__(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Solver(CoreComponent):
    system_prompt: str
    prompt: str
    output_format: str = ""
    def __init__(self):
        super().__init__({})

    def prepare_query(self, problem_statement):
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(problem_statement=problem_statement)}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        self.response_text = response_data[-1]["content"]
        self.response_id = id_manager.get_next_id("solution")

    def format_response(self, calls_left=""):
        return self.output_format.format(
            solution_id=self.response_id,
            solution=self.response_text, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Solution: {self.response_id}"


class DetermineApproaches(CoreComponent):
    system_prompt: str
    prompt: str
    approach_format: str = ""
    output_format: str = ""
    pattern: str = r"###\s*Suggested Approach\s*(.*?)(?=###\s*Suggested Approach\s*|$)"
    def __init__(self, length):
        super().__init__({})
        self.length = length
        self.responses = [Response(origin=self) for _ in range(length)]

    def __index__(self, index):
        return self.responses[index]
    
    def __iter__(self):
        return iter(self.responses)
    
    def __len__(self):
        return self.length

    def prepare_query(self, problem_statement):
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(problem_statement=problem_statement, number_of_approaches=self.length)}
        ]
        return query
    
    def set_response(self, index, response_id, response_text):
        self.responses[index].response_id = response_id
        self.responses[index].response_text = response_text
    
    def process_response(self, response_data, id_manager):
        approaches_text = response_data[-1]["content"]
        matches = re.findall(self.pattern, approaches_text, re.DOTALL)
        for i in range(min(self.length, len(matches))):
            approach_text = matches[i].strip()
            approach_text = re.sub(r"^#+\s*", "", approach_text)
            approach_text = re.sub(r"\s*#+$", "", approach_text)
            approach_id = id_manager.get_next_id("approach")
            self.set_response(i, approach_id, approach_text)
        for i in range(self.length):
            if not self.responses[i].is_ready():
                approach_id = id_manager.get_next_id("approach")
                self.set_response(i, approach_id, "No valid approach generated.")

    def format_response(self, calls_left=""):
        formatted_approaches = []
        for response in self.responses:
            formatted_approaches.append(self.approach_format.format(
                approach_id=response.response_id,
                approach=response.response_text
            ))
        return self.output_format.format(
            approaches="\n\n".join(formatted_approaches),
            tool_calls=calls_left
        )
    
    def __str__(self):
        approach_strs = [resp.response_id for i, resp in enumerate(self.responses)]
        return " - ".join(approach_strs)
    
class ApproachSolver(CoreComponent):
    system_prompt: str
    prompt: str
    output_format: str = ""
    def __init__(self, approach : Response):
        super().__init__({
            "approach": approach
        })

    def prepare_query(self, problem_statement):
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(
                problem_statement=problem_statement, 
                approach=self.input_dict["approach"].response_text)}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        self.response_text = response_data[-1]["content"]
        self.response_id = id_manager.get_next_id("solution")

    def format_response(self, calls_left=""):
        return self.output_format.format(
            solution_id=self.response_id,
            solution=self.response_text, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Solution: {self.input_dict['approach'].response_id} -> {self.response_id}"
    
class VerifySolution(CoreComponent):
    system_prompt: str
    prompt: str
    output_format: str = ""
    def __init__(self, solution : CoreComponent):
        super().__init__({
            "solution": solution
        })

    def prepare_query(self, problem_statement):
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(
                problem_statement=problem_statement, 
                solution=self.input_dict["solution"].response_text)}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        self.response_text = response_data[-1]["content"]
        self.response_id = id_manager.get_next_id("verification")

    def format_response(self, calls_left=""):
        return self.output_format.format(
            verification_id=self.response_id,
            verification=self.response_text, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Verification: {self.input_dict['solution'].response_id} -> {self.response_id}"
    
class ImproveSolution(CoreComponent):
    system_prompt: str
    prompt: str
    improve_prompt: str
    output_format: str = ""
    def __init__(self, solution : CoreComponent, verification : CoreComponent):
        super().__init__({
            "solution": solution,
            "verification": verification
        })

    def prepare_query(self, problem_statement):
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(
                problem_statement=problem_statement, 
            )}, 
            {"role": "assistant", "content": self.input_dict["solution"].response_text},
            {"role": "user", "content": self.improve_prompt.format(
                bug_report=self.input_dict["verification"].response_text
            )}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        self.response_text = response_data[-1]["content"]
        self.response_id = id_manager.get_next_id("solution")

    def format_response(self, calls_left=""):
        return self.output_format.format(
            solution_id=self.response_id,
            solution=self.response_text, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Improved Solution: {self.input_dict['solution'].response_id}, {self.input_dict['verification'].response_id} -> {self.response_id}"

class Selector(CoreComponent):
    system_prompt: str
    prompt: str
    solution_format: str
    output_format: str = ""
    pattern: str = r"boxed\{\s*([A-Za-z]|equal)\s*\}"
    def __init__(self, solutions : list[CoreComponent]):
        super().__init__({
            "solutions": solutions
        })
        self.equal = False
        self.invalid = False
        self.reasoning = ""

    def prepare_query(self, problem_statement):
        formatted_solutions = [
            self.solution_format.format(
                index=chr(65 + i),  # 65 is ASCII for 'A'
                solution=solution.response_text
            )
            for i, solution in enumerate(self.input_dict["solutions"])
        ]
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(
                problem_statement=problem_statement, 
                solutions="\n\n".join(formatted_solutions)
            )}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        selection_text = response_data[-1]["content"]
        match = re.search(self.pattern, selection_text, re.IGNORECASE)
        if not match:
            self.invalid = True
            selected_solution = self.input_dict["solutions"][-1]
        elif match.group(1).lower() == "equal":
            self.equal = True
            selected_solution = self.input_dict["solutions"][-1]
        else:
            index = ord(match.group(1).upper()) - 65  # Convert letter back to index
            if index < 0 or index >= len(self.input_dict["solutions"]):
                self.invalid = True
                selected_solution = self.input_dict["solutions"][-1]
            else:
                selected_solution = self.input_dict["solutions"][index]
        self.response_id = selected_solution.response_id
        self.response_text = selected_solution.response_text
        self.reasoning = selection_text

    def format_response(self, calls_left=""):
        solution_id = self.response_id
        if self.equal:
            solution_id = "All solutions are considered equally good."
        elif self.invalid:
            solution_id = "No valid selection found in the response."
        return self.output_format.format(
            solution_id=solution_id,
            reasoning=self.reasoning, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Selected Solution: {', '.join([response.response_id for response in self.input_dict['solutions']])} -> {self.response_id}"


class Merger(CoreComponent):
    system_prompt: str
    prompt: str
    solution_format: str
    output_format: str = ""
    def __init__(self, solutions : list[CoreComponent]):
        super().__init__({
            "solutions": solutions
        })

    def prepare_query(self, problem_statement):
        formatted_solutions = [
            self.solution_format.format(
                index=chr(65 + i),  # 65 is ASCII for 'A'
                solution=sol.response_text
            )
            for i, sol in enumerate(self.input_dict["solutions"])
        ]
        query = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(
                problem_statement=problem_statement, 
                solutions="\n\n".join(formatted_solutions)
            )}
        ]
        return query
    
    def process_response(self, response_data, id_manager):
        self.response_text = response_data[-1]["content"]
        self.response_id = id_manager.get_next_id("solution")

    def format_response(self, calls_left=""):
        return self.output_format.format(
            solution_id=self.response_id,
            solution=self.response_text, 
            tool_calls=calls_left
        )
    
    def __str__(self):
        return f"Merged Solution: {', '.join([response.response_id for response in self.input_dict['solutions']])} -> {self.response_id}"
    

def set_prompts(prompts_dict):
    DetermineApproaches.system_prompt = prompts_dict["determine_approaches"]["sysprompt"]
    DetermineApproaches.prompt = prompts_dict["determine_approaches"]["prompt"]
    DetermineApproaches.approach_format = prompts_dict["determine_approaches"]["approach_format"]
    DetermineApproaches.output_format = prompts_dict["determine_approaches"]["output_format"]

    Solver.system_prompt = prompts_dict["solver"]["sysprompt"]
    Solver.prompt = prompts_dict["solver"]["prompt"]
    Solver.output_format = prompts_dict["solver"]["output_format"]

    ApproachSolver.system_prompt = prompts_dict["approach_solver"]["sysprompt"]
    ApproachSolver.prompt = prompts_dict["approach_solver"]["prompt"]
    ApproachSolver.output_format = prompts_dict["approach_solver"]["output_format"]

    VerifySolution.system_prompt = prompts_dict["verifier"]["sysprompt"]
    VerifySolution.prompt = prompts_dict["verifier"]["prompt"]
    VerifySolution.output_format = prompts_dict["verifier"]["output_format"]

    ImproveSolution.system_prompt = prompts_dict["solver"]["sysprompt"]
    ImproveSolution.prompt = prompts_dict["solver"]["prompt"]
    ImproveSolution.improve_prompt = prompts_dict["improver"]["prompt"]
    ImproveSolution.output_format = prompts_dict["improver"]["output_format"]

    Selector.system_prompt = prompts_dict["selector"]["sysprompt"]
    Selector.prompt = prompts_dict["selector"]["prompt"]
    Selector.solution_format = prompts_dict["selector"]["solution_format"]
    Selector.output_format = prompts_dict["selector"]["output_format"]

    Merger.system_prompt = prompts_dict["merger"]["sysprompt"]
    Merger.prompt = prompts_dict["merger"]["prompt"]
    Merger.solution_format = prompts_dict["merger"]["solution_format"]
    Merger.output_format = prompts_dict["merger"]["output_format"]