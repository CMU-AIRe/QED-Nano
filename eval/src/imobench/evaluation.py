from imobench.api_client import APIClient
from imobench.utils import sanitize_model_config, extract_html, find_last_boxed_content
from imobench.request_logger import request_logger
from hashlib import md5
from imobench.agents import AgentPool, PureModelSolver
import os
import json
import yaml

import re

def remove_self_evaluation(text):
    """
    Removes sections starting with headings like:
      # Evaluation
      ## Self Evaluation
      ### Final Evaluation
    Accepts 1+ '#' and ignores '*' formatting in the heading.
    """
    pattern = r"(?ms)^#+\s*[*\s]*(?:(?:Self|Final)\s*[*\s]*){0,2}Evaluation\b.*$"
    
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text.strip()

def init_responses(file, overwrite, length):
    if os.path.exists(file) and not overwrite:
        with open(file, "r") as f:
            responses = json.load(f)
        return responses
    else:
        responses = [None] * length
    return responses


def run_bench(model_config, prompt, questions, overwrite=False, other_params=None, path="unknown_model"):
    """
    Runs the ProofBench dataset using the specified model configuration.

    Args:
        model_config (dict): Configuration for the language model.
        prompt (str): The prompt template to use.
        questions (list): List of questions from the ProofBench dataset.

    Returns:
        list: Model responses for each question.
    """
    other_params_string = "" if other_params is None else str(other_params)
    hash_run = md5((str(model_config) + prompt + "".join(questions) + other_params_string).encode()).hexdigest()
    save_file = f"logs/intermediate/{hash_run}.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    request_logger.set_metadata(solver_name=model_config.get("model", path))
    agent = AgentPool.load_agent_from_config(model_config)
    agent.set_path(path.replace(".yaml", ""))
    responses = init_responses(save_file, overwrite, len(questions))
    all_prompts = []
    indices = []
    batch_idx_to_problem_idx = {}  # index in batch -> problem_idx
    batch_idx_to_run_idx = {}  # index in batch -> run_idx
    for i, question in enumerate(questions):
        if responses[i] is not None:
            continue
        full_prompt = prompt.format(problem_statement=question)
        all_prompts.append(full_prompt)
        indices.append(i)
        batch_idx_to_problem_idx[len(all_prompts) - 1] = i
        batch_idx_to_run_idx[len(all_prompts) - 1] = 0

    for solver_response in agent.solve_batch(all_prompts, batch_idx_to_problem_idx, 
                                                 batch_idx_to_run_idx):
        if "content" not in solver_response.conversation[-1]:
            # This can happen with the gpt-oss models sometimes you have a last message "{'role': 'assistant', 'tool_calls': []}"
            print("Warning: No content in response:", solver_response.conversation[-1])
            # pop last
            solver_response.conversation.pop()
        
        responses[indices[solver_response.idx]] = {
            "response": solver_response.conversation[-1]["content"],
            "history": solver_response.history,
            "cost": solver_response.detailed_cost
        }
        # save intermediate results
        with open(save_file, "w") as f:
            json.dump(responses, f, indent=4)


    return responses

def grade_proofbench(model_config, prompt, questions, solutions, gt_solutions, 
                     grading_guidelines, overwrite=False, other_params=None, 
                     grading_formatting=None):
    """
    Grades the solutions for the ProofBench dataset using the specified model configuration.

    Args:
        model_config (dict): Configuration for the language model.
        prompt (str): The prompt template to use for grading.
        questions (list): List of questions from the ProofBench dataset.
        solutions (list): List of model-generated solutions to be graded.
        gt_solutions (list): List of ground truth solutions.
        grading_guidelines (str): Guidelines for grading the solutions.

    Returns:
        list: Grading results for each solution.
    """
    other_params_string = "" if other_params is None else str(other_params)
    hash_run = md5((str(model_config) + prompt + "".join(questions) + "".join(solutions) + other_params_string).encode()).hexdigest()
    save_file = f"logs/intermediate/{hash_run}.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    request_logger.set_metadata(solver_name=model_config.get("model", "unknown_model"))
    api_client = APIClient(**sanitize_model_config(model_config))

    responses = init_responses(save_file, overwrite, len(solutions))

    all_prompts = []
    indices = []
    for i, question, solution, gt_solution, gg in zip(range(len(questions)), questions, solutions, gt_solutions, grading_guidelines):
        if responses[i] is not None:
            continue

        solution = remove_self_evaluation(solution)

        if isinstance(gg, list) and grading_formatting is not None:
            # format the grading guidelines according to the formatting template
            formatted_gg = "\n\n".join([grading_formatting.format(
                title=item.get("title", "No Title"),
                points=item.get("max_points", item.get("points", 0)),
                description=item.get("grading_scheme_desc", item.get("desc", "No Description"))
            ) for item in gg])
            max_points = sum([item.get("max_points", 0) for item in gg])
            if gt_solution:
                full_prompt = prompt.format(
                    problem=question,
                    solution=solution,
                    reference_solution=gt_solution,
                    marking_scheme=formatted_gg,
                    max_points=int(max_points)
                )
            else:
                full_prompt = prompt.format(
                    problem=question,
                    solution=solution,
                    marking_scheme=formatted_gg,
                    max_points=int(max_points)
                )
        else:
            full_prompt = prompt.format(
                problem_statement=question,
                student_answer=solution,
                solution=gt_solution,
                guidelines=gg
            )
        all_prompts.append([
            {
                "role": "user",
                "content": full_prompt
            }
        ])
        indices.append(i)
        

    for idx, response, cost in api_client.run_queries(all_prompts):
        parsed_response = extract_html(response[-1]["content"], "points")
        parsed_grade = int(parsed_response[0]) if parsed_response and parsed_response[0].isdigit() else None
        responses[indices[idx]] = {
            "response": response[-1]["content"],
            "cost": cost,
            "history": response[:-1],
            "parsed_grade": parsed_grade
        }
        # save intermediate results
        with open(save_file, "w") as f:
            json.dump(responses, f, indent=4)

    return responses


def grade_answerbench(model_config, prompt, questions, solutions, answers, overwrite=False, 
                      other_params=None):
    """
    Grades the solutions for the AnswerBench dataset using the specified model configuration.

    Args:
        model_config (dict): Configuration for the language model.
        prompt (str): The prompt template to use for grading.
        questions (list): List of questions from the AnswerBench dataset.
        solutions (list): List of model-generated solutions to be graded.
        answers (list): List of correct answers.

    Returns:
        list: Grading results for each solution.
    """
    other_params_string = "" if other_params is None else str(other_params)
    hash_run = md5((str(model_config) + prompt + "".join(questions) + "".join(solutions) + other_params_string).encode()).hexdigest()
    save_file = f"logs/intermediate/{hash_run}.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    request_logger.set_metadata(solver_name=model_config.get("model", "unknown_model"))

    api_client = APIClient(**sanitize_model_config(model_config))
    responses = init_responses(save_file, overwrite, len(solutions))
    all_prompts = []
    indices = []
    for i, question, solution, answer in zip(range(len(questions)), questions, solutions, answers):
        if responses[i] is not None:
            continue
        
        solution = remove_self_evaluation(solution)
        
        parsed_solution = find_last_boxed_content(solution)
        if not parsed_solution:
            # extract the last five lines as fallback
            lines = solution.strip().split("\n")
            parsed_solution = "..." + "\n".join(lines[-5:]) if len(lines) >= 5 else solution
        full_prompt = prompt.format(
            problem_statement=question,
            student_answer=parsed_solution,
            gold_answer=answer
        )
        all_prompts.append([
            {
                "role": "user",
                "content": full_prompt
            }
        ])
        indices.append(i)
        
    for idx, response, cost in api_client.run_queries(all_prompts):
        parsed_response = find_last_boxed_content(response[-1]["content"])
        correctness = "incorrect" not in parsed_response.lower() if parsed_response else False
        responses[indices[idx]] = {
            "response": response[-1]["content"],
            "cost": cost,
            "history": response[:-1],
            "parsed_answer": parsed_response,
            "is_correct": correctness
        }
        # save intermediate results
        with open(save_file, "w") as f:
            json.dump(responses, f, indent=4)

    return responses