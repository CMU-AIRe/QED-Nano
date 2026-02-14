import argparse
from imobench.evaluation import grade_answerbench, grade_proofbench
from datasets import Dataset, load_dataset
from loguru import logger
import yaml
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AnswerBench dataset.")
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model configuration file. Relative to configs/models/")
    parser.add_argument("--prompt", type=str, default=None, help="Path to the prompt template file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the model solutions file.")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save the results. By default, same file as data_path.")
    parser.add_argument("--problem-column", type=str, default="problem", help="Column name for problems in the dataset.")
    parser.add_argument("--model-solution-column", type=str, default="model_solution", help="Column name for model solutions in the dataset.")
    parser.add_argument("--solution-column", type=str, default="solution", help="Column name for student solutions in the dataset.")
    parser.add_argument("--answer-column", type=str, default="answer", help="Column name for correct answers in the dataset.")
    parser.add_argument("--grading-column", type=str, default="grading_guidelines", help="Column name for grading guidelines in the dataset.")
    parser.add_argument("--final-answer", action="store_true", help="Whether its final answer or proof grading.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite intermediate progress.")
    parser.add_argument("--proofbench", action="store_true", help="Whether to run ProofBench instead of IMOBench.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    args = parser.parse_args()

    if args.prompt is None and args.final_answer:
        args.prompt = "configs/prompts/answerbench.txt"
    elif args.proofbench:
        args.prompt = "configs/prompts/proofbench_proofbench.txt"
        args.grading_column = "grading_scheme"
    elif args.prompt is None and not args.final_answer:
        args.prompt = "configs/prompts/proofbench.txt"
    

    # Load model configuration, prompt template, questions, solutions, and answers
    with open(os.path.join("configs/models", args.model_config + ".yaml"), 'r') as f:
        model_config = yaml.safe_load(f)

    with open(args.prompt, 'r') as f:
        prompt_template = f.read()

    if os.path.exists(args.data_path):
        dataset = load_dataset('json', data_files=args.data_path)[args.split]
    else:
        # load from hf
        dataset = load_dataset(args.data_path)[args.split]

    # only use first 10 samples to test
    # dataset = dataset.select(range(min(10, len(dataset))))

    questions = list(dataset[args.problem_column])
    solutions = list(dataset[args.model_solution_column])

    if args.final_answer:
        answers = list(dataset[args.answer_column])
        results = grade_answerbench(model_config, prompt_template, questions, solutions, answers, 
                                    overwrite=args.overwrite, other_params={
                                        "output_path": args.output_path
                                        })
    else:
        grading_guidelines = list(dataset[args.grading_column])
        gt_solutions = list(dataset[args.solution_column])
        results = grade_proofbench(model_config, prompt_template, questions, solutions, gt_solutions, 
                                   grading_guidelines, overwrite=args.overwrite, other_params={
                                        "output_path": args.output_path
                                        })

    # Save the results
    output_path = args.output_path
    if output_path is None:
        output_path = args.data_path
        assert os.path.exists(output_path), "Output path not specified and data path does not exist."

    # remove any column names that I want to add but already exist
    if "grade_cost" in dataset.column_names:
        dataset = dataset.remove_columns("grade_cost")
    if "schema_0" in dataset.column_names:
        dataset = dataset.remove_columns("schema_0")
    if "grade" in dataset.column_names:
        dataset = dataset.remove_columns("grade")
    if "score" in dataset.column_names:
        dataset = dataset.remove_columns("score")


    dataset = dataset.add_column("grade_cost", [row["cost"] for row in results]) 
    if args.final_answer:
        dataset = dataset.add_column("schema_0", [[{"desc": "Whether the answer is correct.", "title": "Answer Correctness", "points": 1}]] * len(dataset))
        dataset = dataset.add_column("grade", [
            [
                {
                    "points": int(row["is_correct"]), 
                    "desc": row["response"]
                }
            ] for row in results
        ])
        dataset = dataset.add_column("score", [int(row["is_correct"]) for row in results])
    else:
        dataset = dataset.add_column("schema_0", [[{"desc": "Points awarded for the proof.", "title": "Proof Grade", "points": 7}]] * len(dataset))
        dataset = dataset.add_column("grade", [
            [
                {
                    "points": row["parsed_grade"], 
                    "desc": row["response"]
                }
            ] for row in results
        ])
        dataset = dataset.add_column("score", [row["parsed_grade"] for row in results])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_json(output_path)

    sum_cost = sum([row["cost"]["cost"] for row in results])
    logger.info(f"Total cost for grading: ${sum_cost:.4f}")