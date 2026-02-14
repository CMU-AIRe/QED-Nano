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
    parser.add_argument("--data-path", type=str, default=None, help="Path to the model solutions file.")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save the results. By default, same file as data_path.")
    parser.add_argument("--problem-column", type=str, default="problem", help="Column name for problems in the dataset.")
    parser.add_argument("--model-solution-column", type=str, default="model_solution", help="Column name for model solutions in the dataset.")
    parser.add_argument("--solution-column", type=str, default="solution", help="Column name for student solutions in the dataset.")
    parser.add_argument("--answer-column", type=str, default="answer", help="Column name for correct answers in the dataset.")
    parser.add_argument("--grading-column", type=str, default="grading_guidelines", help="Column name for grading guidelines in the dataset.")
    parser.add_argument("--ground-truth", action="store_true", help="Whether to use ground truth solutions for grading.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite intermediate progress.")
    parser.add_argument("--matharena", action="store_true", help="Whether to use MathArena grading bench instead.")
    parser.add_argument("--dict-prompt", default="configs/prompts/gradingbench/dict-to-string.txt", help="Whether to convert grading scheme dict to string.")
    args = parser.parse_args()

    dict_prompt = ""
    if args.dict_prompt:
        with open(args.dict_prompt, 'r') as f:
            dict_prompt = f.read()
    elif args.prompt is None and args.ground_truth:
        args.prompt = "configs/prompts/gradingbench_gt.txt"
    elif args.prompt is None and not args.ground_truth:
        args.prompt = "configs/prompts/gradingbench.txt"
    if args.data_path is None and not args.matharena:
        args.data_path = "lm-provers/olympiads-proof-graderbench"
        args.grading_column = "schema_0"
        args.problem_column = "text"
    elif args.data_path is None and args.matharena:
        args.data_path = "lm-provers/matharena-gradingbench"
        args.model_solution_column = "answer"
        args.grading_column = "details"

    # Load model configuration, prompt template, questions, solutions, and answers
    with open(os.path.join("configs/models", args.model_config + ".yaml"), 'r') as f:
        model_config = yaml.safe_load(f)

    with open(args.prompt, 'r') as f:
        prompt_template = f.read()

    if os.path.exists(args.data_path):
        dataset = load_dataset('json', data_files=args.data_path)['train']
    else:
        # load from hf
        dataset = load_dataset(args.data_path)['train']

    # only use first 10 samples to test
    # dataset = dataset.select(range(min(10, len(dataset))))

    questions = list(dataset[args.problem_column])
    solutions = list(dataset[args.model_solution_column])
    
    if args.matharena or not args.ground_truth:
        grading_guidelines = list(dataset[args.grading_column])
        gt_solutions = ["" for _ in range(len(questions))]
    else:
        grading_guidelines = list(dataset[args.grading_column])
        gt_solutions = list(dataset[args.solution_column])
    results = grade_proofbench(model_config, prompt_template, 
                               questions, solutions, gt_solutions, 
                                grading_guidelines, overwrite=args.overwrite, 
                                other_params={
                                    "output_path": args.output_path,
                                    "ground_truth": args.ground_truth,
                                    "matharena": args.matharena
                                    }, grading_formatting=dict_prompt)

    # Save the results
    output_path = args.output_path
    if output_path is None:
        output_path = args.data_path
        assert os.path.exists(output_path), "Output path not specified and data path does not exist."

    if "grade_cost" in dataset.column_names:
        dataset = dataset.remove_columns("grade_cost")
    if "grade" in dataset.column_names:
        dataset = dataset.remove_columns("grade")
    if "graded_score" in dataset.column_names:
        dataset = dataset.remove_columns("graded_score")
        
    dataset = dataset.add_column("grade_cost", [row["cost"] for row in results]) 
    
    if "schema_0" not in dataset.column_names:
        dataset = dataset.add_column("schema_0", [[{"desc": "Points awarded for the proof.", "title": "Proof Grade", "points": 7}]] * len(dataset))
    dataset = dataset.add_column("grade", [
        [
            {
                "points": row["parsed_grade"], 
                "desc": row["response"]
            }
        ] for row in results
    ])
    dataset = dataset.add_column("graded_score", [row["parsed_grade"] for row in results])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_json(output_path)

    sum_cost = sum([row["cost"]["cost"] for row in results])
    logger.info(f"Total cost for grading: ${sum_cost:.4f}")