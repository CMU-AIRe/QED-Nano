import argparse
from imobench.evaluation import run_bench
from datasets import load_dataset, concatenate_datasets
import yaml
import os
from loguru import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AnswerBench dataset.")
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model configuration file. Relative to configs/models/")
    parser.add_argument("--prompt", type=str, default=None, help="Path to the prompt template file.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the model solutions file. Default to our HF Hub IMOBench datasets.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--problem-column", type=str, default="problem", help="Column name for problems in the dataset.")
    parser.add_argument("--final-answer", action="store_true", help="Whether its final answer or proof grading.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite intermediate progress.")
    parser.add_argument("--n", type=int, default=1, help="Number of times to run the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    args = parser.parse_args()

    if args.prompt is None and args.final_answer:
        args.prompt = "configs/prompts/answerbench_run.txt"
    elif args.prompt is None and not args.final_answer:
        args.prompt = "configs/prompts/proofbench_run.txt"
    if args.data_path is None and args.final_answer:
        args.data_path = "lm-provers/IMOBench-FinalAnswer"
    elif args.data_path is None and not args.final_answer:
        args.data_path = "lm-provers/IMOProofBench"

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
    # dataset = dataset.select(range(min(2, len(dataset))))
    if args.n > 1:
        # Create a list containing the dataset repeated n times
        datasets_list = [dataset] * args.n
        
        # Use the top-level function to combine them
        dataset = concatenate_datasets(datasets_list)


    questions = list(dataset[args.problem_column])
    results = run_bench(model_config, prompt_template, questions, 
                        overwrite=args.overwrite, other_params={
        "output_path": args.output_path,
        "n": args.n,
        "model_config_name": args.model_config,
    }, path=args.model_config)

    # Save the results
    output_path = args.output_path
    
    if args.final_answer:
        dataset = dataset.add_column("schema_0", [[{"desc": "Whether the answer is correct.", "title": "Answer Correctness", "points": 1}]] * len(dataset))
    elif "grading_guidelines" in dataset.column_names:
        dataset = dataset.add_column("schema_0", [[{"desc": grading_guidelines, "title": "Proof Grade", "points": 7}] for grading_guidelines in dataset["grading_guidelines"]])
    dataset = dataset.add_column("model_solution", [row["response"] for row in results])
    try:
        dataset = dataset.add_column("history", [row["history"] for row in results])
    except:
        pass
    dataset = dataset.add_column("cost_run", [row["cost"] for row in results])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_json(args.output_path)

    sum_cost = sum([row["cost"]["cost"] for row in results])
    logger.info(f"Total cost for running: ${sum_cost:.4f}")
