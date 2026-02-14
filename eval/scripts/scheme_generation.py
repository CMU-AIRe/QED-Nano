from imobench.api_client import APIClient
import argparse
from datasets import load_dataset, Dataset
import yaml
import pandas as pd
import os


parser = argparse.ArgumentParser(description="Generate grading schemes for IMO problems.")
parser.add_argument("--model-config", type=str, default="gemini-3-pro", help="Path to the model configuration file. Relative to configs/models/")
parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset.")
parser.add_argument("--output-path", type=str, default="outputs/grading_schemes.jsonl", help="Path to save the generated schemes.")
parser.add_argument("--prompt", type=str, default="configs/prompts/schema_generation.txt", help="Path to the prompt template file.")

args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)

with open(f"configs/models/{args.model_config}.yaml", 'r') as f:
    model_config = yaml.safe_load(f)

with open(args.prompt, 'r') as f:
    SCHEMA_GENERATOR_PROMPT = f.read()

client = APIClient(**model_config)

prompts = []
data = load_dataset(args.data_path, split="train").to_pandas()
# select 10 samples for testing
for problem, solution in zip(data["problem"], data["solution"]):
    prompt = SCHEMA_GENERATOR_PROMPT.format(problem=problem, solution=solution)
    prompts.append([{
        "role": "user",
        "content": prompt
    }])

results = []
total_cost = 0
for idx, result, cost in client.run_queries(prompts):
    schema = result[-1]["content"]
    if "</thought>" in schema:
        schema = schema.split("</thought>")[-1].strip()
    if "Checkpoints" in schema:
        schema = schema[schema.find("Checkpoints"):].strip()
    results.append({
        "problem": data["problem"][idx],
        "solution": data["solution"][idx],
        "grading_scheme": schema,
        "schema_0": data["schema_0"][idx] if "schema_0" in data.columns else None
    })
    total_cost += cost["cost"]
    # intermediate save
    dataframe = pd.DataFrame(results)
    dataframe.to_json(args.output_path, orient="records", lines=True)

print(f"Total cost for scheme generation: ${total_cost:.4f}")