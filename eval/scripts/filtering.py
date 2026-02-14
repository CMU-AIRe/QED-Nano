from imobench.api_client import APIClient
from datasets import Dataset, load_dataset
import argparse
import yaml
import re
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter IMOBench dataset based on grading cost.")
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model configuration file. Relative to configs/models/")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the data with the extra column.")
    parser.add_argument("--prompt", type=str, default="configs/prompts/filtering.txt", help="Path to the prompt template file.")

    args = parser.parse_args()

    data = load_dataset(args.data_path, split="train").to_pandas()
    client = APIClient(**yaml.safe_load(open(f"configs/models/{args.model_config}.yaml")))

    prompts = []

    with open(args.prompt, 'r') as f:
            prompt_template = f.read()
    
    for _, row in data.iterrows():
        prompt = prompt_template.format(problem=row['problem'], 
                                        solution=row['solution'],
                                        path=row["path"],
                                        comments="Comment:\n" + "\n\nComment:\n".join(row['candidates']))
        prompts.append([{
            "role": "user",
            "content": prompt
        }])
    
    total_cost = 0
    json_regex = r"```json(.*?)```"
    data["cost_model"] = 0.0
    data["extra_cols"] = None
    for idx, result, cost in client.run_queries(prompts):
        total_cost += cost["cost"]
        match = re.search(json_regex, result[-1]["content"], re.DOTALL)
        if not match:
            match = result[-1]["content"]
        else:
             match = match.group(1).strip()
        try:
            parsed = json.loads(match)
            data.at[idx, 'extra_cols'] = parsed
        except json.JSONDecodeError:
            data.at[idx, 'extra_cols'] = None
         
    print(f"Total cost for filtering: ${total_cost:.4f}")
    data.to_json(args.output_path, orient="records", lines=True)
