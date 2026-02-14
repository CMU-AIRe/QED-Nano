import argparse
import json
import os
import re
import pandas as pd
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import list_datasets
import argparse
import re
import numpy as np
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request, redirect

parser = argparse.ArgumentParser()
parser.add_argument("--max-rows", type=int, default=5000, help="Maximum number of rows to load from the dataset")
args = parser.parse_args()

datasets = dict()


def get_dataset(dataset_name):
    if dataset_name not in datasets:
        return None
    datasets[dataset_name]["last_accessed"] = datetime.now()
    return datasets[dataset_name]

def load_data_from_path(data_path, max_rows, dataset_config=None, dataset_split="train"):
    if os.path.exists(data_path):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            df = load_dataset('json', data_files=data_path, split=dataset_split).to_pandas()
        else:
            df = load_dataset(data_path, split=dataset_split).to_pandas()
    else:
        if dataset_config:
            df = load_dataset(data_path, dataset_config, split=dataset_split).to_pandas()
        else:
            df = load_dataset(data_path, split=dataset_split).to_pandas()

    df = df.head(max_rows).reset_index(drop=True)
    
    dataset = dict()
    dataset["df"] = df
    dataset["regex_filterable_columns"] = [
        col for col in df.columns if all(isinstance(val, str) for val in df[col])
    ]

    dataset["range_filterable_columns"] = [
        col for col in df.columns if all(isinstance(val, (int, float)) for val in df[col])
    ] + ["idx"]

    dataset["range_default"] = {
        col: (df[col].min(), df[col].max()) for col in dataset["range_filterable_columns"] if col in df.columns and not df[col].empty
    }

    dataset["range_default"]["idx"] = (0, len(df))

    dataset["boolean_filterable_columns"] = [
        col for col in df.columns if all(isinstance(val, bool) for val in df[col])
    ] + ["has_schema", "has_grade", "has_issue"]
    return dataset

app = Flask(__name__)

_dataset_search_cache = dict()
_dataset_info_cache = dict()

@app.before_request
def cleanup_old_datasets():
    for dataset_name, dataset in list(datasets.items()):
        if datetime.now() - dataset["last_accessed"] > timedelta(hours=24):
            del datasets[dataset_name]


@app.route("/")
def index():
    return render_template("main.html")

@app.route("/view/<dataset_name>")
def view_dataset(dataset_name):
    dataset = get_dataset(dataset_name)
    if dataset is None:
        return redirect("/")
    return render_template("index.html", 
        dataset_name=dataset_name,
        regex_filterable_columns=dataset["regex_filterable_columns"],
        range_filterable_columns=dataset["range_filterable_columns"],
        boolean_filterable_columns=dataset["boolean_filterable_columns"],
        range_default=dataset["range_default"],
    )

@app.route("/load", methods=["POST"])
def load_dataset_route():
    dataset_name = request.form.get("dataset")
    dataset_config = request.form.get("config", "").strip()
    dataset_split = request.form.get("split", "").strip() or "train"
    if dataset_name:
        dataset = load_data_from_path(dataset_name, args.max_rows, dataset_config or None, dataset_split)
        dataset_key = dataset_name
        if dataset_config:
            dataset_key = f"{dataset_key}__{dataset_config}"
        dataset_key = f"{dataset_key}__{dataset_split}"
        dataset_key = dataset_key.replace("/", "--")
        dataset["last_accessed"] = datetime.now()
        datasets[dataset_key] = dataset
        return redirect(f"/view/{dataset_key}")
    return redirect("/")

@app.route("/datasets")
def list_datasets_route():
    search = request.args.get("search", "").strip()
    try:
        if search in _dataset_search_cache:
            return jsonify({"datasets": _dataset_search_cache[search]})
        datasets_iter = list_datasets(search=search or None, limit=50)
        dataset_names = [dataset.id for dataset in datasets_iter]
        _dataset_search_cache[search] = dataset_names
        return jsonify({"datasets": dataset_names})
    except Exception as e:
        return jsonify({"datasets": [], "error": str(e)})

@app.route("/dataset-info")
def dataset_info_route():
    dataset_name = request.args.get("name", "").strip()
    dataset_config = request.args.get("config", "").strip() or None
    if not dataset_name:
        return jsonify({"configs": [], "splits": []})

    cache_key = f"{dataset_name}::{dataset_config or ''}"
    if cache_key in _dataset_info_cache:
        return jsonify(_dataset_info_cache[cache_key])

    configs = []
    splits = []
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = []

    try:
        splits = get_dataset_split_names(dataset_name, dataset_config)
    except Exception:
        splits = []

    payload = {"configs": configs, "splits": splits}
    _dataset_info_cache[cache_key] = payload
    return jsonify(payload)

def create_tagline(history_element):
    tagline = ""
    if "role" in history_element and history_element["role"]:
        tagline = f"{history_element['role'].capitalize()}"
    if "type" in history_element and history_element["type"]:
        tagline += f" - {history_element['type'].capitalize()}"
    if "tool_name" in history_element and history_element["tool_name"]:
        tagline += f" ({history_element['tool_name']})"
    return tagline.replace("_", " ")

def postprocess_data(data_df, df):
    data_to_htmlify = []
    sidebar_metadata = []

    for _, row in data_df.iterrows():
        row_dict = {
            "problem_statement": row.get("problem", "No problem found"),
            "solution": row.get("solution", row.get("answer", "No solution found")),
            "model_solution": row.get("model_solution", None),
            "has_issue": False,
            "has_grade": False,
            "has_schema": False,
            "issue_message": ""
        }
        row_metadata = dict()
        try:
            if "schema_0" in df.columns and isinstance(row["schema_0"], (list, np.ndarray)):
                row_dict["has_schema"] = True
                schema = row["schema_0"]
                output = {"max": 0, "elements": []}
                if "grade" in df.columns and isinstance(row["grade"], (list, np.ndarray)):
                    row_dict["has_grade"] = True
                    grade = row["grade"]
                    output["grade"] = 0
                else:
                    grade = [dict()] * len(schema)
                
                for grade_part, schema_part in zip(grade, schema):
                    element = {
                        "title": schema_part.get("title", ""),
                        "max": schema_part.get("points", 0),
                        "grade": grade_part.get("points", None),
                        "description": schema_part.get("desc", ""),
                        "content": grade_part.get("desc", ""),
                    }
                    if element["grade"] is not None:
                        element["correct_indicator"] = "incorrect"
                        if element["grade"] >= element["max"]:
                            element["correct_indicator"] = "correct"
                        elif element["grade"] > 0:
                            element["correct_indicator"] = "semicorrect"

                    output["elements"].append(element)
                    output["max"] += schema_part.get("points", 0)
                    if "grade" in df.columns:
                        output["grade"] += grade_part.get("points", 0)
                row_dict["grading"] = output
            
            if "history" in df.columns:
                row_dict["history"] = row["history"]
                for message in row_dict["history"]:
                    if "tagline" not in message:
                        message["tagline"] = create_tagline(message)
            
            for col in df.columns:
                if col not in ["problem", "solution", "answer", "model_solution", "schema_0", "grade", "history"] and isinstance(row[col], (str, float, int)):
                    row_metadata[col] = row[col]
            row_dict["metadata"] = row_metadata.copy()
        except Exception as e:
            row_dict["has_issue"] = True
            row_dict["issue_message"] = str(e)
        data_to_htmlify.append(row_dict)
        sidebar_metadata.append(row_metadata)
        row_metadata["has_issue"] = row_dict["has_issue"]
        row_metadata["has_grade"] = row_dict["has_grade"]
        row_metadata["has_schema"] = row_dict["has_schema"]
        
    return data_to_htmlify, sidebar_metadata

def metadata_to_string(metadata):
    string = f"{metadata['idx']}"
    if "source" in metadata:
        string += f" - {metadata['source']}"
    if "problem_id" in metadata:
        string += f" - {metadata['problem_id']}"
    return string

@app.route("/data/<dataset_name>")
def load_data(dataset_name):
    dataset = get_dataset(dataset_name)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    df = dataset["df"]
    idx_range_start = int(request.args.get("idx_min", "0"))
    idx_range_end = int(request.args.get("idx_max", str(len(df))))

    data_df = df.copy()
    data_df = data_df.iloc[idx_range_start:idx_range_end]

    for col in dataset["regex_filterable_columns"]:
        col_filter_regex = request.args.get(f"{col}_filter", None)
        if col_filter_regex:
            data_df = data_df[data_df[col].str.contains(re.compile(col_filter_regex), na=False)]

    for col in dataset["range_filterable_columns"]:
        if col == "idx":
            continue
        col_min = request.args.get(f"{col}_min", None)
        col_max = request.args.get(f"{col}_max", None)
        if col_min is not None:
            data_df = data_df[data_df[col] >= float(col_min)]
        if col_max is not None:
            data_df = data_df[data_df[col] <= float(col_max)]

    data_df["idx"] = data_df.index

    data, sidebar_metadata = postprocess_data(data_df, df)

    for i, meta in enumerate(sidebar_metadata):
        meta['idx'] = data_df.index[i]

    for bool_col in dataset["boolean_filterable_columns"]:
        bool_filter = request.args.get(f"{bool_col}_filter", None)
        if bool_filter is not None and bool_filter.lower() in ["true", "false"]:
            bool_filter_value = bool_filter.lower() == "true"
            sidebar_metadata = [
                item for item in sidebar_metadata if item.get(bool_col, False) == bool_filter_value
            ]
            data = [
                item for item in data if item.get(bool_col, False) == bool_filter_value
            ]
    
    # sort by idx
    sidebar_metadata = sorted(sidebar_metadata, key=lambda x: x['idx'])

    return jsonify({
        "sidebar_metadata": [{
            "idx": int(meta['idx']),
            "display_string": metadata_to_string(meta)
        } for meta in sidebar_metadata
        ],
    })

@app.route("/problem/<dataset_name>/<int:idx>")
def get_problem(dataset_name, idx):
    dataset = get_dataset(dataset_name)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404
    
    df = dataset["df"]
    if 0 <= idx < len(df):
        problem_data_df = df.iloc[idx:idx+1].copy()
        data_to_htmlify, _ = postprocess_data(problem_data_df, df)
        if data_to_htmlify:
            return render_template("problem.html", **data_to_htmlify[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
