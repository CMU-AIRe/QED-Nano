import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def expand_output_paths(output_paths):
    expanded, unmatched = [], []
    
    for raw in output_paths:
        p = Path(raw).expanduser()
        
        if p.is_dir():
            matches = sorted(p.glob("*.jsonl"))
        elif any(ch in raw for ch in "*?["):
            matches = sorted(Path(m) for m in glob.glob(str(p), recursive=True)
                           if m.endswith(".jsonl"))
        elif p.is_file() and p.suffix == ".jsonl":
            matches = [p]
        else:
            matches = []
        
        if matches:
            expanded.extend(map(str, matches))
        else:
            unmatched.append(raw)
    
    if unmatched:
        raise SystemExit(f"No files matched: {', '.join(unmatched)}")
    
    deduped = list(dict.fromkeys(expanded))
    if not deduped:
        raise SystemExit("No input files found.")
    
    return deduped

def local_consistency(df):
    consistent = []
    problem_id_col = 'problem_id' if 'problem_id' in df.columns else 'problem_idx'
    if problem_id_col not in df.columns:
        return 0
    for problem_id in df[problem_id_col].unique():
        sub_df = df[df[problem_id_col] == problem_id]
        if len(sub_df) > 1:
            scores = sub_df['graded_score'].values
            consistency = np.mean([(scores[i] <= scores[j]) == (sub_df.iloc[i]['points'] <= sub_df.iloc[j]['points']) 
                                  for i in range(len(scores)) for j in range(i+1, len(scores))])
            consistent.append(consistency)
    return sum(consistent) / len(consistent) if consistent else None

def advantage_diff(df):
    consistent = []
    problem_id_col = 'problem_id' if 'problem_id' in df.columns else 'problem_idx'
    if problem_id_col not in df.columns:
        return 0
    for problem_id in df[problem_id_col].unique():
        sub_df = df[df[problem_id_col] == problem_id]
        if len(sub_df) > 1:
            scores = sub_df['graded_score'].values
            advantage = scores - np.mean(scores)
            points_advantage = sub_df['points'].values - np.mean(sub_df['points'].values)
            consistent.append(np.mean(np.abs(advantage - points_advantage)))
    return sum(consistent) / len(consistent) if consistent else None

def compute_accuracy(df):
    acc = 0
    for _, row in df.iterrows():
        if row['graded_score'] == 0 and row['points'] == 0:
            acc += 1
        elif row['graded_score'] == 7 and row['points'] == 7:
            acc += 1
        elif 1 <= row['graded_score'] <= 3 and 1 <= row['points'] <= 3:
            acc += 1
        elif 4 <= row['graded_score'] <= 6 and 4 <= row['points'] <= 6:
            acc += 1
    return acc / len(df)

def nanmean_or_none(values):
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))

def nansum_or_none(values):
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nansum(arr))

def token_stats(tokens, mask):
    selected = tokens[mask]
    if len(selected) == 0:
        return None, None, None
    return float(selected.min()), float(selected.mean()), float(selected.max())

def fmt_token(value):
    return "N/A" if value is None or pd.isna(value) else f"{int(value)}"

def truncated_frac(dfs):
    if not dfs:
        return None
    if not all("model_solution" in df.columns for df in dfs):
        return None
    all_solutions = pd.concat([df["model_solution"] for df in dfs], ignore_index=True)
    is_empty = all_solutions.apply(lambda x: isinstance(x, str) and x == "")
    return float(is_empty.mean())

def fmt_frac(value):
    return "N/A" if value is None or np.isnan(value) else f"{value:.4f}"

def _get_cost_tokens(cost_obj, key):
    if not isinstance(cost_obj, dict):
        return np.nan
    value = cost_obj.get(key, np.nan)
    return np.nan if value is None else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Statistics.")
    parser.add_argument(
        "output_path",
        type=str,
        nargs="+",
        help="Path(s) or glob pattern(s) to evaluation result file(s) (jsonl, e.g. outputs/run-*.jsonl).",
    )
    parser.add_argument("--merge-method", type=str, default="mean", choices=["mean", "max", "min"], help="Method to merge multiple result files.")
    parser.add_argument("--grader", action="store_true", help="Whether the results are from a grader.")
    args = parser.parse_args()
    output_paths = expand_output_paths(args.output_path)

    # load json lines file
    dfs = [pd.read_json(path, lines=True) for path in output_paths]

    if len({len(df) for df in dfs}) != 1:
        raise SystemExit("All input files must have the same number of rows to merge.")


    if args.grader:
        df = dfs[0].copy()
        graded_scores = [
            df_i["graded_score"].fillna(0, inplace=False) for df_i in dfs
        ]
        graded_scores_df = pd.concat(graded_scores, axis=1)
        if args.merge_method == "mean":
            df["graded_score"] = graded_scores_df.mean(axis=1)
        elif args.merge_method == "max":
            df["graded_score"] = graded_scores_df.max(axis=1)
        elif args.merge_method == "min":
            df["graded_score"] = graded_scores_df.min(axis=1)
        mae = abs(df['graded_score'] - df['points']).mean()
        acc = compute_accuracy(df)
        lc = local_consistency(df)
        adv = advantage_diff(df)
        df["output_tokens"] = df["grade_cost"].apply(lambda x: x['output_tokens'])
        tokens_mean = df['output_tokens'].mean()
        tokens_95 = df['output_tokens'].quantile(0.95)
        tokens_max = df['output_tokens'].max()

        # CoT tokens = cost_run.output_tokens - grade_cost.input_tokens
        # Solution tokens = grade_cost.input_tokens
        df["solution_tokens"] = df["grade_cost"].apply(lambda x: x['input_tokens'])
        df["cot_tokens"] = df["cost_run"].apply(lambda x: _get_cost_tokens(x, "output_tokens")) - df["solution_tokens"]

        # Print summary table
        print("\n" + "=" * 50)
        print("Summary Table")
        print("=" * 50)
        print(f"{'Metric':<25} {'Value':<20}")
        print("-" * 50)
        print(f"{'MAE':<25} {mae:<20.4f}")
        print(f"{'Accuracy':<25} {acc:<20.4f}")
        print(f"{'Local Consistency':<25} {lc if lc is not None else 'N/A':<20}")
        print(f"{'Advantage Diff':<25} {adv if adv is not None else 'N/A':<20}")
        print("-" * 50)
        print(f"{'CoT Tokens (Min)':<25} {df['cot_tokens'].min():<20.1f}")
        print(f"{'CoT Tokens (Mean)':<25} {df['cot_tokens'].mean():<20.1f}")
        print(f"{'CoT Tokens (Max)':<25} {df['cot_tokens'].max():<20.1f}")
        print("-" * 50)
        print(f"{'Solution Tokens (Min)':<25} {df['solution_tokens'].min():<20.1f}")
        print(f"{'Solution Tokens (Mean)':<25} {df['solution_tokens'].mean():<20.1f}")
        print(f"{'Solution Tokens (Max)':<25} {df['solution_tokens'].max():<20.1f}")
        print("-" * 50)
        print(f"{'Grade Tokens (Mean)':<25} {tokens_mean:<20.1f}")
        print(f"{'Grade Tokens (95%)':<25} {tokens_95:<20.1f}")
        print(f"{'Grade Tokens (Max)':<25} {tokens_max:<20.1f}")
        print("=" * 50)

        print("\nScore Distribution:")
        print(df['graded_score'].value_counts().sort_index())
        exit(0)

    df = dfs[0].copy()
    id_col = 'question_id' if 'question_id' in df.columns else 'problem_id'
    if len(dfs) > 1 and "score" in df.columns:
        run_means = [df_i["score"].fillna(0).mean() for df_i in dfs]

        if args.merge_method == "mean":
            final_score = np.mean(run_means)
        elif args.merge_method == "max":
            final_score = np.max(run_means)
        elif args.merge_method == "min":
            final_score = np.min(run_means)

        std_at_k = np.std(run_means, ddof=1)

        # For Basic/Advanced breakdown, also do run-level aggregation
        ids0 = df[id_col].astype(str)
        if any("Basic" in str(q) for q in ids0):
            split_names = ["Basic", "Advanced"]

            def split_masks(ids):
                is_0 = ids.str.contains("Basic")
                return is_0, ~is_0
        else:
            split_names = ["2024/2025", "Others"]

            def split_masks(ids):
                is_0 = ids.str.contains("2024") | ids.str.contains("2025")
                return is_0, ~is_0

        split0_run_means = []
        split1_run_means = []
        for df_i in dfs:
            ids = df_i[id_col].astype(str)
            is_0, is_1 = split_masks(ids)
            split0_run_means.append(df_i[is_0]["score"].fillna(0).mean())
            split1_run_means.append(df_i[is_1]["score"].fillna(0).mean())

        split0_mean = nanmean_or_none(split0_run_means)
        split1_mean = nanmean_or_none(split1_run_means)

        # CoT tokens = cost_run.output_tokens - grade_cost.input_tokens
        # Solution tokens = grade_cost.input_tokens
        all_solution_tokens = pd.concat(
            [df_i["grade_cost"].apply(lambda x: x["input_tokens"]) for df_i in dfs],
            ignore_index=True,
        )
        all_run_output_tokens = pd.concat(
            [df_i["cost_run"].apply(lambda x: _get_cost_tokens(x, "output_tokens")) for df_i in dfs],
            ignore_index=True,
        )
        per_run_output_sums = [
            nansum_or_none(df_i["cost_run"].apply(lambda x: _get_cost_tokens(x, "output_tokens")))
            for df_i in dfs
        ]
        tok_sum_per_run = nanmean_or_none(per_run_output_sums)
        all_cot_tokens = all_run_output_tokens - all_solution_tokens
        all_scores = pd.concat(
            [df_i["score"].fillna(0) for df_i in dfs],
            ignore_index=True,
        )

        out_correct_min, out_correct_mean, out_correct_max = token_stats(
            all_run_output_tokens, all_scores >= 5
        )
        out_incorrect_min, out_incorrect_mean, out_incorrect_max = token_stats(
            all_run_output_tokens, all_scores < 5
        )

        def fmt_score(value):
            return "N/A" if value is None or np.isnan(value) else f"{value:.4f}"

        trunc_frac = truncated_frac(dfs)

        # Print summary table
        headers = [
            "Overall",
            "Std",
            split_names[0],
            split_names[1],
            "CoT Min",
            "CoT Mean",
            "CoT Max",
            "Sol Min",
            "Sol Mean",
            "Sol Max",
            "Tok Mean",
            "Tok Sum/Run",
            "TokC Min",
            "TokC Mean",
            "TokC Max",
            "TokI Min",
            "TokI Mean",
            "TokI Max",
            "Truncated Frac",
        ]
        colw = max(12, max(len(h) for h in headers) + 1)
        header_line = " ".join(f"{h:<{colw}}" for h in headers)
        values = [
            fmt_score(final_score),
            fmt_score(std_at_k),
            fmt_score(split0_mean),
            fmt_score(split1_mean),
            fmt_token(all_cot_tokens.min()),
            fmt_token(all_cot_tokens.mean()),
            fmt_token(all_cot_tokens.max()),
            fmt_token(all_solution_tokens.min()),
            fmt_token(all_solution_tokens.mean()),
            fmt_token(all_solution_tokens.max()),
            fmt_token(all_run_output_tokens.mean()),
            fmt_token(tok_sum_per_run),
            fmt_token(out_correct_min),
            fmt_token(out_correct_mean),
            fmt_token(out_correct_max),
            fmt_token(out_incorrect_min),
            fmt_token(out_incorrect_mean),
            fmt_token(out_incorrect_max),
            fmt_frac(trunc_frac),
        ]
        value_line = " ".join(f"{v:<{colw}}" for v in values)
        sep_len = max(len(header_line), len(value_line))

        print("\n" + "=" * sep_len)
        print(f"Summary Table ({args.merge_method}@{len(dfs)} runs)")
        print("-" * sep_len)
        print(header_line)
        print("-" * sep_len)
        print(value_line)

    else:
        # Single run case
        score_column = "score" if "score" in df.columns else "graded_score"
        if score_column not in df.columns:
            df[score_column] = 0
            df["grade_cost"] = [{"input_tokens": 0, "output_tokens": 0} for _ in range(len(df))]
            df["cost_run"] = [{"input_tokens": 0, "output_tokens": 0} for _ in range(len(df))]
        mean_score = df[score_column].mean()
        
        if any("Basic" in str(q) for q in df[id_col]):
            basic_mean = df[df[id_col].str.contains('Basic')][score_column].mean()
            advanced_mean = df[~df[id_col].str.contains('Basic')][score_column].mean()
            split_names = ['Basic', 'Advanced']
        else:
            basic_mean = df[df[id_col].str.contains('2024') | df[id_col].str.contains('2025')][score_column].mean()
            advanced_mean = df[~df[id_col].str.contains('2024') & ~df[id_col].str.contains('2025')][score_column].mean()
            split_names = ['2024/2025', 'Others']

        # CoT tokens = cost_run.output_tokens - grade_cost.input_tokens
        # Solution tokens = grade_cost.input_tokens
        df["solution_tokens"] = df["grade_cost"].apply(lambda x: x['input_tokens'])
        df["cot_tokens"] = df["cost_run"].apply(lambda x: _get_cost_tokens(x, "output_tokens")) - df["solution_tokens"]
        df["run_output_tokens"] = df["cost_run"].apply(lambda x: _get_cost_tokens(x, "output_tokens"))

        out_correct_min, out_correct_mean, out_correct_max = token_stats(
            df["run_output_tokens"], df[score_column].fillna(0) >= 6
        )
        out_incorrect_min, out_incorrect_mean, out_incorrect_max = token_stats(
            df["run_output_tokens"], df[score_column].fillna(0) < 6
        )

        trunc_frac = truncated_frac([df])

        # Print summary table
        headers = [
            "Overall",
            split_names[0],
            split_names[1],
            "CoT Min",
            "CoT Mean",
            "CoT Max",
            "Sol Min",
            "Sol Mean",
            "Sol Max",
            "Tok Mean",
            "Tok Sum/Run",
            "TokC Min",
            "TokC Mean",
            "TokC Max",
            "TokI Min",
            "TokI Mean",
            "TokI Max",
            "Truncated Frac",
        ]
        colw = max(12, max(len(h) for h in headers) + 1)
        header_line = " ".join(f"{h:<{colw}}" for h in headers)
        values = [
            f"{mean_score:.4f}",
            f"{basic_mean:.4f}",
            f"{advanced_mean:.4f}",
            fmt_token(df["cot_tokens"].min()),
            fmt_token(df["cot_tokens"].mean()),
            fmt_token(df["cot_tokens"].max()),
            fmt_token(df["solution_tokens"].min()),
            fmt_token(df["solution_tokens"].mean()),
            fmt_token(df["solution_tokens"].max()),
            fmt_token(df["run_output_tokens"].mean()),
            fmt_token(nansum_or_none(df["run_output_tokens"])),
            fmt_token(out_correct_min),
            fmt_token(out_correct_mean),
            fmt_token(out_correct_max),
            fmt_token(out_incorrect_min),
            fmt_token(out_incorrect_mean),
            fmt_token(out_incorrect_max),
            fmt_frac(trunc_frac),
        ]
        value_line = " ".join(f"{v:<{colw}}" for v in values)
        sep_len = max(len(header_line), len(value_line))

        print("\n" + "=" * sep_len)
        print("Summary Table (single run)")
        print("-" * sep_len)
        print(header_line)
        print("-" * sep_len)
        print(value_line)

    
