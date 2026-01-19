"""
This assumes that all rastrigin hparam search experiments (on three task types of easy,
mid, hard) are in the outputs directory in the conventional log format. It goes over
all of the logs and for each algo-task pair, finds the config that got the best qd score.
The result will be saved in `best_configs.json`.

This was done to analyze the results of rastrigin_hparam_search.py and to find the
best hparams for each algorithm on the rastrigin domain.
"""

import json
import yaml
from pathlib import Path
from collections import defaultdict
import sys


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flattens a nested dictionary. For example:
    {'a': {'b': 1}} becomes {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    """
    Main function to find experiments, aggregate results, calculate statistics,
    and save them to JSON files.
    Considers all experiments in the outputs directory that start with a number.
    """
    outputs_dir = Path("outputs")
    if not outputs_dir.is_dir():
        print(f"Error: Directory '{outputs_dir}' not found.")
        sys.exit(1)

    # best_hparams[algo] = (best_qd_score, exp_dir, best_params)
    best_hparams = defaultdict(lambda: (-9e9, "", None))

    print("Searching for experiment results...")
    experiment_dirs = []
    for date_dir in outputs_dir.iterdir():
        # Only consider directories that start with a number (e.g., "2025-08-16")
        if date_dir.is_dir() and date_dir.name[0].isdigit():
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir():
                    experiment_dirs.append(time_dir)

    if not experiment_dirs:
        print("No experiment directories found.")
        return

    print(f"Found {len(experiment_dirs)} experiments. Aggregating data...")

    # Loop through each experiment directory to extract data
    for exp_dir in experiment_dirs:
        try:
            # Load the algorithm name from the config file
            config_path = exp_dir / ".hydra" / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            algo_name = config.get("algo_name", "unknown_algo")
            task_type = config["task"]["task_type"]

            # Load the metrics from the results file
            metrics_path = exp_dir / "evaluation_results" / "metrics.json"
            with metrics_path.open("r") as f:
                metrics = json.load(f)

            qd_score = metrics["qd_metrics"]["qd_score"]

            if qd_score > best_hparams[f"{algo_name}_{task_type}"][0]:
                best_hparams[f"{algo_name}_{task_type}"] = (
                    qd_score,
                    str(exp_dir),
                    config,
                )

        except FileNotFoundError as e:
            print(f"Skipping directory {exp_dir.name}: {e.filename} not found.")
        except Exception as e:
            print(f"Skipping directory {exp_dir.name} due to an error: {e}")

    best_configs_file = "best_configs.json"
    with open(best_configs_file, "w") as f:
        json.dump(best_hparams, f, indent=4)

    print("\nDone!")


if __name__ == "__main__":
    main()
