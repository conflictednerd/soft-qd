"""
This code conducts a hyperparamter search for all algorithms on Rastrigin (easy, mid,
hard) tasks. It tries out 24 configurations for each algo-task pair.
Later, the generated logs should be evaluated and processed using
`_rastrigin_find_best_hparam.py` script.
"""

import subprocess
import itertools
import time
from collections import deque
from typing import List, Dict, Any
import psutil

# CONFIGURATION

MAX_PARALLEL_JOBS: int = 4

TASKS = ["rastrigin_easy", "rastrigin_mid", "rastrigin_hard"]

ALGOS = [
    {
        "name": "pga_me",
        "config_name": "pga_me.yaml",
        "hparams": {
            "iso_sigma": [0.1, 0.5, 1, 2],
            "line_sigma": [0.1, 0.2, 0.3],
            "grad_step_size": [0.05, 0.1],
        },
        "extra_args": {"num_iterations": 2000, "task.normalized_descriptors": "true"},
    },
    {
        "name": "cma_mega",
        "config_name": "cma_mega.yaml",
        "hparams": {
            "optim_lr": [0.05, 0.1, 0.5, 1.0],
            "sigma0": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "extra_args": {"num_iterations": 1000, "task.normalized_descriptors": "true"},
    },
    {
        "name": "cma_maega",
        "config_name": "cma_maega.yaml",
        "hparams": {
            "optim_lr": [0.05, 0.1, 0.5, 1.0],
            "sigma0": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "extra_args": {"num_iterations": 1000, "task.normalized_descriptors": "true"},
    },
    {
        "name": "softqd",
        "config_name": "config.yaml",
        "hparams": {
            "sigma_rule": [0.001, 0.005, 0.01, 0.05, 0.1, 1, 2, 4],
            "optimizer.learning_rate": [0.01, 0.05, 0.1],
        },
        "extra_args": {"num_iterations": 4000},
    },
    {
        "name": "cma_mae",
        "config_name": "cma_mae.yaml",
        "hparams": {
            "archive_lr": [0.001, 0.01, 0.05, 0.1],
            "sigma0": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "extra_args": {"num_iterations": 1000, "task.normalized_descriptors": "true"},
    },
]


def generate_commands() -> List[str]:
    """Generates all command strings for the hyperparameter search."""
    commands = []
    for algo in ALGOS:
        hparam_names = list(algo["hparams"].keys())
        hparam_value_lists = list(algo["hparams"].values())

        for hparam_values in itertools.product(*hparam_value_lists):
            hparam_dict = dict(zip(hparam_names, hparam_values))

            hparam_dict.update(algo.get("extra_args", {}))

            # Format the hyperparameters into a command-line string
            hparam_str = " ".join(
                f"{key}={value}" for key, value in hparam_dict.items()
            )

            for task in TASKS:
                command = (
                    f'python -m src.main --config-name="{algo["config_name"]}" '
                    f"task={task} {hparam_str}"
                )
                commands.append(command)
    return commands


def run_in_parallel(commands_to_run: List[str]) -> List[str]:
    """Executes commands in parallel, respecting the MAX_PARALLEL_JOBS limit."""
    command_queue = deque(commands_to_run)
    total_commands = len(commands_to_run)
    completed_count = 0

    active_processes: List[Dict[str, Any]] = []
    failed_commands: List[str] = []

    print(f"Progress: {completed_count}/{total_commands}")

    while command_queue or active_processes:
        # Launch new processes if there's capacity and commands are available
        while len(active_processes) < MAX_PARALLEL_JOBS and command_queue:
            current_ram_percent = psutil.virtual_memory().percent
            if current_ram_percent < 85.0:
                cmd = command_queue.popleft()
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                active_processes.append({"process": process, "cmd": cmd})
                print(f"Launched new process. RAM usage: {current_ram_percent:.1f}%")
                time.sleep(10)
            else:
                # If RAM is high, stop trying to launch and wait
                print(
                    f"Pausing launches. RAM usage at {current_ram_percent:.1f}% >= 85%",
                    " " * 20,
                )
                break  # Exit the launch loop to check for finished jobs

        # Check for and manage completed processes
        remaining_processes = []
        for p_info in active_processes:
            process, cmd = p_info["process"], p_info["cmd"]
            if process.poll() is None:  # None means the process is still running
                remaining_processes.append(p_info)
            else:  # The process has finished
                completed_count += 1
                print(f"Progress: {completed_count}/{total_commands}")
                if process.returncode != 0:
                    failed_commands.append(cmd)

        active_processes = remaining_processes

        # Pause briefly to prevent the loop from consuming too much CPU
        time.sleep(10)

    return failed_commands


def main():
    """Main function to generate, run, and report the search."""
    print("Hyperparameter Search Script")

    all_commands = generate_commands()
    print(f"Generated a total of {len(all_commands)} commands to run.")

    print(f"\nStarting execution with up to {MAX_PARALLEL_JOBS} parallel jobs...")
    failed_runs = run_in_parallel(all_commands)

    print("\nExecution Complete")
    total_runs = len(all_commands)
    num_failed = len(failed_runs)
    num_success = total_runs - num_failed

    print(f"Successful runs: {num_success}/{total_runs}")
    print(f"Failed runs:     {num_failed}/{total_runs}")

    if failed_runs:
        log_file = "failed_runs.log"
        print(f"\nSaving {num_failed} failed command(s) to '{log_file}'.")
        with open(log_file, "w") as f:
            for cmd in failed_runs:
                f.write(f"{cmd}\n")
        print("Done.")


if __name__ == "__main__":
    main()
