"""
This code runs all Rastrigin experiments (easy, mid, hard variants) for all algorithms.
It runs each algorithm-task pair with 10 seeds. Optimal hyperparameters, found via
`_rastrigin_hparam_search.py` are used.
"""

import copy
import subprocess
import time
from collections import deque
from typing import Any, Dict, List

import psutil

# CONFIGURATION

MAX_PARALLEL_JOBS: int = 4

TASKS = ["rastrigin_easy", "rastrigin_mid", "rastrigin_hard"]
SEEDS = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010)

ALGOS = [
    {
        "name": "softqd",
        "config_name": "config.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 1000,
                "sigma_rule": 0.1,
                "optimizer.learning_rate": 0.05,
            },
            "rastrigin_mid": {
                "num_iterations": 1000,
                "sigma_rule": 0.5,
                "optimizer.learning_rate": 0.05,
            },
            "rastrigin_hard": {
                "num_iterations": 1000,
                "sigma_rule": 1.0,
                "optimizer.learning_rate": 0.05,
            },
        },
    },
    {
        "name": "pga_me",
        "config_name": "pga_me.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 8000,
                "task.normalized_descriptors": "true",
                "iso_sigma": 2.0,
                "line_sigma": 0.2,
                "grad_step_size": 0.1,
            },
            "rastrigin_mid": {
                "num_iterations": 8000,
                "task.normalized_descriptors": "true",
                "iso_sigma": 2.0,
                "line_sigma": 0.1,
                "grad_step_size": 0.1,
            },
            "rastrigin_hard": {
                "num_iterations": 8000,
                "task.normalized_descriptors": "true",
                "iso_sigma": 0.5,
                "line_sigma": 0.3,
                "grad_step_size": 0.1,
            },
        },
    },
    {
        "name": "cma_mega",
        "config_name": "cma_mega.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "optim_lr": 0.5,
                "sigma0": 2.0,
            },
            "rastrigin_mid": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "optim_lr": 0.1,
                "sigma0": 10.0,
            },
            "rastrigin_hard": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "optim_lr": 0.1,
                "sigma0": 10.0,
            },
        },
    },
    {
        "name": "cma_maega",
        "config_name": "cma_maega.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.01,
                "optim_lr": 0.05,
                "sigma0": 0.5,
            },
            "rastrigin_mid": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.01,
                "optim_lr": 0.05,
                "sigma0": 10.0,
            },
            "rastrigin_hard": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.01,
                "optim_lr": 0.05,
                "sigma0": 0.5,
            },
        },
    },
    {
        "name": "cma_mae",
        "config_name": "cma_mae.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.05,
                "sigma0": 2.0,
            },
            "rastrigin_mid": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.01,
                "sigma0": 5.0,
            },
            "rastrigin_hard": {
                "num_iterations": 1900,
                "task.normalized_descriptors": "true",
                "archive_lr": 0.01,
                "sigma0": 2.0,
            },
        },
    },
    {
        "name": "dns",
        "config_name": "dns.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 8000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": False,
                "task.normalized_descriptors": True,
            },
            "rastrigin_mid": {
                "num_iterations": 8000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": False,
                "task.normalized_descriptors": True,
            },
            "rastrigin_hard": {
                "num_iterations": 8000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": False,
                "task.normalized_descriptors": True,
            },
        },
    },
    {
        "name": "dns_grad",
        "config_name": "dns.yaml",
        "args": {
            "rastrigin_easy": {
                "num_iterations": 4000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": True,
                "task.normalized_descriptors": True,
            },
            "rastrigin_mid": {
                "num_iterations": 4000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": True,
                "task.normalized_descriptors": True,
            },
            "rastrigin_hard": {
                "num_iterations": 4000,
                "k": 8,
                "line_sigma": 0.5,
                "iso_sigma": 0.05,
                "use_grad": True,
                "task.normalized_descriptors": True,
            },
        },
    },
]


def generate_commands() -> List[str]:
    """Generates all command strings for the hyperparameter search."""
    commands = []
    for seed in SEEDS:
        for task in TASKS:
            for algo in ALGOS:
                args_dict = copy.deepcopy(algo["args"][task])
                args_dict["seed"] = seed
                args_str = " ".join(
                    f"{key}={value}" for key, value in args_dict.items()
                )
                command = f"python -m src.main --config-name={algo['config_name']} task={task} {args_str}"
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
    with open("./.all_commands.txt", "w") as f:
        f.write("\n".join(all_commands))
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
