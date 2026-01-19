import time
from typing import Any, Dict, Tuple

import jax
import numpy as np
import jax.numpy as jnp
from ribs.archives import CVTArchive
from ribs.emitters import GradientArborescenceEmitter
from ribs.schedulers import Scheduler
import wandb
from src.tasks.base import Task

# TODO: If evaluation required too much memory at once, we can evaluate a batch of
# solutions in smaller mini-batches.
# NOTE: For running LSI: python -m src.main --config-name cma_maega.yaml task=lsi num_emitters=1 archive_lr=0.02 optim_lr=0.05 num_iterations=2000 population_siztask.normalized_descriptors=true


def train(
    config: Dict[str, Any], task: Task
) -> Tuple[np.ndarray, Dict[str, list], Dict[str, Any]]:
    """Main training loop."""
    # TODO: Set numpy/jax seed in the main function
    key = jax.random.PRNGKey(config["seed"])

    # Initialize solutions for emitters
    key, init_key = jax.random.split(key)
    init_solutions = np.array(
        task.get_random_solution(config["num_emitters"], init_key)
    )

    # Initializations
    assert task.normalized_descriptors == True
    training_archive = CVTArchive(
        solution_dim=np.prod(task.solution_size),
        cells=config["population_size"],
        ranges=[(0, 1) for _ in range(task.descriptor_dim)],
        learning_rate=config["archive_lr"],
        threshold_min=0.0,
        qd_score_offset=0.0,
        use_kd_tree=task.descriptor_dim < 10,
        seed=config["seed"],
        dtype=np.float32,
    )

    result_archive = CVTArchive(
        solution_dim=np.prod(task.solution_size),
        cells=config["population_size"],
        ranges=[(0, 1) for _ in range(task.descriptor_dim)],
        qd_score_offset=0.0,
        custom_centroids=training_archive.centroids.copy(),  # Same centroids as train archive
        use_kd_tree=task.descriptor_dim < 10,
        seed=config["seed"],
        dtype=np.float32,
    )

    emitters = [
        GradientArborescenceEmitter(
            archive=training_archive,
            x0=init_solutions[i].reshape(-1),
            sigma0=config["sigma0"],
            lr=config["optim_lr"],  # learning rate for gradient optimizer
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            grad_opt=config["grad_opt"],
            batch_size=config["batch_size"] - 1,
            seed=i,
        )
        for i in range(config["num_emitters"])
    ]

    scheduler = Scheduler(
        archive=training_archive, emitters=emitters, result_archive=result_archive
    )

    # Training
    logs = {
        "total_evals": [],
        "fitnesses": [],
        "fitness_mean": [],
        "fitness_std": [],
        "fitness_max": [],
        "fitness_min": [],
        "archive_size": [],
        "descriptors": [],
        "avg_pairwise_distance": [],
        "qd_score": [],
        "grad_eval_time": [],
        "dqd_tell_time": [],
        "regular_eval_time": [],
        "regular_tell_time": [],
    }
    print("Starting optimization...")
    total_evals = 0
    for i in range(config["num_iterations"]):
        t = time.perf_counter()
        key, eval_key, eval_key_2 = jax.random.split(key, 3)

        # 1. ask_dqd() -> tell_dqd()
        solutions = jnp.asarray(scheduler.ask_dqd()).reshape(-1, *task.solution_size)
        bsz = solutions.shape[0]
        eval_output = task.evaluate(solutions, eval_key, return_grad=True)
        jacobians = jnp.concatenate(
            (
                eval_output.fitness_grads.reshape(bsz, 1, -1),
                eval_output.descriptor_grads.reshape(bsz, task.descriptor_dim, -1),
            ),
            axis=1,
        )
        jacobians.block_until_ready()
        logs["grad_eval_time"].append(time.perf_counter() - t)
        t = time.perf_counter()
        scheduler.tell_dqd(
            np.array(eval_output.fitnesses),
            np.array(eval_output.descriptors),
            np.array(jacobians),
        )
        logs["dqd_tell_time"].append(time.perf_counter() - t)
        t = time.perf_counter()
        total_evals += bsz

        # 2. ask() -> tell()
        solutions = jnp.asarray(scheduler.ask()).reshape(-1, *task.solution_size)
        bsz = solutions.shape[0]
        eval_output = task.evaluate(solutions, eval_key_2, return_grad=False)
        eval_output.fitnesses.block_until_ready()
        fs, bs = np.array(eval_output.fitnesses), np.array(eval_output.descriptors)
        logs["regular_eval_time"].append(time.perf_counter() - t)
        t = time.perf_counter()
        scheduler.tell(fs, bs)
        logs["regular_tell_time"].append(time.perf_counter() - t)
        t = time.perf_counter()
        total_evals += bsz

        # Compute and record logs
        if i % config["log_frequency"] == 0:
            logs["total_evals"].append(total_evals)
            logs["fitnesses"].append(result_archive.data("objective"))
            logs["fitness_mean"].append(result_archive.stats.obj_mean)
            logs["fitness_std"].append(result_archive.data("objective").std())
            logs["fitness_max"].append(result_archive.stats.obj_max)
            logs["fitness_min"].append(float(result_archive.data("objective").min()))
            logs["archive_size"].append(len(result_archive))
            logs["descriptors"].append(result_archive.data("measures"))
            dist_sq = np.sum(
                np.square(
                    result_archive.data("measures")[:, None, :]
                    - result_archive.data("measures")[None, :, :]
                ),
                axis=-1,
            )
            logs["avg_pairwise_distance"].append(
                float(np.mean(jnp.sqrt(dist_sq[np.triu_indices(dist_sq.shape[0], 1)])))
            )
            logs["qd_score"].append(result_archive.stats.qd_score)

            print(
                f"Iter {i:5d} | "
                f"QD Score: {result_archive.stats.qd_score:8.2f} | "
                f"Fitness (max/mean): {result_archive.stats.obj_max:6.2f}/{result_archive.stats.obj_mean:6.2f}"
            )

            if config["wandb"]["enable"]:
                wandb.log(
                    {
                        "iteration": i,
                        "qd_score": result_archive.stats.qd_score,
                        "fitness/mean": result_archive.stats.obj_mean,
                        "fitness/std": result_archive.data("objective").std(),
                        "fitness/max": result_archive.stats.obj_max,
                        "fitness/min": logs["fitness_min"][-1],
                        "avg_pairwise_dist": logs["avg_pairwise_distance"][-1],
                        "total_evals": total_evals,
                        "time/grad_eval_time": logs["grad_eval_time"][-1],
                        "time/regular_eval_time": logs["regular_eval_time"][-1],
                        "time/dqd_tell_time": logs["dqd_tell_time"][-1],
                        "time/regular_tell_time": logs["regular_tell_time"][-1],
                    }
                )

    print("Optimization finished.")

    return (
        result_archive.data("solution").reshape(-1, *task.solution_size),
        logs,
        {"archive_data": result_archive.data()},
    )
