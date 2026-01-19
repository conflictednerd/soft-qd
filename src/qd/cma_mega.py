from typing import Any, Dict, Tuple

import jax
import numpy as np
import jax.numpy as jnp
from ribs.archives import CVTArchive
from ribs.emitters import GradientArborescenceEmitter
from ribs.schedulers import Scheduler
import wandb
from src.tasks.base import Task


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
        qd_score_offset=0.0,
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
            grad_opt=config["grad_opt"],
            batch_size=config["batch_size"] - 1,
            seed=i,
        )
        for i in range(config["num_emitters"])
    ]

    scheduler = Scheduler(archive=training_archive, emitters=emitters)

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
    }
    print("Starting optimization...")
    total_evals = 0
    for i in range(config["num_iterations"]):
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
        scheduler.tell_dqd(
            np.array(eval_output.fitnesses),
            np.array(eval_output.descriptors),
            np.array(jacobians),
        )
        total_evals += bsz

        # 2. ask() -> tell()
        solutions = jnp.asarray(scheduler.ask()).reshape(-1, *task.solution_size)
        bsz = solutions.shape[0]
        eval_output = task.evaluate(solutions, eval_key_2, return_grad=False)
        fs, bs = np.array(eval_output.fitnesses), np.array(eval_output.descriptors)
        scheduler.tell(fs, bs)
        total_evals += bsz

        # Compute and record logs
        if i % config["log_frequency"] == 0:
            logs["total_evals"].append(total_evals)
            logs["fitnesses"].append(training_archive.data("objective"))
            logs["fitness_mean"].append(training_archive.stats.obj_mean)
            logs["fitness_std"].append(training_archive.data("objective").std())
            logs["fitness_max"].append(training_archive.stats.obj_max)
            logs["fitness_min"].append(float(training_archive.data("objective").min()))
            logs["archive_size"].append(len(training_archive))
            logs["descriptors"].append(training_archive.data("measures"))
            dist_sq = np.sum(
                np.square(
                    training_archive.data("measures")[:, None, :]
                    - training_archive.data("measures")[None, :, :]
                ),
                axis=-1,
            )
            logs["avg_pairwise_distance"].append(
                float(np.mean(jnp.sqrt(dist_sq[np.triu_indices(dist_sq.shape[0], 1)])))
            )
            logs["qd_score"].append(training_archive.stats.qd_score)

            print(
                f"Iter {i:5d} | "
                f"QD Score: {training_archive.stats.qd_score:8.2f} | "
                f"Fitness (max/mean): {training_archive.stats.obj_max:6.2f}/{training_archive.stats.obj_mean:6.2f}"
            )

            if config["wandb"]["enable"]:
                wandb.log(
                    {
                        "iteration": i,
                        "qd_score": training_archive.stats.qd_score,
                        "fitness/mean": training_archive.stats.obj_mean,
                        "fitness/std": training_archive.data("objective").std(),
                        "fitness/max": training_archive.stats.obj_max,
                        "fitness/min": logs["fitness_min"][-1],
                        "avg_pairwise_dist": logs["avg_pairwise_distance"][-1],
                        "total_evals": total_evals,
                    }
                )

    print("Optimization finished.")

    return (
        training_archive.data("solution").reshape(-1, *task.solution_size),
        logs,
        {"archive_data": training_archive.data()},
    )
