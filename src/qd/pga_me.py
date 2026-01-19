from typing import Any, Dict, Tuple

import jax
import numpy as np
import jax.numpy as jnp
from ribs.archives import CVTArchive
from ribs.emitters import IsoLineEmitter
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
    init_solutions = np.array(task.get_random_solution(2, init_key))

    # Initializations
    assert task.normalized_descriptors == True
    archive = CVTArchive(
        solution_dim=np.prod(task.solution_size),
        cells=config["population_size"],
        ranges=[(0, 1) for _ in range(task.descriptor_dim)],
        qd_score_offset=0.0,
        seed=config["seed"],
        use_kd_tree=task.descriptor_dim < 10,
    )

    iso_line_emitter = IsoLineEmitter(
        archive=archive,
        iso_sigma=config["iso_sigma"],
        line_sigma=config["line_sigma"],
        x0=init_solutions[0].reshape(-1),
        batch_size=config["batch_size"] // 2,
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
    }
    print("Starting optimization...")
    total_evals = 0

    # Add some batches of random solutions to begin with
    for _ in range(4):
        key, init_key, eval_key = jax.random.split(key, 3)
        solutions = np.array(
            task.get_random_solution(config["batch_size"] // 2, init_key)
        )
        eval_output = task.evaluate(solutions, eval_key, return_grad=False)
        fs, bs = np.asarray(eval_output.fitnesses), np.asarray(eval_output.descriptors)
        archive.add(solutions.reshape(solutions.shape[0], -1), fs, bs)
        total_evals += len(fs)

    for i in range(config["num_iterations"]):
        key, eval_key_1, eval_key_2, eval_key_3 = jax.random.split(key, 4)
        iso_solutions = iso_line_emitter.ask()
        reshaped_iso_solutions = jnp.asarray(iso_solutions).reshape(
            -1, *task.solution_size
        )
        iso_eval_output = task.evaluate(
            reshaped_iso_solutions, eval_key_1, return_grad=False
        )
        fs, bs = np.asarray(iso_eval_output.fitnesses), np.asarray(
            iso_eval_output.descriptors
        )
        archive.add(iso_solutions, fs, bs)
        total_evals += len(fs)

        g_solutions = archive.sample_elites(config["batch_size"] // 2)["solution"]
        reshaped_g_solutions = jnp.asarray(g_solutions).reshape(-1, *task.solution_size)
        g_eval_output = task.evaluate(
            reshaped_g_solutions, eval_key_2, return_grad=True
        )
        total_evals += len(g_solutions)
        updated_g_solutions = (
            reshaped_g_solutions
            + config["grad_step_size"] * g_eval_output.fitness_grads
        )
        g_eval_output = task.evaluate(
            updated_g_solutions, eval_key_3, return_grad=False
        )
        fs, bs = np.asarray(g_eval_output.fitnesses), np.asarray(
            g_eval_output.descriptors
        )
        updated_g_solutions = np.array(updated_g_solutions).reshape(
            updated_g_solutions.shape[0], -1
        )

        archive.add(updated_g_solutions, fs, bs)
        total_evals += len(fs)

        # Compute and record logs
        if i % config["log_frequency"] == 0:
            logs["total_evals"].append(total_evals)
            logs["fitnesses"].append(archive.data("objective"))
            logs["fitness_mean"].append(archive.stats.obj_mean)
            logs["fitness_std"].append(float(archive.data("objective").std()))
            logs["fitness_max"].append(archive.stats.obj_max)
            logs["fitness_min"].append(float(archive.data("objective").min()))
            logs["archive_size"].append(len(archive))
            logs["descriptors"].append(archive.data("measures"))
            dist_sq = np.sum(
                np.square(
                    archive.data("measures")[:, None, :]
                    - archive.data("measures")[None, :, :]
                ),
                axis=-1,
            )
            logs["avg_pairwise_distance"].append(
                float(np.mean(jnp.sqrt(dist_sq[np.triu_indices(dist_sq.shape[0], 1)])))
            )
            logs["qd_score"].append(archive.stats.qd_score)

            print(
                f"Iter {i:5d} | "
                f"QD Score: {archive.stats.qd_score:8.2f} | "
                f"Fitness (max/mean): {archive.stats.obj_max:6.2f}/{archive.stats.obj_mean:6.2f}"
            )

            if config["wandb"]["enable"]:
                wandb.log(
                    {
                        "iteration": i,
                        "qd_score": archive.stats.qd_score,
                        "fitness/mean": archive.stats.obj_mean,
                        "fitness/std": logs["fitness_std"][-1],
                        "fitness/max": archive.stats.obj_max,
                        "fitness/min": logs["fitness_min"][-1],
                        "avg_pairwise_dist": logs["avg_pairwise_distance"][-1],
                        "total_evals": total_evals,
                    }
                )

    print("Optimization finished.")

    return (
        archive.data("solution").reshape(-1, *task.solution_size),
        logs,
        {"archive_data": archive.data()},
    )
