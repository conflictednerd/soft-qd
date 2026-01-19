from typing import Any, Dict, Tuple

import jax
import numpy as np
import jax.numpy as jnp
from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
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
        learning_rate=config["archive_lr"],
        threshold_min=0.0,
        seed=config["seed"],
        use_kd_tree=task.descriptor_dim < 10,
    )
    result_archive = CVTArchive(
        solution_dim=np.prod(task.solution_size),
        cells=config["population_size"],
        ranges=[(0, 1) for _ in range(task.descriptor_dim)],
        qd_score_offset=0.0,
        custom_centroids=training_archive.centroids.copy(),  # Same centroids as train archive
        seed=config["seed"],
        use_kd_tree=task.descriptor_dim < 10,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=training_archive,
            x0=init_solutions[i].reshape(-1),
            sigma0=config["sigma0"],
            ranker="imp",
            es="sep_cma_es" if config["use_separable"] else "cma_es",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=config["batch_size"],
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
    }
    print("Starting optimization...")
    total_evals = 0
    for i in range(config["num_iterations"]):
        key, eval_key = jax.random.split(key)
        solutions = jnp.asarray(scheduler.ask())
        reshaped_solutions = solutions.reshape(-1, *task.solution_size)
        eval_output = task.evaluate(reshaped_solutions, eval_key, return_grad=False)
        fs, bs = np.asarray(eval_output.fitnesses), np.asarray(eval_output.descriptors)
        scheduler.tell(fs, bs)
        total_evals += len(fs)

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
                    }
                )

    print("Optimization finished.")

    return (
        result_archive.data("solution").reshape(-1, *task.solution_size),
        logs,
        {"archive_data": result_archive.data()},
    )
