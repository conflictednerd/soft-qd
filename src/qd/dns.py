"""
An implementation of Dominated Novelty Search (https://arxiv.org/abs/2502.00593) with
Iso-line and gradient-based emitters, based on the author's implementation:
https://github.com/adaptive-intelligent-robotics/Dominated-Novelty-Search/tree/main
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import wandb
from src.tasks.base import Task


class AdaptivePopulation(NamedTuple):
    """
    Class for the adaptive population.

    Args:
            solutions: Genotypes of the individuals in the population.
            fitnesses: Fitnesses of the individuals in the population.
            descriptors: Descriptors of the individuals in the population.
    """

    solutions: jax.Array
    fitnesses: jax.Array
    descriptors: jax.Array


@partial(jax.jit, static_argnames=("num_samples",))
def sample(population: AdaptivePopulation, num_samples: int, key) -> jax.Array:
    """Sample elements from a population.

    Args:
        population: the population to sample from
        num_samples: the number of elements to be sampled
        key: a jax PRNG random key

    Returns:
        samples: a batch of sampled solutions
    """
    return population.solutions[
        jax.random.choice(
            key, population.solutions.shape[0], shape=(num_samples,), replace=True
        )
    ]


@partial(jax.jit, static_argnames=("k",))
def add(
    population: AdaptivePopulation,
    new_solutions: jax.Array,
    new_fitnesses: jax.Array,
    new_descriptors: jax.Array,
    k: int,
) -> AdaptivePopulation:
    """Adds a batch of evaluated solutions to the population.

    Args:
        population: population to add to
        new_solutions: genotypes of the individuals to be considered
            for addition in the population.
        new_fitnesses: associated fitness.
        new_descriptors: associated descriptors.
        k: number of nearest neighbors to consider

    Returns:
        A new unstructured population where the relevant individuals have been
        added.
    """
    solutions = jnp.concatenate([population.solutions, new_solutions], axis=0)
    fitnesses = jnp.concatenate([population.fitnesses, new_fitnesses], axis=0)
    descriptors = jnp.concatenate([population.descriptors, new_descriptors], axis=0)
    # Fitter
    fitter = fitnesses[:, None] <= fitnesses[None, :]
    fitter = jnp.fill_diagonal(
        fitter, False, inplace=False
    )  # an individual can not be fitter than itself

    # Distance to k-fitter-nearest neighbors
    distance = jnp.linalg.norm(
        descriptors[:, None, :] - descriptors[None, :, :], axis=-1
    )
    distance = jnp.where(fitter, distance, jnp.inf)
    values, indices = jax.vmap(partial(jax.lax.top_k, k=k))(-distance)
    distance = jnp.mean(
        -values, where=jnp.take_along_axis(fitter, indices, axis=1), axis=-1
    )  # if number of fitter individuals is less than k, top_k will return at least one inf
    distance = jnp.where(
        jnp.isnan(distance), jnp.inf, distance
    )  # if no individual is fitter, set distance to inf

    # Sort by distance to k-fitter-nearest neighbors
    indices = jnp.argsort(distance, descending=True)
    # Select top N most fit
    indices = indices[: population.solutions.shape[0]]

    # Sort
    solutions = solutions[indices]
    descriptors = descriptors[indices]
    fitnesses = fitnesses[indices]

    return AdaptivePopulation(
        solutions=solutions,
        fitnesses=fitnesses,
        descriptors=descriptors,
    )


def isoline_variation(x, key, iso_sigma, line_sigma):
    """
    Iso+Line variation operator for a batch of solutions.

    Args:
        x: jnp.ndarray of shape (2*batch_size, dim, ...) Batch of solutions.
        key: jax random key
        iso_sigma: isotropic noise scale
        line_sigma: line noise scale

    Returns:
        offspring: jnp.ndarray of same shape as x
    """
    batch_size = x.shape[0] // 2
    x1, x2 = x[:batch_size], x[batch_size:]

    key, key_iso, key_line = jax.random.split(key, 3)

    iso_noise = jax.random.normal(key_iso, shape=x1.shape) * iso_sigma

    line_noise = jax.random.normal(key_line, shape=(batch_size,)) * line_sigma
    line_noise = line_noise.reshape((batch_size,) + (1,) * (x.ndim - 1))

    offspring = x1 + iso_noise + line_noise * (x2 - x1)
    return offspring


def _evaluate_all_solutions(
    task,
    solutions: jax.Array,
    key: jax.Array,
    batch_size: int,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """
    Evaluates solutions and returns their fitnesses and descriptors.
    Used for initializing the fitness and BD values
    """
    num_solutions = solutions.shape[0]
    all_fitnesses = []
    all_descriptors = []

    for i in range(0, num_solutions, batch_size):
        batch_solutions = solutions[i : i + batch_size]
        key, eval_key = jax.random.split(key)

        eval_output = task.evaluate(batch_solutions, eval_key, return_grad=False)

        all_fitnesses.append(jnp.clip(eval_output.fitnesses, eps))
        all_descriptors.append(eval_output.descriptors)

    fitnesses = jnp.concatenate(all_fitnesses)
    descriptors = jnp.concatenate(all_descriptors)

    return fitnesses, descriptors


def train(
    config: Dict[str, Any], task: Task
) -> Tuple[np.ndarray, Dict[str, list], Dict[str, Any]]:
    """Main training loop."""
    key = jax.random.PRNGKey(config["seed"])

    # Initializations
    key, init_key, init_eval_key = jax.random.split(key, 3)
    init_solutions = np.array(
        task.get_random_solution(config["population_size"], init_key)
    )
    init_fitnesses, init_descriptors = _evaluate_all_solutions(
        task, init_solutions, init_eval_key, config["eval_batch_size"]
    )
    population = AdaptivePopulation(
        solutions=init_solutions,
        fitnesses=init_fitnesses,
        descriptors=init_descriptors,
    )
    print(f"Task normalization is {task.normalized_descriptors}")

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
    }
    print("Starting optimization...")
    total_evals = 0
    for i in range(config["num_iterations"]):
        key, iso_sample_key, eval_key, iso_variation_key = jax.random.split(key, 4)

        iso_solutions = sample(
            population, 2 * config["isoline_batch_size"], iso_sample_key
        )
        iso_solutions = isoline_variation(
            iso_solutions, iso_variation_key, config["iso_sigma"], config["line_sigma"]
        )
        iso_eval_output = task.evaluate(iso_solutions, eval_key, return_grad=False)
        fs, bs = iso_eval_output.fitnesses, iso_eval_output.descriptors

        if config["use_grad"]:
            key, sample_key, eval_key = jax.random.split(key, 3)
            grad_solutions = sample(population, config["grad_batch_size"], sample_key)
            g_eval_output = task.evaluate(
                grad_solutions, jax.random.fold_in(eval_key, 1), return_grad=True
            )
            grad_solutions = (
                grad_solutions + config["grad_step_size"] * g_eval_output.fitness_grads
            )
            g_eval_output = task.evaluate(
                grad_solutions, jax.random.fold_in(eval_key, 2), return_grad=False
            )

            all_solutions = jnp.concatenate([iso_solutions, grad_solutions], axis=0)
            all_fs = jnp.concatenate([fs, g_eval_output.fitnesses], axis=0)
            all_bs = jnp.concatenate([bs, g_eval_output.descriptors], axis=0)
            population = add(population, all_solutions, all_fs, all_bs, config["k"])
            total_evals += len(fs) + 2 * len(grad_solutions)
        else:
            print("Adding to population...")
            population = add(population, iso_solutions, fs, bs, config["k"])
            total_evals += len(fs)

        # Compute and record logs
        if i % config["log_frequency"] == 0:
            logs["total_evals"].append(total_evals)
            logs["fitnesses"].append(np.array(population.fitnesses))
            logs["fitness_mean"].append(np.mean(population.fitnesses))
            logs["fitness_std"].append(np.std(population.fitnesses))
            logs["fitness_max"].append(population.fitnesses.max())
            logs["fitness_min"].append(population.fitnesses.min())
            logs["archive_size"].append(len(population.solutions))
            logs["descriptors"].append(np.array(population.descriptors))
            dist_sq = np.sum(
                np.square(
                    population.descriptors[:, None, :]
                    - population.descriptors[None, :, :]
                ),
                axis=-1,
            )
            logs["avg_pairwise_distance"].append(
                float(np.mean(jnp.sqrt(dist_sq[np.triu_indices(dist_sq.shape[0], 1)])))
            )

            print(
                f"Iter {i:5d} | "
                f"APD: {logs['avg_pairwise_distance'][-1]:.4f} | "
                f"Fitness (max/mean): {logs['fitness_max'][-1]:6.2f}/{logs['fitness_mean'][-1]:6.2f}"
            )

            if config["wandb"]["enable"]:
                wandb.log(
                    {
                        "iteration": i,
                        "fitness/mean": logs["fitness_mean"][-1],
                        "fitness/std": logs["fitness_std"][-1],
                        "fitness/max": logs["fitness_max"][-1],
                        "fitness/min": logs["fitness_min"][-1],
                        "avg_pairwise_dist": logs["avg_pairwise_distance"][-1],
                        "total_evals": total_evals,
                    }
                )

    print("Optimization finished.")

    return (
        np.array(population.solutions),
        logs,
        {
            "fitnesses": np.array(population.fitnesses),
            "descriptors": np.array(population.descriptors),
        },
    )
