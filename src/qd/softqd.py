import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple, Dict, Any, Callable, Tuple, Union
from functools import partial
import wandb

from src.tasks.base import Task


class TrainState(NamedTuple):
    """State of the optimization process."""

    solutions: jax.Array
    fitnesses: jax.Array
    descriptors: jax.Array
    opt_state: optax.OptState
    sigma_sq: float
    key: jax.Array


def population_optimizer(
    optimizer_fn: Callable[[], optax.GradientTransformation], population_size: int
) -> optax.GradientTransformation:
    """Wraps an Optax optimizer to operate on a population of models.

    This function uses `jax.vmap` to create an optimizer that can update an entir
    population of model parameters simultaneously. It assumes that the parameters
    for the whole population are stacked into a single JAX array with a leading batch
    dimension, i.e., shape `(population_size, ...)`.

    The returned optimizer's `update` function supports an optional `indices` argument
    to perform a sparse update on only a subset of the population.

    Args:
        optimizer_fn: A function that returns a base `optax`
            optimizer instance, e.g., `lambda: optax.adam(1e-3)`.
        population_size: The total number of models in the population.

    Returns:
        An `optax.GradientTransformation` tailored for population-based updates.
    """
    base_optimizer = optimizer_fn()

    v_init = jax.vmap(base_optimizer.init)
    v_update = jax.vmap(base_optimizer.update)

    def init_fn(params: jax.Array) -> optax.OptState:
        """Initializes the optimizer state for the entire population."""
        return v_init(params)

    def update_fn(
        gradients: jax.Array,
        state: optax.OptState,
        params: jax.Array | None = None,
        *,
        indices: jax.Array | None = None,
    ) -> tuple[jax.Array, optax.OptState]:
        """Performs a full or partial update on the population's parameters."""
        if indices is None:
            updates, new_state = v_update(gradients, state, params)
            return updates, new_state
        else:
            selected_params = params[indices] if params is not None else None
            selected_state = jax.tree.map(lambda x: x[indices], state)

            selected_updates, new_selected_state = v_update(
                gradients, selected_state, selected_params
            )

            # Create a full-sized zero-update tensor and scatter the computed updates.
            full_updates = (
                jnp.zeros(
                    (population_size, *gradients.shape[1:]), dtype=gradients.dtype
                )
                .at[indices]
                .set(selected_updates)
            )

            # Scatter the new states for the selection back into the full state pytree.
            new_full_state = jax.tree.map(
                lambda full, new: full.at[indices].set(new),
                state,
                new_selected_state,
            )

            return full_updates, new_full_state

    return optax.GradientTransformation(init_fn, update_fn)


@partial(
    jax.jit,
    static_argnames=("task_evaluate", "num_neighbors", "eps"),
)
def _compute_gradient(
    solutions: jax.Array,
    fitnesses: jax.Array,
    descriptors: jax.Array,
    batch_indices: jax.Array,
    sigma_sq: float,
    task_evaluate: Callable,
    key: jax.Array,
    num_neighbors: int,  # Number of nearest neighbors considered in repulsive term. If -1, all solutions are considered.
    eps: float = 1e-6,
) -> jax.Array:
    """Computes the gradient of the objective A(theta) with respect to a batch of solutions."""
    k, *d = solutions.shape
    if num_neighbors == -1:
        num_neighbors = k

    # 1. Evaluate batch of solutions to get fitness, descriptors, and base gradients
    eval_output = task_evaluate(solutions[batch_indices], key)
    batch_f = eval_output.fitnesses  # Shape: (bsz, )
    batch_b = eval_output.descriptors  # Shape: (bsz, m)
    batch_grad_f = eval_output.fitness_grads  # Shape: (bsz, d)
    batch_grad_b = eval_output.descriptor_grads  # Shape: (bsz, m, d)

    f = fitnesses.at[batch_indices].set(batch_f)
    b = descriptors.at[batch_indices].set(batch_b)

    # 2. Compute pairwise components for the repulsive term
    # Pairwise distances in behavior space: ||b_i - b_j||^2
    b_batch_diff = batch_b[:, None, :] - b[None, :, :]  # Shape: (bsz, k, m)
    dist_sq = jnp.sum(jnp.square(b_batch_diff), axis=-1)  # Shape: (bsz, k)

    # 2.5 Use topk closest solutions for the repulsive force
    _, top_indices = jax.lax.top_k(-dist_sq, k=num_neighbors)
    f_nns = f[top_indices]
    b_batch_diff = jnp.take_along_axis(
        b_batch_diff, top_indices[:, :, None], axis=1
    )  # Shape: (bsz, nns, m)
    dist_sq = jnp.take_along_axis(dist_sq, top_indices, axis=1)  # Shape: (bsz, nns)

    # Kernel matrix: exp(-dist^2 / sigma^2)
    # Remove self-interactions and negligibles
    kernel = jnp.exp(-dist_sq / sigma_sq)  # Shape: (bsz, nns)
    is_self = batch_indices[:, None] == top_indices  # Shape: (bsz, nns)
    is_negligible = kernel < eps
    kernel = jnp.where(jnp.logical_or(is_self, is_negligible), 0, kernel)

    # Pairwise fitness term: sqrt(f_i * f_j)
    f_sqrt_prod = jnp.sqrt(
        jnp.clip(batch_f[:, None], 0) * jnp.clip(f_nns, 0)
    )  # Shape: (bsz, nns)

    # 3. Compute the gradient of the repulsive part for the selected batch
    # Compute the scalar factor for the fitness gradient term
    # x[b] = 0.5 sum_j sqrt(f_j/f_b) * kernel(b, j)
    scalar_f_grad_factor = 0.5 * jnp.sum(
        jnp.sqrt(jnp.clip(f_nns, 0) / jnp.clip(batch_f[:, None], eps)) * kernel, axis=1
    )  # Shape: (bsz, )
    scalar_f_grad_factor *= (batch_f > 0).astype(batch_f.dtype)
    # Scale the gradient of each solution b by its corresponding scalar factor
    grad_f_term = jnp.einsum("b,b...->b...", scalar_f_grad_factor, batch_grad_f)

    # Grad of repulsive term w.r.t. behaviors b_k
    # Proportional to: -2/sigma^2 * sqrt(f_i*f_j) * K_ij * (b_k - b_j)
    batch_grad_repulsion_b_scalar = (
        (-2.0 / sigma_sq) * f_sqrt_prod * kernel
    )  # Shape: (bsz, nns)
    # Contract with (b_k - b_j) and the jacobian grad_b
    grad_b_term = jnp.einsum(
        "bn,bnm,bm...->b...",
        batch_grad_repulsion_b_scalar,
        b_batch_diff,
        batch_grad_b,
    )  # b: batch, n: neighbors, m: descriptor dim

    # Total gradient of the repulsive sum
    grad_repulsion_total = grad_f_term + grad_b_term

    # 4. Final Gradient: grad(A) = grad(sum f_k) - grad(sum R_ij)
    # The first term is simply the sum of individual fitness gradients.
    total_gradient = batch_grad_f - grad_repulsion_total / 2
    # VERY IMPORTANT: To maximize the the archive score, we minimize it's negation
    total_gradient = -total_gradient

    return total_gradient


@partial(
    jax.jit,
    static_argnames=(
        "task_evaluate",
        "optimizer_update",
        "num_neighbors",
    ),
)
def _train_step(
    state: TrainState,
    task_evaluate: Callable,
    optimizer_update: Callable,
    batch_indices: jax.Array,
    num_neighbors: int,
) -> TrainState:
    """A single JIT-compiled training step."""
    key, eval_key, re_eval_key = jax.random.split(state.key, 3)

    # Compute the gradient of the SoftQD objective
    batch_gradient = _compute_gradient(
        state.solutions,
        state.fitnesses,
        state.descriptors,
        batch_indices,
        state.sigma_sq,
        task_evaluate,
        eval_key,
        num_neighbors,
    )

    updates, new_opt_state = optimizer_update(
        batch_gradient, state.opt_state, indices=batch_indices
    )
    new_solutions = optax.apply_updates(state.solutions, updates)

    # Re-evaluate the solutions that were updated to get their new fitness/descriptors
    #   This is needed so that future updates use correct data.
    re_eval_output = task_evaluate(new_solutions[batch_indices], re_eval_key)
    new_fitnesses = state.fitnesses.at[batch_indices].set(re_eval_output.fitnesses)
    new_descriptors = state.descriptors.at[batch_indices].set(
        re_eval_output.descriptors
    )

    return TrainState(
        solutions=new_solutions,
        fitnesses=new_fitnesses,
        descriptors=new_descriptors,
        opt_state=new_opt_state,
        sigma_sq=state.sigma_sq,
        key=key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "task_evaluate",
        "optimizer_update",
        "population_size",
        "batch_size",
        "num_neighbors",
        "sigma_rule",
    ),
)
def _train_epoch(
    state: TrainState,
    task_evaluate: Callable,
    optimizer_update: Callable,
    key: jax.Array,
    population_size: int,
    batch_size: int,
    num_neighbors: int,
    sigma_rule: Union[str, float],
    eps: float = 1e-6,
) -> TrainState:
    batched_indices = jax.random.permutation(key, population_size).reshape(
        -1, batch_size
    )

    final_state, _ = jax.lax.scan(
        lambda carry, batch_inds: (
            _train_step(
                carry,
                task_evaluate,
                optimizer_update,
                batch_inds,
                num_neighbors,
            ),
            None,
        ),
        state,
        batched_indices,
    )

    if sigma_rule in ["nn", "inverse_nn"]:
        # Update sigma^2 using mean Nearest-Neighbor distance

        b_diff = state.descriptors[:, None, :] - state.descriptors[None, :, :]
        dist_sq = jnp.sum(jnp.square(b_diff), axis=-1)  # Shape: (k, k)
        nns = jnp.min(
            jnp.where(jnp.eye(population_size, dtype=bool), jnp.inf, dist_sq), axis=1
        )
        new_sigma_sq = jnp.clip(jnp.nanmean(nns), eps)
        if sigma_rule == "inverse_nn":
            new_sigma_sq = 1 / new_sigma_sq
        final_state = final_state._replace(sigma_sq=new_sigma_sq)

    return final_state


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
    assert (
        config["population_size"] % config["batch_size"] == 0
    ), f"Population size ({config['population_size']}) must be divisible by batch size ({config['batch_size']})"
    key = jax.random.PRNGKey(config["seed"])

    # Initialize solutions
    key, init_state_key, init_key, init_eval_key = jax.random.split(key, 4)
    init_solutions = task.get_random_solution(config["population_size"], init_key)
    init_fitnesses, init_descriptors = _evaluate_all_solutions(
        task, init_solutions, init_eval_key, config["batch_size"]
    )

    # Initialize optimizer
    optimizer = population_optimizer(
        lambda: optax.adam(config["optimizer"]["learning_rate"]),
        config["population_size"],
    )
    opt_state = optimizer.init(init_solutions)

    # Initial state
    state = TrainState(
        solutions=init_solutions,
        fitnesses=init_fitnesses,
        descriptors=init_descriptors,
        opt_state=opt_state,
        sigma_sq=1.0 if isinstance(config["sigma_rule"], str) else config["sigma_rule"],
        key=init_state_key,
    )

    jitted_epoch = partial(
        _train_epoch,
        task_evaluate=task.evaluate,
        optimizer_update=optimizer.update,
        population_size=config["population_size"],
        batch_size=config["batch_size"],
        num_neighbors=config["num_neighbors"],
        sigma_rule=config["sigma_rule"],
    )

    logs = {
        "total_evals": [],
        "fitnesses": [],
        "fitness_mean": [],
        "fitness_std": [],
        "fitness_max": [],
        "fitness_min": [],
        "sigma_sq": [],
        "objective": [],
        "descriptors": [],
        "avg_pairwise_distance": [],
    }
    print("Starting optimization...")
    total_evals = 0
    for i in range(config["num_iterations"]):
        # Go over mini-batches of data
        key, epoch_key = jax.random.split(key)
        state = jitted_epoch(state=state, key=epoch_key)
        total_evals += config["population_size"]

        # Compute and record logs
        if i % config["log_frequency"] == 0:
            f = state.fitnesses
            b = state.descriptors
            e_sigma = 1.0  # Just some number

            # Recompute objective A(theta) for logging
            dist_sq = jnp.sum(
                jnp.square(
                    state.descriptors[:, None, :] - state.descriptors[None, :, :]
                ),
                axis=-1,
            )
            kernel = jnp.exp(-dist_sq / e_sigma)
            f_sqrt_prod = jnp.sqrt(jnp.clip(f[:, None], 0) * jnp.clip(f[None, :], 0))
            repulsion_term = jnp.sum(jnp.triu(kernel * f_sqrt_prod, k=1))
            objective_value = jnp.sum(f) - repulsion_term

            logs["total_evals"].append(total_evals)
            logs["fitnesses"].append(np.asarray(f))
            logs["fitness_mean"].append(float(jnp.mean(f)))
            logs["fitness_std"].append(float(jnp.std(f)))
            logs["fitness_max"].append(float(jnp.max(f)))
            logs["fitness_min"].append(float(jnp.min(f)))
            logs["sigma_sq"].append(float(state.sigma_sq))
            logs["objective"].append(float(objective_value))
            logs["descriptors"].append(np.asarray(b))
            logs["avg_pairwise_distance"].append(
                float(jnp.mean(jnp.sqrt(dist_sq[jnp.triu_indices(b.shape[0], 1)])))
            )

            print(
                f"Iter {i:5d} | "
                f"Objective: {objective_value:8.2f} | "
                f"Fitness (max/mean): {jnp.max(f):6.2f}/{jnp.mean(f):6.2f} | "
                f"Sigma^2: {state.sigma_sq:6.3f}"
            )

            if config["wandb"]["enable"]:
                wandb.log(
                    {
                        "iteration": i,
                        "objective": float(objective_value),
                        "fitness/mean": float(jnp.mean(f)),
                        "fitness/std": float(jnp.std(f)),
                        "fitness/max": float(jnp.max(f)),
                        "fitness/min": float(jnp.min(f)),
                        "sigma_sq": float(state.sigma_sq),
                        "avg_pairwise_dist": logs["avg_pairwise_distance"][-1],
                        "total_evals": total_evals,
                    }
                )

    print("Optimization finished.")
    return np.asarray(state.solutions), logs, {}  # solutions, logs, artifacts
