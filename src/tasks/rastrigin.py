from functools import partial
import jax
import jax.numpy as jnp
from typing import Tuple

from src.tasks.base import Task, EvalOutput


@partial(jax.jit, static_argnames=("solution_dim", "descriptor_dim"))
def _rastrigin_fn(
    solution: jax.Array, solution_dim: int, descriptor_dim: int
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-compiled Rastrigin function for a single solution.
    The implementation follows the CMA-MAE paper. The only difference is the
    normalization of descriptors to [0, 1]
    Rastrigin function is defined as:

    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    """

    target_shift = 5.12 * 0.4
    best_obj = 0.0
    displacement = -5.12 * jnp.ones_like(solution) - target_shift
    sum_terms = jnp.square(displacement) - 10 * jnp.cos(2 * jnp.pi * displacement)
    worst_obj = 10 * solution_dim + jnp.sum(sum_terms)

    displacement = solution - target_shift
    sum_terms = jnp.square(displacement) - 10 * jnp.cos(2 * jnp.pi * displacement)
    raw_obj = 10 * solution_dim + jnp.sum(sum_terms)

    # Normalized to [0, 100] where worst objective is approximated as the bottom left point
    objective = 100 * (raw_obj - worst_obj) / (best_obj - worst_obj)  # (-infty, 100]

    clip_mask = jnp.abs(solution) > 5.12
    # If we don't use safe_denom, jax will evaluate both branches of jnp.where and
    # may encounter a division by zero, creating NaNs.
    safe_denom = jnp.where(clip_mask, solution, 1.0)
    clipped = jnp.where(clip_mask, 5.12 / safe_denom, solution)

    # Calculate descriptors by splitting the solution into `descriptor_dim` chunks.
    descriptors = clipped.reshape((descriptor_dim, -1)).mean(axis=1)
    descriptors = (descriptors + 5.12) / (2 * 5.12)  # Normalize to [0, 1]

    return objective, descriptors


class RastriginTask(Task):
    """Implements the Rastrigin benchmark task."""

    def __init__(
        self,
        solution_dim: int,
        descriptor_dim: int,
        normalized_descriptors: bool = False,
    ):
        """
        Fitness is in (-infty, 100)
        """
        self.solution_size = (solution_dim,)
        self.descriptor_dim = descriptor_dim
        self.normalized_descriptors = normalized_descriptors
        self.init_range = [-5.12, 5.12]

        # Concatenate objective and descriptors to compute the Jacobian at once
        def _combined_fn(solution):
            eps = 1e-6
            objective, descriptors = _rastrigin_fn(
                solution, solution_dim, self.descriptor_dim
            )
            if not self.normalized_descriptors:
                descriptors = jax.scipy.special.logit(
                    jnp.clip(descriptors, eps, 1 - eps)
                )

            objective = jnp.expand_dims(objective, axis=0)
            return jnp.concatenate([objective, descriptors])

        def _jac_fn_with_aux(solution):
            combined_output = _combined_fn(solution)
            return combined_output, combined_output

        self._vmapped_jac_and_value_fn = jax.jit(
            jax.vmap(jax.jacrev(_jac_fn_with_aux, has_aux=True))
        )
        self._vmapped_value_fn = jax.jit(jax.vmap(_combined_fn))

    @partial(jax.jit, static_argnames=("self", "return_grad"))
    def evaluate(
        self, solutions: jax.Array, key: jax.Array, return_grad: bool = True
    ) -> EvalOutput:
        if return_grad:
            # Jacobian's shape: (K, 1 + m, D)
            jacobians, combined_values = self._vmapped_jac_and_value_fn(solutions)
            fitness_grads = jacobians[:, 0, :]
            descriptor_grads = jacobians[:, 1:, :]
        else:
            combined_values = self._vmapped_value_fn(solutions)  # Shape: (K, 1 + m)
            fitness_grads = jnp.zeros(1)
            descriptor_grads = jnp.zeros(1)

        fitnesses = combined_values[:, 0]
        descriptors = combined_values[:, 1:]

        return EvalOutput(
            fitnesses=fitnesses,
            descriptors=descriptors,
            fitness_grads=fitness_grads,
            descriptor_grads=descriptor_grads,
        )

    @partial(jax.jit, static_argnames=("self"))
    def vanilla_evaluate(
        self, solution: jax.Array, key: jax.Array
    ) -> Tuple[float, jax.Array]:
        fit, desc = _rastrigin_fn(solution, self.solution_size[0], self.descriptor_dim)
        return float(fit), desc

    @partial(jax.jit, static_argnames=("self", "n"))
    def get_random_solution(self, n, key):
        key, init_key = jax.random.split(key)
        init_solutions = jax.random.uniform(
            init_key,
            shape=(n, self.solution_size[0]),
            minval=self.init_range[0],
            maxval=self.init_range[1],
        )
        return init_solutions
