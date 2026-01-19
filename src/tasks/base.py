from typing import Protocol, NamedTuple, Tuple
import jax


class EvalOutput(NamedTuple):
    """Holds the results of evaluating a batch of K solutions.

    It contains the fitness, behavior descriptors, and their gradients with
    respect to the solution parameters.

    Attributes:
        fitnesses (jax.Array): Fitness values for each solution.
            Shape: (K,)
        descriptors (jax.Array): Behavior descriptors for each solution.
            Shape: (K, m)
        fitness_grads (jax.Array): The gradient of the fitness function for
            each solution. Shape: (K, D). For solution `k`, this is the
            vector `\grad_\theta f(\theta_k)`.
        descriptor_grads (jax.Array): The Jacobian of the behavior function
            for each solution. Shape: (K, m, D). For solution `k`, this is
            the matrix `J_b(\theta_k)` where `J[i, j] = \partial b_i / \partial \theta_j`.
    """

    fitnesses: jax.Array
    descriptors: jax.Array
    fitness_grads: jax.Array
    descriptor_grads: jax.Array


class Task(Protocol):
    """Defines the interface for a task."""

    solution_size: Tuple[int]
    descriptor_dim: int

    def evaluate(
        self, solutions: jax.Array, key: jax.Array, return_grad: bool = True
    ) -> EvalOutput:
        """
        Evaluates a batch of K solutions, providing fitness, descriptors,
        and their gradients with respect to the solutions.
        """
        ...

    def simple_evaluate(
        self, solution: jax.Array, key: jax.Array
    ) -> Tuple[float, jax.Array]:
        """
        Evaluates a single solution, returning only its fitness and descriptor.
        Useful for evaluations and analysis without gradient overhead.
        """
        ...

    def get_random_solution(self, n: int, key: jax.Array) -> jax.Array:
        """
        Returns `n` randomly initialized solutions.
        """
        ...
