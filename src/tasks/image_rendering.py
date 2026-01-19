from functools import partial
import jax
import jax.numpy as jnp
from typing import Tuple
from PIL import Image
import dm_pix as pix
from src.tasks.base import Task, EvalOutput


@partial(jax.jit, static_argnames=("width", "height", "softness"))
def render(circles, width, height, softness) -> jax.Array:
    """
    Renders an entire image by alpha-compositing circles.

    Args:
        circles (jax.Array): A (N, 7) array of circle parameters.
        width (int): The width of the canvas.
        height (int): The height of the canvas.
        softness (float): Controls the anti-aliasing of circle edges.

    Returns:
        jax.Array: An (H, W, 3) array representing the rendered image.
    """
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(height),
        jnp.arange(width),
        indexing="ij",
    )

    def draw_circle(carry, circle_params):
        """Draws a single circle onto the canvas"""
        eps = 1e-6

        canvas, log_transparency = carry
        cx_logit, cy_logit, radius_logit, r_logit, g_logit, b_logit, alpha_logit = (
            circle_params
        )
        cx = jax.nn.sigmoid(cx_logit) * width
        cy = jax.nn.sigmoid(cy_logit) * height
        max_radius = min(width, height) / 2.0
        radius = jax.nn.sigmoid(radius_logit) * max_radius + 1.0
        color_circle = jax.nn.sigmoid(jnp.array([r_logit, g_logit, b_logit]))
        alpha_circle = jax.nn.sigmoid(alpha_logit)

        # Calculate alpha mask for current circle
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        val = (dist_sq - radius**2) / (softness + eps)
        pixel_alpha = alpha_circle * jax.nn.sigmoid(-val)  # 1-sigmoid(val)
        pixel_alpha = jnp.expand_dims(pixel_alpha, axis=-1)  # (H, W, 1)

        # Alpha compositing
        pixel_color = color_circle * pixel_alpha
        canvas = pixel_color + (1 - pixel_alpha) * canvas
        log_transparency = log_transparency + jnp.log(1.0 - pixel_alpha + eps)

        return (canvas, log_transparency), None

    initial_canvas = jnp.full((height, width, 3), 0.0)
    initial_log_transparency = jnp.full((height, width, 1), 0.0)

    (final_canvas, final_log_transparency), _ = jax.lax.scan(
        draw_circle, (initial_canvas, initial_log_transparency), circles
    )

    # Add white background
    final_canvas = final_canvas + jnp.exp(final_log_transparency) * jnp.full(
        (height, width, 3), 1.0
    )

    return final_canvas


class ImageRenderingTask(Task):
    """Implements the Image Rendering benchmark task."""

    def __init__(
        self,
        target_image: str,
        num_circles: int = 128,
        canvas_width: int = 128,
        canvas_height: int = 128,
        softness: float = 20.0,
        objective_type: str = "ssim",
        normalized_descriptors: bool = False,
    ):
        self.solution_size = (num_circles, 7)
        self.descriptor_dim = 5

        self.num_circles = num_circles
        self.width = canvas_width
        self.height = canvas_height
        self.softness = softness
        self.objective_type = objective_type
        self.normalized_descriptors = normalized_descriptors
        assert objective_type in [
            "ssim",
            "mse",
        ], f'Unknown objective type: "{objective_type}"'
        self.target_image = self.load_image(target_image)

        # Concatenate objective and descriptors to compute the Jacobian at once
        def _combined_fn(solution):
            eps = 1e-6
            rendered_image = render(solution, self.width, self.height, self.softness)
            if self.objective_type == "mse":
                mse_loss = jnp.mean((rendered_image - self.target_image) ** 2)
                # Pixels are in [0, 1], so max MSE is 1
                max_loss = 1.0
                objective = max_loss - mse_loss
            elif self.objective_type == "ssim":
                objective = jnp.mean(
                    pix.ssim(
                        rendered_image, self.target_image, max_val=1.0, filter_size=5
                    )
                )
                objective = (objective + 1) / 2  # SSIM is in [-1, 1]

            descriptors = self.get_behavioral_descriptors(solution)
            if not self.normalized_descriptors:
                descriptors = jax.scipy.special.logit(
                    jnp.clip(descriptors, eps, 1 - eps)
                )

            objective = 100 * objective  # [0, 100]
            objective = jnp.expand_dims(objective, axis=0)
            return jnp.concatenate([objective, descriptors])

        self._vmapped_jac_fn = jax.jit(jax.vmap(jax.jacrev(_combined_fn)))
        self._vmapped_value_fn = jax.jit(jax.vmap(_combined_fn))

    def load_image(self, image_path: str) -> jax.Array:
        """Loads image from a given path and returns normalized images"""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        # Normalize to [0, 1]
        img = jnp.array(img, dtype=jnp.float32) / 255.0
        return img

    @partial(jax.jit, static_argnames=("self", "return_grad"))
    def evaluate(
        self, solutions: jax.Array, key: jax.Array, return_grad: bool = True
    ) -> EvalOutput:
        combined_values = self._vmapped_value_fn(solutions)  # Shape: (K, 1 + m)
        fitnesses = combined_values[:, 0]
        descriptors = combined_values[:, 1:]

        if return_grad:
            jacobians = self._vmapped_jac_fn(solutions)  # Shape: (K, 1 + m, D)
            fitness_grads = jacobians[:, 0, :]
            descriptor_grads = jacobians[:, 1:, :]
        else:
            fitness_grads = jnp.zeros(1)
            descriptor_grads = jnp.zeros(1)

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
        combined_values = self._vmapped_value_fn(jnp.expand_dims(solution, 0))
        fitness = combined_values[0, 0]
        descriptors = combined_values[0, 1:]

        return float(fitness), descriptors

    @partial(jax.jit, static_argnames=("self", "n"))
    def get_random_solution(self, n, key):
        key, init_key = jax.random.split(key)
        init_solutions = jax.random.uniform(
            init_key,
            shape=(n, self.num_circles, 7),
            minval=-2.0,
            maxval=2.0,
        )
        return init_solutions

    @partial(jax.jit, static_argnames=("self"))
    def get_behavioral_descriptors(self, circles: jax.Array) -> jax.Array:
        """
        Computes behavioral descriptors for a set of parameterized circles.

        Each descriptor is normalized to the [0, 1] range. A higher value
        corresponds to a higher degree of the described property (e.g.,
        higher radius mean, more clustering).

        Args:
            circles: An (N, 7) array of circle parameter logits.

        Returns:
            A (5,) array containing the normalized descriptors:
            [0] mean_r: The average circle size (0: small, 1: large).
            [1] std_r: The uniformity of circle sizes (0: uniform, 1: varied).
            [2] color_spread: The variety of colors in the palette (0: monochrome, 1: high variety).
            [3] color_harmony: Hue agreement (0: dissonant/spread, 1: harmonious/clustered).
            [4] cluster_coef: Spatial clustering (0: dispersed, 1: tightly clustered).
        """
        cx = jax.nn.sigmoid(circles[:, 0]) * float(self.width)
        cy = jax.nn.sigmoid(circles[:, 1]) * float(self.height)
        centers = jnp.stack([cx, cy], axis=1)

        max_radius = float(min(self.width, self.height)) / 2.0
        radii = jax.nn.sigmoid(circles[:, 2]) * max_radius + 1.0

        color_logits = circles[:, 3:6]
        colors_rgb = jax.nn.sigmoid(color_logits)

        # Radius Descriptors
        mean_r = (jnp.mean(radii) - 1.0) / max_radius
        std_r = jnp.sqrt(jnp.var(radii) + 1e-6)
        # The max std is 0.5 * range_length
        std_r = std_r / (0.5 * max_radius)

        # Color Descriptors
        mean_rgb = jnp.mean(colors_rgb, axis=0, keepdims=True)
        # Average distance from the mean color
        color_spread = jnp.mean(jnp.linalg.norm(colors_rgb - mean_rgb, axis=1))
        # Max possible spread is from a corner to center of unit cube, sqrt(3)/2
        color_spread = color_spread / (jnp.sqrt(3.0) / 2)

        # For harmony, convert to HSV and find the mean length of hue vectors
        hues = pix.rgb_to_hsv(colors_rgb)[:, 0]
        hues = jnp.stack([jnp.cos(hues), jnp.sin(hues)], axis=1)
        color_harmony = jnp.linalg.norm(jnp.mean(hues, axis=0))

        # Spatial Clustering Descriptor ( O(N^2) )
        diff = centers[:, None, :] - centers[None, :, :]
        squared_dist = jnp.sum(jnp.square(diff), axis=-1)
        dist_matrix = jnp.sqrt(squared_dist + 1e-6)

        dist_matrix = jnp.where(
            jnp.eye(circles.shape[0], dtype=bool), jnp.inf, dist_matrix
        )
        k_smallest_dists = jax.lax.top_k(-dist_matrix, k=5)[0] * -1
        mean_knn_dist = jnp.mean(k_smallest_dists)
        # Normalize by 0.25 * canvas diagonal and invert so more clustered -> 1
        cluster_coef = 1.0 - jnp.clip(
            mean_knn_dist / (0.25 * jnp.sqrt(self.width**2 + self.height**2)), 0, 1
        )

        descriptors = jnp.stack(
            [mean_r, std_r, color_spread, color_harmony, cluster_coef]
        ).clip(0, 1)

        return descriptors
