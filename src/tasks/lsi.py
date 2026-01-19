from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from flaxmodels.stylegan2 import (
    Generator as StyleGAN2Generator,
)
from flaxmodels.stylegan2 import (
    MappingNetwork,
    SynthesisNetwork,
)
from transformers import CLIPProcessor, FlaxCLIPModel

from src.tasks.base import EvalOutput, Task

# NOTE: In CMA-MAE's StyleGAN2 experiment, they compute the fitness by creating 32 crops
# of a generated 1024x1024 image and averageing the clip score of them.


# Helper functions for loss calculations
def _normalize(x: jax.Array, axis: int = -1) -> jax.Array:
    """Operates on batches. Input: (bsz, n), Output: (bsz, n)"""
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)


def _spherical_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Output is in [0, (pi^2)/2] where lower is more similar.
    """
    x, y = _normalize(x), _normalize(y)
    return jnp.square(jnp.arcsin(jnp.linalg.norm(x - y, axis=-1) / 2)) * 2


def _cos_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """Output is in [0, pi]"""
    x, y = _normalize(x), _normalize(y)
    return jnp.arcsin(jnp.linalg.norm(x - y, axis=-1) / 2) * 2


class LatentIlluminationTask(Task):
    """Implements the LSI Task"""

    def __init__(
        self,
        target_prompt: str,
        descriptor_prompts: List[Tuple[str, str]],
        normalized_descriptors: bool,
        stylegan_model_name: str,
        clip_model_name: str,
        seed: int,
    ):
        """
        Fitness is in (-infty, 100)
        """
        self.normalized_descriptors = normalized_descriptors
        self.clip_input_size = 224
        rng, gan_init_rng, gan_stats_rng = jax.random.split(jax.random.PRNGKey(seed), 3)

        print("Loading models and setting up task...")
        # Instantiate Generator to get its properties and parameters
        gan = StyleGAN2Generator(pretrained=stylegan_model_name)
        gan_params = gan.init(gan_init_rng, jnp.ones((1, gan.z_dim)))

        self.mapping_module = MappingNetwork(
            z_dim=gan.z_dim,
            c_dim=gan.c_dim,
            w_dim=gan.w_dim,
            num_ws=gan.num_ws,
            num_layers=gan.num_mapping_layers,
            name="mapping_network",
        )
        self.synthesis_module = SynthesisNetwork(
            resolution=gan.resolution,
            num_channels=gan.num_channels,
            w_dim=gan.w_dim,
            name="synthesis_network",
        )

        # Extract the parameters for each sub-module
        self.mapping_params = {
            "params": gan_params["params"]["mapping_network"],
            "moving_stats": gan_params["moving_stats"]["mapping_network"],
        }
        self.synthesis_params = {
            "params": gan_params["params"]["synthesis_network"],
            "noise_consts": gan_params["noise_consts"]["synthesis_network"],
        }

        self.num_ws, self.w_dim, self.z_dim = gan.num_ws, gan.w_dim, gan.z_dim
        self.solution_size = (gan.num_ws * gan.w_dim,)
        self.descriptor_dim = len(descriptor_prompts)
        self.w_avg_vec = gan_params["moving_stats"]["mapping_network"]["w_avg"]
        self.w_stds, self.q_norm = self._init_stylegan_stats(gan_stats_rng)
        del gan

        self.clip = FlaxCLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.target_embedding = self._embed_text(target_prompt)  # Shape: (1, 512)
        self.descriptor_embeddings = jnp.array(
            [
                (self._embed_text(p).squeeze(), self._embed_text(n).squeeze())
                for p, n in descriptor_prompts
            ]
        )

        # NOTE: This function operates on batches of solutions (unlike other tasks)
        def _combined_fn(solutions: jax.Array) -> jax.Array:
            batch_size = solutions.shape[0]
            q = solutions.reshape((batch_size, self.num_ws, self.w_dim))
            w_plus = q * self.w_stds + self.w_avg_vec

            img = self.synthesis_module.apply(
                self.synthesis_params,
                dlatents_in=w_plus,
                noise_mode="const",
            )

            processed_img = self._preprocess_for_clip(img)
            img_embedding = self.clip.get_image_features(
                processed_img
            )  # Shape: (bsz, 512)

            dist_loss = _spherical_distance(img_embedding, self.target_embedding)
            dist_loss = 100 * dist_loss / (jnp.pi * jnp.pi / 2)  # [0, 100]
            reg_loss = 0.2 * jnp.square(
                jnp.maximum(
                    jnp.linalg.norm(solutions.reshape(batch_size, -1), axis=1),
                    self.q_norm,
                )
                - self.q_norm
            )
            fitness = 100 - (dist_loss + reg_loss)  # (-infty, 100]

            pos_loss = _cos_distance(
                jnp.expand_dims(img_embedding, axis=1),
                self.descriptor_embeddings[:, 0, :],
            )
            neg_loss = _cos_distance(
                jnp.expand_dims(img_embedding, axis=1),
                self.descriptor_embeddings[:, 1, :],
            )

            descriptors = (pos_loss - neg_loss + jnp.pi) / (2 * jnp.pi)  # [0, 1]
            # Per the pyribs implementation and our own empirical observations, the values
            # of descriptors are almost always in [0.45, 0.55]. So, we rescale them.
            descriptors = (jnp.clip(descriptors, 0.45, 0.55) - 0.45) * 10

            if not self.normalized_descriptors:
                descriptors = jax.scipy.special.logit(
                    jnp.clip(descriptors, 1e-6, 1 - 1e-6)
                )

            return jnp.concatenate(
                [jnp.expand_dims(fitness, axis=1), descriptors], axis=1
            )

        def _single_combined_fn_with_aux(solution: jax.Array) -> jax.Array:
            out = _combined_fn(jnp.expand_dims(solution, 0)).squeeze(0)
            return out, out

        self._vmapped_value_fn = jax.jit(_combined_fn)
        self._vmapped_jac_fn = jax.jit(
            jax.vmap(jax.jacrev(_single_combined_fn_with_aux, has_aux=True))
        )

    @partial(jax.jit, static_argnames=("self",))
    def _preprocess_for_clip(self, image_batch: jax.Array) -> jax.Array:
        """
        Processes the output of the synthetic network (pixel values in [-1, 1]) using
        clip's mean/std and resizes it to the appropriate size.

        Input shape: (batch_size, height, width, num_channels)
        Output shape: (batch_size, num_channels, height, width)
        """
        assert image_batch.ndim == 4 and image_batch.shape[-1] == 3  # Channel-last
        OPENAI_CLIP_MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073])
        OPENAI_CLIP_STD = jnp.array([0.26862954, 0.26130258, 0.27577711])
        image_batch = jax.image.resize(
            image_batch,
            shape=(image_batch.shape[0], self.clip_input_size, self.clip_input_size, 3),
            method="bicubic",
        )
        image_batch = (image_batch + 1.0) / 2.0  # [0, 1]
        image_batch = (image_batch - OPENAI_CLIP_MEAN) / (OPENAI_CLIP_STD)  # normalize
        image_batch = jnp.transpose(image_batch, (0, 3, 1, 2))

        return image_batch

    @partial(jax.jit, static_argnames=("self",))
    def get_image_from_solution(self, solution: jax.Array) -> jax.Array:
        """Converts a single solution into an image"""
        q = solution.reshape((self.num_ws, self.w_dim))
        w_plus = q * self.w_stds + self.w_avg_vec

        image_tensor = self.synthesis_module.apply(
            self.synthesis_params,
            dlatents_in=jnp.expand_dims(w_plus, axis=0),
            noise_mode="const",
        )

        image_uint8 = jnp.clip(
            (image_tensor.squeeze(0) + 1.0) / 2.0 * 255.0, 0, 255
        ).astype(jnp.uint8)

        return image_uint8

    @partial(jax.jit, static_argnames=("self",))
    def _init_stylegan_stats(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        z = jax.random.normal(key, (10000, self.z_dim))
        ws = self.mapping_module.apply(
            self.mapping_params, z, truncation_psi=0.7, skip_w_avg_update=True
        )

        w_stds = jnp.std(ws, axis=0)
        qs_flat = ((ws - self.w_avg_vec) / w_stds).reshape(10000, -1)
        # vectors whose norms are smaller than q_norm will be penalized
        q_norm = jnp.mean(jnp.linalg.norm(qs_flat, axis=1)) * 0.35
        return w_stds, q_norm

    def _embed_text(self, prompt: str) -> jax.Array:
        text_inputs = self.clip_processor(
            text=[prompt], return_tensors="jax", padding=True
        )
        text_embedding = self.clip.get_text_features(**text_inputs)

        return text_embedding

    @partial(jax.jit, static_argnames=("self", "return_grad"))
    def evaluate(
        self, solutions: jax.Array, key: jax.Array, return_grad: bool = True
    ) -> EvalOutput:
        combined = self._vmapped_value_fn(solutions)
        fitnesses, descriptors = combined[:, 0], combined[:, 1:]
        if return_grad:
            # Since memory consumption during backwards pass is high, we process large
            # batches using mini batches.

            mini_batch_size = min(solutions.shape[0], 4)
            n_mini_batches = solutions.shape[0] // mini_batch_size

            assert solutions.shape[0] % mini_batch_size == 0

            _, (jacobians_chunked, values_chunked) = jax.lax.scan(
                lambda carry, mb: (carry, self._vmapped_jac_fn(mb)),
                None,
                solutions.reshape(
                    n_mini_batches, mini_batch_size, *solutions.shape[1:]
                ),
            )

            jacobians = jacobians_chunked.reshape(
                solutions.shape[0], *jacobians_chunked.shape[2:]
            )
            combined = values_chunked.reshape(solutions.shape[0], -1)
            fitness_grads, descriptor_grads = jacobians[:, 0, :], jacobians[:, 1:, :]
        else:
            combined = self._vmapped_value_fn(solutions)
            fitness_grads, descriptor_grads = (
                jnp.zeros_like(solutions),
                jnp.zeros_like(descriptors),
            )
        fitnesses, descriptors = combined[:, 0], combined[:, 1:]
        return EvalOutput(fitnesses, descriptors, fitness_grads, descriptor_grads)

    def vanilla_evaluate(
        self, solution: jax.Array, key: jax.Array
    ) -> Tuple[float, jax.Array]:
        fit, desc = self.evaluate(
            jnp.expand_dims(solution, axis=0), key, return_grad=False
        )[:2]
        return float(fit[0]), desc[0]

    def get_random_solution(
        self,
        n: int,
        key: jax.Array,
        batch_size: int = 64,
    ) -> jax.Array:
        """
        Following pyribs' example, finds good starting latent by selecting top
        performing solutions from a larger pool of randomly generated ones.
        """

        @jax.jit
        def _get_loss(qs: jax.Array) -> jax.Array:
            w_plus = qs * self.w_stds + self.w_avg_vec

            images = self.synthesis_module.apply(
                self.synthesis_params, dlatents_in=w_plus, noise_mode="const"
            )
            images = self._preprocess_for_clip(images)
            image_embs = self.clip.get_image_features(images)

            return _spherical_distance(image_embs, self.target_embedding)

        @jax.jit
        def _evaluate_batch(batch_key: jax.Array):
            mini_batch_size = 8
            z_vectors = jax.random.normal(batch_key, (batch_size, self.z_dim))
            ws = self.mapping_module.apply(
                self.mapping_params,
                z_vectors,
                truncation_psi=0.7,
                skip_w_avg_update=True,
            )
            q_batch = (ws - self.w_avg_vec) / self.w_stds

            best_loss = jnp.array(jnp.inf)
            best_q = jnp.zeros_like(q_batch[0])

            for i in range(0, batch_size, mini_batch_size):
                q_mini_batch = q_batch[i : i + mini_batch_size]
                losses = _get_loss(q_mini_batch)

                min_loss_in_mini_batch = jnp.min(losses)
                best_idx_in_mini_batch = jnp.argmin(losses)

                # update function when new minimum is found
                def _set_new_best(_):
                    return (
                        min_loss_in_mini_batch,
                        q_mini_batch[best_idx_in_mini_batch],
                    )

                # keep current best if no improvement
                def _keep_current_best(_):
                    return (best_loss, best_q)

                best_loss, best_q = jax.lax.cond(
                    min_loss_in_mini_batch < best_loss,
                    _set_new_best,
                    _keep_current_best,
                    operand=None,
                )

            return best_q

        keys = jax.random.split(key, n)

        sols = []
        for i, batch_key in enumerate(keys):
            q = _evaluate_batch(batch_key)
            sols.append(q)

        return jnp.array(sols)


if __name__ == "__main__":
    task = LatentIlluminationTask(
        target_prompt="A photo of Tom Cruise",
        descriptor_prompts=[
            ("Photo of a young kid", "Photo of an elderly person"),
            ("Photo of a person with long hair", "Photo of a person with short hair"),
        ],
        normalized_descriptors=False,
        stylegan_model_name="ffhq",
        clip_model_name="openai/clip-vit-base-patch32",
        seed=23,
    )
    rng = jax.random.PRNGKey(32)
    x = task.get_random_solution(n=8, batch_size=32, key=rng)
    print(x.shape)
    res = task.evaluate(x, rng, return_grad=True)
    print(res.fitnesses)
