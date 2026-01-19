# Example usage: python -m src.evaluate "outputs/Data/Time"

import argparse
import json
from pathlib import Path
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_compilation_cache_dir", "./cache")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from omegaconf import OmegaConf
from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from vendi_score import vendi

from src.tasks.utils import create_task

# Constants
EVAL_DIR_NAME = "evaluation_results"
BATCH_SIZE = 16
SEED = 232323
# NOTE: Use 512 cells for rastrigin and lsi and 1024 for rendering
CVT_CELLS = 1024


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def reevaluate_solutions(
    task, solutions: np.ndarray, key: jax.random.PRNGKey
) -> tuple[np.ndarray, np.ndarray]:
    """Re-evaluates solutions and returns their fitnesses and descriptors."""
    print(f"Re-evaluating {len(solutions)} solutions in batches of {BATCH_SIZE}...")
    num_solutions = solutions.shape[0]
    all_fitnesses = []
    all_descriptors = []

    for i in range(0, num_solutions, BATCH_SIZE):
        batch_solutions = jnp.asarray(solutions[i : i + BATCH_SIZE])
        key, eval_key = jax.random.split(key)

        eval_output = task.evaluate(batch_solutions, eval_key, return_grad=False)

        all_fitnesses.append(np.array(eval_output.fitnesses))
        all_descriptors.append(np.array(eval_output.descriptors))

    fitnesses = np.concatenate(all_fitnesses)
    descriptors = np.concatenate(all_descriptors)

    return fitnesses, descriptors


def compute_pairwise_distances(descriptors: np.ndarray, output_dir: Path):
    """Computes pairwise distance stats and saves a sorted heatmap."""
    print("Computing pairwise distances...")
    distances = pairwise_distances(descriptors, metric="euclidean")

    # Sort the distance matrix for better visualization using hierarchical clustering
    try:
        if descriptors.shape[0] > 1:
            Z = linkage(descriptors, "ward")
            # Sort the matrix by cluster labels
            labels = fcluster(Z, t=0.1, criterion="distance")
            sorted_indices = np.argsort(labels)
            distances_sorted = distances[sorted_indices, :][:, sorted_indices]
        else:
            distances_sorted = distances
    except ValueError:
        print(
            "Could not perform clustering for sorting heatmap. Using unsorted distances."
        )
        distances_sorted = distances

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        distances_sorted,
        cmap="viridis",
        ax=ax,
        # cbar_kws={"label": "Euclidean Distance"},
        vmin=0.0,
        vmax=0.8,
        xticklabels=False,
        yticklabels=False,
    )
    # ax.set_title("Pairwise Descriptor Distances")
    # ax.set_xlabel("Solution Index")
    # ax.set_ylabel("Solution Index")
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_distances_heatmap.png", dpi=300)
    # plt.savefig(output_dir / "pairwise_distances_heatmap.pdf")
    plt.close(fig)

    # Exclude diagonal (zeros) for mean calculation
    mean_dist = np.mean(distances[np.triu_indices_from(distances, k=1)])
    return {"mean": mean_dist}


def compute_fitness_stats(fitnesses: np.ndarray, output_dir: Path):
    """Computes fitness stats and saves a histogram."""
    print("Computing fitness statistics...")
    stats = {
        "min": np.min(fitnesses),
        "max": np.max(fitnesses),
        "mean": np.mean(fitnesses),
        "std": np.std(fitnesses),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(fitnesses, kde=True, ax=ax, bins=30)
    ax.set_title("Fitness Distribution")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Count")
    ax.axvline(
        stats["mean"], color="r", linestyle="--", label=f'Mean: {stats["mean"]:.2f}'
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fitness_histogram.png", dpi=300)
    plt.close(fig)

    return stats


def compute_vendi_score(descriptors: np.ndarray, fitnesses: np.ndarray):
    """Computes Vendi Score and Quality-Weighted Vendi Score."""
    print("Computing Vendi Score...")
    if descriptors.shape[0] == 0:
        return {"vendi_score": 0.0, "quality_weighted_vendi_score": 0.0}

    # Gaussian Kernel (RBF)
    VENDI_SIGMA = np.sqrt(descriptors.shape[-1] / 6.0)  # Heuristic
    k = lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / (VENDI_SIGMA**2))

    score = vendi.score(descriptors, k=k)

    mean_fitness = np.mean(fitnesses)
    quality_weighted_score = score * mean_fitness

    return {
        "vendi_score": score,
        "quality_weighted_vendi_score": quality_weighted_score,
    }


def compute_qd_metrics(
    descriptors: np.ndarray, fitnesses: np.ndarray, descriptor_dim: int
):
    """Computes QD Score and Coverage using a fixed-seed CVT archive."""
    print("Computing QD Score and Coverage...")
    if descriptors.shape[0] == 0:
        return {"qd_score": 0.0, "coverage": 0.0}

    # Create a CVT archive with a fixed seed for reproducibility
    archive = CVTArchive(
        solution_dim=0,  # Not needed
        ranges=[(0.0, 1.0)] * descriptor_dim,
        cells=CVT_CELLS,
        use_kd_tree=descriptor_dim < 10,
        seed=SEED,
    )

    archive.add(
        solution=np.zeros((len(fitnesses), 0)),  # Dummy solutions
        objective=fitnesses,
        measures=descriptors,
    )

    return {
        "qd_score": archive.stats.qd_score,
        "coverage": archive.stats.coverage,
        "size": len(archive),
        "archive": archive,
    }


def compute_approx_soft_qd_score(descriptors: np.ndarray, fitnesses: np.ndarray):
    """
    Computes the approximate soft QD score.

    NOTE: Remember to first transform the descriptors from [0, 1]
    to (-inf, +inf) using the logit function.
    """
    eps = 1e-6
    e_sigma = descriptors.shape[-1] / 6.0

    descriptors = scipy.special.logit(np.clip(descriptors, eps, 1 - eps))

    # Compute SoftQD objective objective S(theta)
    dist_sq = np.sum(
        np.square(descriptors[:, None, :] - descriptors[None, :, :]),
        axis=-1,
    )
    kernel = np.exp(-dist_sq / e_sigma)
    f_sqrt_prod = np.sqrt(
        jnp.clip(fitnesses[:, None], 0) * jnp.clip(fitnesses[None, :], 0)
    )
    repulsion_term = np.sum(np.triu(kernel * f_sqrt_prod, k=1))
    objective_value = np.sum(fitnesses) - repulsion_term

    return {"approx_soft_qd_score": objective_value}


def plot_archive(archive: CVTArchive, output_dir: Path):
    if archive.measure_dim != 2:
        return
    else:
        plt.style.use("seaborn-v0_8-white")
        plt.figure(figsize=(8, 6))
        cvt_archive_heatmap(archive, vmin=0, vmax=100)
        # plt.title(f"CVT archive with {CVT_CELLS} cells")
        plt.xlabel("Descriptor 1")
        plt.ylabel("Descriptor 2")
        plt.savefig(output_dir / "cvt_archive.png", dpi=300)
        plt.savefig(output_dir / "cvt_archive.pdf")
        plt.close()


def plot_embeddings(descriptors: np.ndarray, fitnesses: np.ndarray, output_dir: Path):
    """Generates and saves PCA and t-SNE plots."""
    if descriptors.shape[0] < 3 or descriptors.shape[1] < 2:
        print("Not enough data points or dimensions for embedding plots.")
        return

    pca = PCA(n_components=2)
    desc_2d_pca = pca.fit_transform(descriptors)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        desc_2d_pca[:, 0],
        desc_2d_pca[:, 1],
        c=fitnesses,
        cmap="viridis",
        alpha=0.8,
        vmin=0,
        vmax=100,
    )
    plt.colorbar(sc, label="Fitness")
    ax.set_title("PCA of Descriptors")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_plot.png", dpi=300)
    plt.close(fig)

    tsne = TSNE(
        n_components=2, perplexity=min(32, len(descriptors) - 1), random_state=SEED
    )
    desc_2d_tsne = tsne.fit_transform(descriptors)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        desc_2d_tsne[:, 0],
        desc_2d_tsne[:, 1],
        c=fitnesses,
        cmap="viridis",
        alpha=0.8,
        vmin=0,
        vmax=100,
    )
    plt.colorbar(sc, label="Fitness")
    ax.set_title("t-SNE of Descriptors")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_plot.png", dpi=300)
    plt.close(fig)


def main(logdir: str):

    # Load data
    log_path = Path(logdir).resolve()

    config_path = log_path / ".hydra" / "config.yaml"
    solutions_path = log_path / "solutions.npy"

    assert log_path.is_dir(), f"Log directory not found: {log_path}"
    assert (
        config_path.exists() and solutions_path.exists()
    ), f"Missing 'config.yaml' or 'solutions.npy' in {log_path}"

    output_dir = log_path / EVAL_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    cfg = OmegaConf.load(config_path)
    task_cfg = OmegaConf.to_container(cfg.task)
    # IMPORTANT: Override descriptor normalization
    task_cfg["normalized_descriptors"] = True

    solutions = np.load(solutions_path)
    print(f"Loaded {len(solutions)} solutions from {solutions_path}")

    # Evaluate solutions
    task = create_task(task_cfg)
    key = jax.random.PRNGKey(SEED)
    fitnesses, descriptors = reevaluate_solutions(task, solutions, key)

    # Compute stats
    metrics = {}
    metrics["pairwise_distance"] = compute_pairwise_distances(descriptors, output_dir)
    metrics["fitness"] = compute_fitness_stats(fitnesses, output_dir)
    metrics["vendi_score_metrics"] = compute_vendi_score(descriptors, fitnesses)
    qd_metrics = compute_qd_metrics(descriptors, fitnesses, task.descriptor_dim)
    qd_archive = qd_metrics.pop("archive")
    metrics["qd_metrics"] = qd_metrics
    metrics["soft_qd_metrics"] = compute_approx_soft_qd_score(descriptors, fitnesses)

    plot_embeddings(descriptors, fitnesses, output_dir)
    plot_archive(qd_archive, output_dir)

    # Save
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logdir",
        type=str,
        help="Path to the Hydra log directory (e.g., outputs/Date/Time/)",
    )
    args = parser.parse_args()

    main(args.logdir)
