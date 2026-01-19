import os
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax

jax.config.update("jax_compilation_cache_dir", "./cache")
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

import numpy as np

from src.qd.cma_mae import train as train_cma_mae
from src.qd.cma_maega import train as train_cma_maega
from src.qd.cma_mega import train as train_cma_mega
from src.qd.dns import train as train_dns
from src.qd.nslc import train as train_nslc
from src.qd.pga_me import train as train_pga_me
from src.qd.softqd import train as train_softqd
from src.tasks.utils import create_task


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run the optimization."""
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    # Initialize wandb if enabled
    if cfg.wandb.enable:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.task.task_name}_{cfg.algo_name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
        )

    # Instantiate the task
    task = create_task(OmegaConf.to_container(cfg.task, resolve=True))

    if cfg.algo_name == "softqd":
        train = train_softqd
    elif cfg.algo_name == "cma_mae":
        train = train_cma_mae
    elif cfg.algo_name == "cma_mega":
        train = train_cma_mega
    elif cfg.algo_name == "cma_maega":
        train = train_cma_maega
    elif cfg.algo_name == "nslc":
        train = train_nslc
    elif cfg.algo_name == "pga_me":
        train = train_pga_me
    elif cfg.algo_name == "dns":
        train = train_dns
    else:
        raise ValueError(f"Algorithm {cfg.algo_name} not found.")

    # Run the training
    final_solutions, logs, artifacts = train(cfg, task)

    # Save results
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Saving results to: {output_dir}")

    # Save final solutions
    solution_path = os.path.join(output_dir, "solutions.npy")
    np.save(solution_path, np.asarray(final_solutions))

    # Save logs and artifacts
    log_path = os.path.join(output_dir, "logs.pkl")
    with open(log_path, "wb") as f:
        pickle.dump(logs, f)
    artifacts_path = os.path.join(output_dir, "artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
