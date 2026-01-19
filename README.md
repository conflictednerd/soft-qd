# Soft Quality-Diversity Optimization (SQUAD)

This repository contains the reference implementation of **Soft QD Using Approximated Diversity** , introduced in the paper:

> **Soft Quality-Diversity Optimization**  
> *[arXiv:2512.00810](https://arxiv.org/abs/2512.00810)*

**SQUAD** is a Quality-Diversity (QD) algorithm that optimizes the **Soft QD Score**, a discretization-free objective that unifies diversity and performance and enables gradient-based optimization. It enables efficient optimization of a population of diverse solutions for differentiable QD tasks.

## Installation

This codebase is primarily built on **[JAX](https://docs.jax.dev/en/latest/index.html)**. Some of the baseline QD algorithms are implemented using **[pyribs](https://pyribs.org/)**.

### Requirements

- Python 3.10

Install dependencies:

```bash
pip install -r requirements.txt
````

Notes:

* JAX, CUDA, and numpy versions must be compatible and match the specified ones.
* `flaxmodels` is installed from a pinned Git commit (see `requirements.txt`). You can install it by going to the `flaxmodels` directory and running `pip install -e .`.

If you install JAX manually, use:

```bash
pip install "jax[cuda]"
pip install flax
```

Make sure that the installed `jax`, `jaxlib`, and CUDA versions are consistent.

## Repository Structure

```text
src/
  main.py           # Entry point for running experiments
  evaluate.py       # Evaluation and metrics
  qd/               # QD algorithms
    softqd.py       # SQUAD implementation
    cma_mae.py
    cma_maega.py
    cma_mega.py
    dns.py
    nslc.py
    pga_me.py
  tasks/            # Optimization tasks
    base.py
    rastrigin.py
    sphere.py
    lsi.py
    image_rendering.py
configs/
  config.yaml       # Global configuration
  task/             # Task-specific configs
scripts/            # Experiment and ablation scripts
assets/             # Assets used by some tasks
```

## Algorithms

Implemented QD algorithms include:
* **[Soft QD (SQUAD)](https://arxiv.org/abs/2512.00810)**
* [CMA-ME](https://arxiv.org/abs/1912.02400)
* [CMA-MAE](https://arxiv.org/abs/2205.10752)
* [CMA-MEGA](https://arxiv.org/abs/2106.03894)
* [CMA-MAEGA](https://arxiv.org/abs/2205.10752)
* [DNS](https://arxiv.org/abs/2502.00593)
* [NSLC](https://dl.acm.org/doi/10.1145/2001576.2001606)
* [PGA-ME](https://github.com/ollebompa/PGA-MAP-Elites) (For DQD tasks)

Each algorithm is implemented as a single-file, self-contained module under `src/qd/`.

## Tasks

We also implement several DQD benchmark tasks in JAX:
* Sphere
* Rastrigin (easy / mid / hard)
* Latent Space Illumination (LSI)
* Image rendering

Tasks are defined in `src/tasks/` and configured via YAML files in `configs/task/`.

## Running Experiments

To run an experiment you should run 
```bash
python -m src.main
```

Experiments are configured via Hydra. For example, to run CMA-MAE on the image rendering task you can run:

```bash
python -m src.main --config-name "cma_mae.yaml" task=image_rendering task.normalized_descriptors=True
```

- The `scripts/` directory contains shell scripts used to reproduce experiments and ablations reported in the paper. You can use these commands as references.
- Note that other than SQUAD, all the algorithms must use `task.normalized_descriptors=True` since they need a bounded behavior space to define the QD archive.

## Citation

If you use this code in your work, please cite:

```bibtex
@article{hedayatian2025soft,
  title={Soft Quality-Diversity Optimization},
  author={Hedayatian, Saeed and Nikolaidis, Stefanos},
  journal={arXiv preprint arXiv:2512.00810},
  year={2025}
}
```
