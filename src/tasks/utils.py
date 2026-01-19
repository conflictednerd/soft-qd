from typing import Dict
from src.tasks.sphere import SphereTask
from src.tasks.rastrigin import RastriginTask
from src.tasks.image_rendering import ImageRenderingTask
from src.tasks.lsi import LatentIlluminationTask


def create_task(cfg: Dict):
    """Instantiates the task from the task configuration."""
    task_name = cfg["task_name"]

    if task_name == "sphere":
        return SphereTask(
            solution_dim=cfg["solution_dim"],
            descriptor_dim=cfg["descriptor_dim"],
            normalized_descriptors=cfg["normalized_descriptors"],
        )
    elif task_name == "rastrigin":
        return RastriginTask(
            solution_dim=cfg["solution_dim"],
            descriptor_dim=cfg["descriptor_dim"],
            normalized_descriptors=cfg["normalized_descriptors"],
        )
    elif task_name == "image_rendering":
        return ImageRenderingTask(
            target_image=cfg["target_image"],
            num_circles=cfg["num_circles"],
            canvas_width=cfg["canvas_width"],
            canvas_height=cfg["canvas_height"],
            softness=cfg["softness"],
            objective_type=cfg["objective_type"],
            normalized_descriptors=cfg["normalized_descriptors"],
        )
    elif task_name == "lsi":
        return LatentIlluminationTask(
            target_prompt=cfg["target_prompt"],
            descriptor_prompts=cfg["descriptor_prompts"],
            normalized_descriptors=cfg["normalized_descriptors"],
            stylegan_model_name=cfg["stylegan_model_name"],
            clip_model_name=cfg["clip_model_name"],
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
