# import pathlib

# from typing import Optional, Union

# from lightning.pytorch.loggers.wandb import WandbLogger
# from omegaconf import DictConfig

# from .exp_base import BaseExperiment
# from .scene_diffusion import SceneDiffusionExperiment

# # each key has to be a yaml file under '[project_root]/configurations/experiment'
# # without .yaml suffix
# exp_registry = dict(
#     scene_diffusion=SceneDiffusionExperiment,
# )


# def build_experiment(
#     cfg: DictConfig,
#     logger: Optional[WandbLogger] = None,
#     ckpt_path: Optional[Union[str, pathlib.Path]] = None,
# ) -> BaseExperiment:
#     """
#     Build an experiment instance based on registry
#     :param cfg: configuration file
#     :param logger: optional logger for the experiment
#     :param ckpt_path: optional checkpoint path for saving and loading
#     :return:
#     """
#     if cfg.experiment._name not in exp_registry:
#         raise ValueError(
#             f"Experiment {cfg.experiment._name} not found in registry "
#             f"{list(exp_registry.keys())}. Make sure you register it correctly in "
#             "'experiments/__init__.py' under the same name as yaml file."
#         )

import pathlib

from typing import Optional, Union

from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

#     return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
from steerable_scene_generation.experiments.scene_diffusion import (
    SceneDiffusionExperiment,
)

from .custom_scene_diffusion import CustomSceneDiffusionExperiment
from .exp_base import BaseExperiment
from .scene_diffusion import SceneDiffusionExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment'
# without .yaml suffix
exp_registry = dict[str, type[SceneDiffusionExperiment]](
    scene_diffusion=SceneDiffusionExperiment,
    custom_scene_diffusion=CustomSceneDiffusionExperiment,  # Add the custom experiment
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    # Check if we're using the custom dataset
    if cfg.dataset._name == "custom_scene":
        # Use the custom experiment when the dataset is custom_scene
        print(f"[DEBUG] Using custom scene diffusion experiment")
        return CustomSceneDiffusionExperiment(cfg, logger, ckpt_path)

    # Otherwise, use the standard   registry lookup
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry "
            f"{list(exp_registry.keys())}. Make sure you register it correctly in "
            "'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
