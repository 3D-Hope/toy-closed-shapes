"""
Script for sampling from a trained model using the custom scene dataset.
"""

import logging
import os
import pickle
import sys

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults

from steerable_scene_generation.datasets.custom_scene import get_dataset_raw_and_encoded
from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
    CustomDataset,
    update_data_file_paths,
)
from steerable_scene_generation.experiments import build_experiment
from steerable_scene_generation.utils.ckpt_utils import (
    download_latest_or_best_checkpoint,
    download_version_checkpoint,
    is_run_id,
)
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
from steerable_scene_generation.utils.omegaconf import register_resolvers
import matplotlib.pyplot as plt

# Add logging filters.
filter_drake_vtk_warning()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ[
    "HF_HOME"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
os.environ[
    "HF_DATASETS_CACHE"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

    # Set random seed.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get configuration values with defaults.
    num_scenes = cfg.get("num_scenes", 1)
    print(f"[DEBUG] Number of scenes to sample: {num_scenes}")

    # Set predict mode.
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False
    # print(f"[DEBUG] Predict config: {cfg}")
    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(f"[DEBUG] cfg_choice: {cfg_choice}")

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")

    # Initialize wandb.
    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)
    load_id = cfg.load
    name = f"custom_sampling_{load_id}"
    wandb.init(
        name=name,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg),
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
    )

    # Load the checkpoint.
    if is_run_id(load_id):
        # Download the checkpoint from wandb.
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        version = cfg["checkpoint_version"]
        print(f"[Ashok] checkpoint version from cfg: {version}")
        if version is not None and isinstance(version, int):
            print(f"[Ashok] downloading checkpoint version: {version}")
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )

        else:
            print(
                f"[Ashok] no checkpoint version specified, using_best {cfg.get('use_best', False)}"
            )
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path,
                download_dir=download_dir,
                use_best=cfg.get("use_best", False),
            )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    batch_size = cfg.num_scenes
    x = CustomDataset(config, split="train").raw_dataset
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    plt.scatter(x[:, 0], x[:, 1], s=1)
    plt.axis("equal")
    plt.savefig(output_dir / "training_points.png")
    plt.close()
    dataset = CustomDataset(config, split="test")
    # Create a dataloader for the limited dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100000,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=cfg.experiment.test.pin_memory,
    )


    # Build experiment with custom dataset
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)

    try:
        print("[DEBUG] Starting to sample scenes...")
        # Sample scenes from the model
        sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

        print(f"[DEBUG] Sampled scenes shape: {sampled_scenes.shape}")


        # with open(output_dir / "raw_sampled_scenes.pkl", "wb") as f:
        #     pickle.dump(sampled_scenes, f)
        x = sampled_scenes.detach().cpu().numpy()
        
        # Normalize 
        # x = (x * std + mean)

        plt.scatter(x[:, 0], x[:, 1], s=1)
        plt.axis("equal")
        plt.savefig(output_dir / "sampled_points.png")
        plt.close()
        print(f"gt saved at {output_dir / 'training_points.png'}")
        
        print(f"pred saved at {output_dir / 'sampled_points.png'}")

    except Exception as e:
        logging.error(f"Error during sampling: {str(e)}")
        # Still try to save any partial results
        try:
            if "sampled_scenes" in locals():
                sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
                text_path = output_dir / "partial_sampled_scenes.txt"
                with open(text_path, "w") as f:
                    f.write(
                        f"Partial sampled scenes shape: {sampled_scenes_np.shape}\n\n"
                    )
                    f.write(
                        np.array2string(
                            sampled_scenes_np, threshold=np.inf, precision=6
                        )
                    )
                logging.info(f"Saved partial sampled scenes to {text_path}")
                wandb.save(str(text_path))
        except:
            pass

        raise e

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
