"""
Script to generate a denoising trajectory for a single scene and save each timestep
as a separate scene in a pickle file (compatible with rendering scripts).

Usage:
    python scripts/save_trajectory_as_scenes.py \
        load=<checkpoint_path_or_run_id> \
        scene_idx=0 \
        output_dir=trajectory_outputs
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from tqdm import tqdm

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


def postprocess_scene_to_bbox_params(scene_np, n_classes=22):
    """
    Convert a single scene tensor to bbox_params_dict format.
    
    Args:
        scene_np: numpy array of shape (N, V) - single scene
        n_classes: number of object classes
        
    Returns:
        dict with keys: class_labels, translations, sizes, angles, objfeats_32
    """
    class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
    
    for j in range(scene_np.shape[0]):
        class_label_idx = np.argmax(scene_np[j, :n_classes])
        if class_label_idx != n_classes - 1:  # ignore if empty token
            ohe = np.zeros(n_classes - 1)
            ohe[class_label_idx] = 1
            class_labels.append(ohe)
            translations.append(scene_np[j, n_classes : n_classes + 3])
            sizes.append(scene_np[j, n_classes + 3 : n_classes + 6])
            angles.append(scene_np[j, n_classes + 6 : n_classes + 8])
            try:
                objfeats_32.append(scene_np[j, n_classes + 8 : n_classes + 8 + 32])
            except Exception:
                objfeats_32 = None
    
    return {
        "class_labels": np.array(class_labels)[None, :],
        "translations": np.array(translations)[None, :],
        "sizes": np.array(sizes)[None, :],
        "angles": np.array(angles)[None, :],
        "objfeats_32": np.array(objfeats_32)[None, :] if objfeats_32 is not None else None,
    }


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Resolve the config
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Check if load path is provided
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get scene index to generate trajectory for
    scene_idx = cfg.get("scene_idx", 0)
    print(f"[INFO] Generating trajectory for scene index: {scene_idx}")

    # Set predict mode
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")

    # Load the checkpoint
    load_id = cfg.load
    if is_run_id(load_id):
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        version = cfg.get("checkpoint_version", None)
        
        if version is not None and isinstance(version, int):
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )
        else:
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path,
                download_dir=download_dir,
                use_best=cfg.get("use_best", False),
            )
    else:
        checkpoint_path = Path(load_id)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load datasets
    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"], config),
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    print(f"[INFO] Dataset size: {len(raw_dataset)}")

    # Create CustomDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    # Validate scene index
    if scene_idx >= len(custom_dataset):
        raise ValueError(
            f"scene_idx={scene_idx} is out of range. Dataset has {len(custom_dataset)} scenes."
        )

    # Build experiment
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    algo = experiment.algo
    
    # Get the conditioning data for this specific scene
    scene_data = custom_dataset[scene_idx]
    
    # Create a batch with just this one scene (but we need batch_size copies for RL trajectory generation)
    batch_size = 1  # Generate trajectory for single scene
    batch = {
        "scenes": scene_data["scenes"].unsqueeze(0).repeat(batch_size, 1, 1).to(algo.device),
        "idx": torch.tensor([scene_idx] * batch_size, device=algo.device),
    }
    
    # Add fpbpn if available
    if "fpbpn" in scene_data:
        batch["fpbpn"] = scene_data["fpbpn"]

    print(f"[INFO] Generating trajectory for scene {scene_idx}...")
    
    # Generate trajectory using the RL trainer's method
    with torch.no_grad():
        trajectories, log_probs, cond_dict = algo.generate_trajs_for_ddpo(
            batch=batch
        )
    
    # trajectories shape: (B, T+1, N, V) where T+1 is number of timesteps including initial noise
    print(f"[INFO] Generated trajectory shape: {trajectories.shape}")
    print(f"[INFO] Number of timesteps: {trajectories.shape[1]}")
    
    # Unnormalize trajectories
    trajectories_unnorm = algo.dataset.inverse_normalize_scenes(
        trajectories.reshape(-1, trajectories.shape[-1])
    ).reshape(trajectories.shape)
    
    # Convert to numpy
    trajectories_np = trajectories_unnorm.cpu().numpy()
    
    # Take the first sample from batch (all should be from same scene_idx)
    trajectory_single = trajectories_np[0]  # Shape: (T+1, N, V)
    
    print(f"[INFO] Processing {trajectory_single.shape[0]} timesteps...")
    
    # Determine number of classes
    n_classes = 25 if cfg.dataset.data.room_type == "livingroom" else 22
    
    # Process each timestep as a separate scene
    all_timestep_scenes = []
    successful_timesteps = []
    
    for t_idx in tqdm(range(trajectory_single.shape[0]), desc="Processing timesteps"):
        scene_at_t = trajectory_single[t_idx]  # Shape: (N, V)
        
        # Check for NaNs
        if np.isnan(scene_at_t).any():
            print(f"[WARNING] Skipping timestep {t_idx} due to NaN values")
            continue
        
        try:
            # Convert to bbox_params format
            bbox_params_dict = postprocess_scene_to_bbox_params(scene_at_t, n_classes)
            
            # Post-process using encoded_dataset
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v[0] for k, v in boxes.items()}
            
            all_timestep_scenes.append(bbox_params)
            successful_timesteps.append(t_idx)
            
        except Exception as e:
            print(f"[WARNING] Skipping timestep {t_idx} due to error: {e}")
            continue
    
    print(f"[INFO] Successfully processed {len(all_timestep_scenes)} timesteps")
    
    # Create ThreedFrontResults object
    # Use the same scene_idx for all timesteps (as requested)
    indices_for_all_timesteps = [scene_idx] * len(all_timestep_scenes)
    
    threed_front_results = ThreedFrontResults(
        raw_train_dataset,
        raw_dataset,
        config,
        indices_for_all_timesteps,  # All have same index
        all_timestep_scenes,
    )
    
    # Save to pickle
    output_path = output_dir / f"trajectory_scene_{scene_idx}_as_scenes.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(threed_front_results, f)
    
    print(f"[SUCCESS] Saved trajectory with {len(all_timestep_scenes)} timesteps to: {output_path}")
    
    # Save metadata
    metadata = {
        "scene_idx": scene_idx,
        "num_timesteps": len(all_timestep_scenes),
        "timestep_indices": successful_timesteps,
        "checkpoint_path": str(checkpoint_path),
        "trajectory_shape": trajectory_single.shape,
    }
    
    metadata_path = output_dir / f"trajectory_scene_{scene_idx}_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"[INFO] Saved metadata to: {metadata_path}")
    
    # Also save raw trajectory for reference
    raw_trajectory_path = output_dir / f"trajectory_scene_{scene_idx}_raw.pkl"
    with open(raw_trajectory_path, "wb") as f:
        pickle.dump({
            "trajectory": trajectory_single,
            "scene_idx": scene_idx,
            "log_probs": log_probs[0].cpu().numpy(),
        }, f)
    
    print(f"[INFO] Saved raw trajectory to: {raw_trajectory_path}")
    print("\n[DONE] You can now render this trajectory using existing rendering scripts!")
    print(f"      python scripts/render_samples_semantic.py {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
