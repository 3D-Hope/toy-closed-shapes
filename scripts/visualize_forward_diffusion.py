"""
Script to visualize the forward diffusion process (noising) for multiple scenes.
Shows how clean scenes are progressively corrupted with noise over timesteps.

Usage:
    python scripts/visualize_forward_diffusion.py \
        load=<checkpoint_path> \
        num_scenes=5 \
        output_subdir=forward_diffusion_viz \
        num_timesteps=10
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
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


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get configuration values
    num_scenes = cfg.get("num_scenes", 5)
    num_timesteps = cfg.get("num_timesteps", 10)  # Number of timesteps to visualize
    print(f"[INFO] Number of scenes to visualize: {num_scenes}")
    print(f"[INFO] Number of timesteps to visualize: {num_timesteps}")

    # Get output directory for batch
    output_subdir = cfg.get("output_subdir", "forward_diffusion_viz")
    print(f"[INFO] Will save visualizations to subdirectory: {output_subdir}")

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

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

    # Load the checkpoint.
    load_id = cfg.load
    if is_run_id(load_id):
        # Download the checkpoint from wandb.
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        version = cfg.get("checkpoint_version")
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
        # Use local path.
        checkpoint_path = Path(load_id)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load datasets for postprocessing
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

    # Create a CustomSceneDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    print(f"[INFO] Dataset size: {len(custom_dataset)}")
    
    # Create indices with resampling if needed
    dataset_size = len(custom_dataset)
    num_scenes_to_sample = num_scenes
    
    if num_scenes_to_sample <= dataset_size:
        indices = list(range(num_scenes_to_sample))
    else:
        print(f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes.")
        print(f"[INFO] Will resample with replacement to generate {num_scenes_to_sample} scenes.")
        indices = [i % dataset_size for i in range(num_scenes_to_sample)]
    
    # Create subset of dataset with the indices
    from torch.utils.data import Subset
    limited_dataset = Subset(custom_dataset, indices)
    
    # Store the actual dataset indices for each sample
    sampled_dataset_indices = indices.copy()
    
    print(f"[DEBUG] Full dataset size: {dataset_size}")
    print(f"[DEBUG] Sampling {num_scenes_to_sample} scenes")
    print(f"[DEBUG] Sample indices: {sampled_dataset_indices}")
    
    # Use batch size from config
    batch_size = cfg.experiment.get("test", {}).get(
        "batch_size", cfg.experiment.validation.batch_size
    )
    print(f"[DEBUG] Using batch size: {batch_size}")
    
    # Create a dataloader for the limited dataset
    dataloader = torch.utils.data.DataLoader(
        limited_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Build experiment and load algo
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    algo = experiment._build_algo(ckpt_path=checkpoint_path)
    
    print(f"[DEBUG] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = algo.load_state_dict(state_dict, strict=False)
    
    # Load EMA if available
    if getattr(algo, "ema", None) and "ema_state_dict" in ckpt:
        algo.ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"[DEBUG] Loaded EMA state dict")
    
    print(f"[DEBUG] Missing keys: {len(missing)}")
    print(f"[DEBUG] Unexpected keys: {len(unexpected)}")
    
    device = algo.device
    print(f"[INFO] Running on device: {device}")
    
    algo.put_model_in_eval_mode()
    
    # Create output subdirectory
    batch_output_dir = output_dir / output_subdir
    batch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Import scheduler types
    from diffusers import DDIMScheduler, DDPMScheduler
    
    room_type = getattr(cfg.dataset.data, "room_type", "bedroom")
    
    # Determine number of classes
    if cfg.dataset.data.room_type == "livingroom":
        n_classes = 25
    else:
        n_classes = 22
    
    # Select timesteps to visualize (evenly spaced from 0 to max)
    max_timesteps = cfg.algorithm.noise_schedule.num_train_timesteps
    if num_timesteps > max_timesteps:
        num_timesteps = max_timesteps
        print(f"[WARNING] Requested {num_timesteps} timesteps but max is {max_timesteps}. Using {max_timesteps}.")
    
    # Create evenly spaced timesteps including 0 and max
    timesteps_to_visualize = np.linspace(0, max_timesteps - 1, num_timesteps, dtype=int)
    print(f"[INFO] Visualizing timesteps: {timesteps_to_visualize}")
    
    # Process batches from dataloader
    global_scene_idx = 0
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
        current_batch_size = batch_data["scenes"].shape[0]
        batch_dataset_indices = batch_data["idx"].cpu().numpy() if isinstance(batch_data["idx"], torch.Tensor) else batch_data["idx"]
        print(f"\n[INFO] ========== Processing batch {batch_idx+1}/{len(dataloader)} ({current_batch_size} scenes) ===========")
        
        with torch.no_grad():
            # Get clean scenes (x0)
            x0 = batch_data["scenes"].to(device)  # Shape: (B, N, V)
            
            # Store forward diffusion trajectory for entire batch
            forward_trajectories_batch = []
            
            # Apply forward diffusion for each selected timestep
            for t in tqdm(timesteps_to_visualize, desc=f"Forward diffusion batch {batch_idx+1}"):
                # Sample noise
                noise = torch.randn_like(x0)
                
                # Add noise according to the noise schedule
                # Using the noise scheduler's add_noise method
                timesteps_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                xt = algo.noise_scheduler.add_noise(x0, noise, timesteps_tensor)
                
                forward_trajectories_batch.append(xt.clone())
            
            # Stack: (T, B, N, V)
            forward_trajectories_batch = torch.stack(forward_trajectories_batch, dim=0)
        
        # Process each scene in the batch separately
        for scene_in_batch_idx in range(current_batch_size):
            dataset_idx = int(batch_dataset_indices[scene_in_batch_idx])
            print(f"\n[INFO] Processing scene {global_scene_idx+1}/{num_scenes_to_sample} (dataset_idx={dataset_idx})")
            
            # Extract trajectory for this scene: (T, N, V)
            trajectory_single = forward_trajectories_batch[:, scene_in_batch_idx, :, :].detach().cpu().numpy()
            
            # Postprocess each timestep
            bbox_params_list = []
            for t_idx in range(trajectory_single.shape[0]):
                # scene_normalized = torch.from_numpy(trajectory_single[t_idx]).unsqueeze(0).to(device)
                
                # Denormalize
                # scene_unnormalized = algo.dataset.inverse_normalize_scenes(scene_normalized)
                # scene_unnormalized = scene_normalized.squeeze(0).cpu().numpy()
                scene_unnormalized = trajectory_single[t_idx] # (N, V) # in reality this is still normalized, need to unnormalize using post_process
                
                # print(f"[DEBUG] scene unnormalized {scene_unnormalized.shape}")
                # Convert to bbox params
                if room_type == "bedroom":
                    translations = scene_unnormalized[:, :3]
                    sizes = scene_unnormalized[:, 3:6]
                    rotations = scene_unnormalized[:, 6:8]
                    class_labels = np.argmax(scene_unnormalized[:, 8:8+n_classes], axis=-1)
                    
                elif room_type == "livingroom":
                    translations = scene_unnormalized[:, :3]
                    sizes = scene_unnormalized[:, 3:6]
                    rotations = scene_unnormalized[:, 6:8]
                    class_labels = np.argmax(scene_unnormalized[:, 8:8+n_classes], axis=-1)
                else:
                    raise ValueError(f"Unknown room type: {room_type}")
                
                bbox_params_list.append(
                {
                    "class_labels": np.array(class_labels)[None, :],
                    "translations": np.array(translations)[None, :],
                    "sizes": np.array(sizes)[None, :],
                    "angles": np.array(rotations)[None, :],
                    # "objfeats_32": np.array(objfeats_32)[None, :]
                    # if objfeats_32 is not None
                    # else None,
                }
            )            
            # Post-process layouts
            layout_list = []
            successful_timesteps = []
            
            for t_idx, bbox_params_dict in enumerate(bbox_params_list):
                try:
                    layout = encoded_dataset.post_process(bbox_params_dict)
                    layout_list.append({k: v[0] for k, v in layout.items()})
                    successful_timesteps.append(timesteps_to_visualize[t_idx])
                except Exception as e:
                    print(f"[WARNING] Failed to post-process timestep {timesteps_to_visualize[t_idx]}: {e}")
                    continue
            
            # Use the actual dataset index for ThreedFrontResults
            indices_list = [dataset_idx] * len(layout_list)
            
            # Create ThreedFrontResults
            threed_front_results = ThreedFrontResults(
                raw_train_dataset, raw_dataset, config, indices_list, layout_list
            )
            
            # Save trajectory
            output_pkl_path = batch_output_dir / f"forward_diffusion_scene{global_scene_idx}.pkl"
            pickle.dump(threed_front_results, open(output_pkl_path, "wb"))
            
            print(f"[SUCCESS] Saved forward diffusion trajectory with {len(layout_list)} timesteps to: {output_pkl_path}")
            print(f"[INFO] Timesteps: {successful_timesteps}")
            
            global_scene_idx += 1
    
    print(f"\n[COMPLETE] Generated {global_scene_idx} forward diffusion trajectories in: {batch_output_dir}")
    print(f"\n[INFO] Timesteps visualized: {timesteps_to_visualize.tolist()}")
    print(f"[INFO] You can now render each trajectory using:")
    print(f"       python ../ThreedFront/scripts/render_results.py --retrieve_by_size --no_texture --without_floor {batch_output_dir}/forward_diffusion_scene<N>.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
