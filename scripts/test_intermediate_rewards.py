"""
Script to test intermediate rewards during denoising trajectory.

This script tests the feasibility of computing rewards at intermediate denoising steps.
For each intermediate step, we:
1. Denoise to that step (e.g., 50 iterations)
2. Take the noisy sample at that step
3. Quickly denoise to completion using few steps (e.g., 5 steps)
4. Compute reward for the fully denoised sample
5. Save all intermediate samples and their rewards

This helps determine if intermediate rewards can guide RL training.
"""

import logging
import os
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
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
from steerable_scene_generation.algorithms.scene_diffusion.ddpo_helpers import (
    ddim_step_with_logprob,
    ddpm_step_with_logprob,
)

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


def quick_denoise_to_completion(
    algo,
    xt_intermediate,
    current_timestep_idx,
    data_batch,
    cfg,
    device,
    use_ema,
    num_quick_steps=10,
):
    """
    Quickly denoise an intermediate noisy sample to completion.
    
    Args:
        algo: The algorithm/model
        xt_intermediate: Intermediate noisy sample (B, N, V)
        current_timestep_idx: Current timestep index in the full schedule
        data_batch: Conditioning data
        cfg: Config
        device: Device
        use_ema: Whether to use EMA model
        num_quick_steps: Number of quick denoising steps to completion
        
    Returns:
        Fully denoised sample (B, N, V)
    """
    from diffusers import DDIMScheduler, DDPMScheduler
    
    # Create a new scheduler with fewer steps for quick denoising
    if isinstance(algo.noise_scheduler, DDIMScheduler):
        quick_scheduler = DDIMScheduler(**algo.noise_scheduler.config)
        quick_scheduler.set_timesteps(num_quick_steps, device=device)
    else:
        raise NotImplementedError("Quick denoising only implemented for DDIMScheduler.")
        # For DDPM, we'll use DDIM for quick denoising as it's more efficient
        # quick_scheduler = DDIMScheduler(
        #     num_train_timesteps=algo.noise_scheduler.config.num_train_timesteps,
        #     beta_start=algo.noise_scheduler.config.beta_start,
        #     beta_end=algo.noise_scheduler.config.beta_end,
        #     beta_schedule=algo.noise_scheduler.config.beta_schedule,
        #     trained_betas=algo.noise_scheduler.config.trained_betas,
        #     clip_sample=algo.noise_scheduler.config.clip_sample,
        #     prediction_type=algo.noise_scheduler.config.prediction_type,
        # )
        # quick_scheduler.set_timesteps(num_quick_steps, device=device)
    
    # Get the timestep value at current position in original schedule
    original_timesteps = algo.noise_scheduler.timesteps
    if current_timestep_idx >= len(original_timesteps):
        # Already at the end
        return xt_intermediate
    
    current_t_value = original_timesteps[current_timestep_idx].item()
    
    # Find closest timestep in quick scheduler that's >= current_t_value
    quick_timesteps = quick_scheduler.timesteps
    start_idx = 0
    for i, t in enumerate(quick_timesteps):
        if t.item() <= current_t_value:
            start_idx = i
            break
    
    # Denoise from this point
    xt = xt_intermediate.clone()
    
    with torch.no_grad():
        for t in quick_timesteps[start_idx:]:
            residual = algo.predict_noise(xt, t, cond_dict=data_batch, use_ema=use_ema)
            
            # Always use DDIM step for quick denoising
            xt_next, _ = ddim_step_with_logprob(
                scheduler=quick_scheduler,
                model_output=residual,
                timestep=t,
                sample=xt,
                eta=0.0,  # Deterministic for consistency
            )
            xt = xt_next
    
    return xt


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
    num_scenes = cfg.get("num_scenes", 1)
    print(f"[INFO] Number of scenes to test: {num_scenes}")

    # Get intermediate steps to test
    intermediate_steps = cfg.get("intermediate_steps", [10, 25, 50, 75, 100, 125, 150])
    print(f"[INFO] Testing intermediate steps: {intermediate_steps}")
    
    # Number of quick denoising steps
    num_quick_steps = cfg.get("num_quick_steps", 10)
    print(f"[INFO] Quick denoising steps: {num_quick_steps}")

    # Get output directory
    output_subdir = cfg.get("output_subdir", "intermediate_rewards_test")
    print(f"[INFO] Will save results to subdirectory: {output_subdir}")

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

    # Create dataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    print(f"[INFO] Dataset size: {len(custom_dataset)}")
    
    # Create limited dataset
    dataset_size = len(custom_dataset)
    num_scenes_to_sample = num_scenes
    
    if num_scenes_to_sample <= dataset_size:
        indices = list(range(num_scenes_to_sample))
    else:
        print(f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes.")
        indices = [i % dataset_size for i in range(num_scenes_to_sample)]
    
    from torch.utils.data import Subset
    limited_dataset = Subset(custom_dataset, indices)
    
    # Create dataloader
    batch_size = cfg.experiment.get("test", {}).get(
        "batch_size", cfg.experiment.validation.batch_size
    )
    
    dataloader = torch.utils.data.DataLoader(
        limited_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    # Build experiment and algo
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    algo = experiment._build_algo(ckpt_path=checkpoint_path)
    
    print(f"[DEBUG] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = algo.load_state_dict(state_dict, strict=False)
    
    if getattr(algo, "ema", None) and "ema_state_dict" in ckpt:
        algo.ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"[DEBUG] Loaded EMA state dict")
    
    device = algo.device
    print(f"[INFO] Using device: {device}")

    algo.put_model_in_eval_mode()
    use_ema = cfg.algorithm.ema.use and cfg.algorithm.get("test", {}).get("use_ema", True)
    print(f"[INFO] Using EMA model: {use_ema}")
    
    # Create output directory
    batch_output_dir = output_dir / output_subdir
    batch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Import scheduler types
    from diffusers import DDIMScheduler, DDPMScheduler
    
    room_type = getattr(cfg.dataset.data, "room_type", "bedroom")
    if isinstance(algo.noise_scheduler, DDIMScheduler):
        algo.noise_scheduler.set_timesteps(
            cfg.algorithm.noise_schedule.ddim.num_inference_timesteps, device=device
        )
    
    num_objects_per_scene = (
        cfg.dataset.max_num_objects_per_scene
        + cfg.algorithm.num_additional_tokens_for_sampling
    )
    
    # Determine number of classes
    if cfg.dataset.data.room_type == "livingroom":
        n_classes = 25
    else:
        n_classes = 22
    
    # Storage for results
    # Format: {scene_idx: {step: {"reward": float, "sample": tensor, "quick_denoised": tensor}}}
    results = {}
    
    global_scene_idx = 0
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
        current_batch_size = batch_data["scenes"].shape[0]
        batch_dataset_indices = batch_data["idx"].cpu().numpy() if isinstance(batch_data["idx"], torch.Tensor) else batch_data["idx"]
        
        print(f"\n[INFO] ========== Processing batch {batch_idx+1}/{len(dataloader)} ({current_batch_size} scenes) ===========")
        
        with torch.no_grad():
            # Sample initial noise
            if room_type == "bedroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        current_batch_size,
                        num_objects_per_scene,
                        algo.scene_vec_desc.get_object_vec_len(),
                    )
                ).to(device)
            elif room_type == "livingroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        current_batch_size,
                        num_objects_per_scene,
                        cfg.algorithm.custom.num_classes
                        + cfg.algorithm.custom.translation_dim
                        + cfg.algorithm.custom.size_dim
                        + cfg.algorithm.custom.angle_dim
                        + cfg.algorithm.custom.objfeat_dim,
                    )
                ).to(device)
            else:
                raise ValueError(f"Unknown room type: {room_type}")
            
            # Prepare conditioning batch
            data_batch = {
                "scenes": batch_data["scenes"].to(device),
                "idx": batch_data["idx"].to(device),
            }
            
            for key in ["fpbpn", "text_cond", "text_cond_coarse"]:
                if key in batch_data:
                    val = batch_data[key]
                    if key == "fpbpn" and not isinstance(val, torch.Tensor):
                        val = torch.tensor(val, device=device)
                    if isinstance(val, torch.Tensor):
                        data_batch[key] = val.to(device)
                    else:
                        data_batch[key] = val
            
            # Store intermediate samples
            intermediate_samples = {}  # {step_idx: tensor}
            
            # Full denoising loop
            for t_idx, t in enumerate(
                tqdm(algo.noise_scheduler.timesteps, desc=f"Denoising batch {batch_idx+1}", leave=False)
            ):
                residual = algo.predict_noise(xt, t, cond_dict=data_batch, use_ema=use_ema)
                
                if isinstance(algo.noise_scheduler, DDPMScheduler):
                    xt_next, log_prop = ddpm_step_with_logprob(
                        scheduler=algo.noise_scheduler,
                        model_output=residual,
                        timestep=t,
                        sample=xt,
                    )
                else:
                    xt_next, log_prop = ddim_step_with_logprob(
                        scheduler=algo.noise_scheduler,
                        model_output=residual,
                        timestep=t,
                        sample=xt,
                        eta=algo.cfg.noise_schedule.ddim.eta,
                    )
                
                xt = xt_next
                
                # Save intermediate samples at specified steps
                if t_idx in intermediate_steps:
                    intermediate_samples[t_idx] = xt.clone()
                    print(f"[INFO] Saved intermediate sample at step {t_idx}")
            
            # Add final denoised sample
            intermediate_samples[len(algo.noise_scheduler.timesteps)] = xt.clone()
            
            # Now quick denoise each intermediate sample and compute rewards
            print(f"\n[INFO] Quick denoising and computing rewards for intermediate samples...")
            
            # Dictionary to collect all layouts for each scene: {scene_in_batch_idx: [(step_idx, layout, reward), ...]}
            scene_layouts = {i: [] for i in range(current_batch_size)}
            
            for step_idx in tqdm(sorted(intermediate_samples.keys()), desc="Processing intermediate steps"):
                xt_intermediate = intermediate_samples[step_idx]
                
                # Quick denoise to completion (skip for final step)
                if step_idx == len(algo.noise_scheduler.timesteps):
                    x0_denoised = xt_intermediate
                else:
                    x0_denoised = quick_denoise_to_completion(
                        algo=algo,
                        xt_intermediate=xt_intermediate,
                        current_timestep_idx=step_idx,
                        data_batch=data_batch,
                        cfg=cfg,
                        device=device,
                        use_ema=use_ema,
                        num_quick_steps=num_quick_steps,
                    )
                
                # Compute reward for each scene in batch
                x0_denoised_normalized = x0_denoised  # Already in normalized space
                
                # Compute rewards (use algo's reward computation)
                try:
                    rewards = algo.compute_rewards_from_trajs(
                        trajectories=x0_denoised_normalized.unsqueeze(1),  # Add time dimension (B, 1, N, V)
                        cond_dict=data_batch,
                        are_trajectories_normalized=True,
                    )
                    rewards = rewards.cpu().numpy()
                except Exception as e:
                    raise RuntimeError(f"Failed to compute rewards at step {step_idx}: {e}")
                
                # Process each scene in batch and collect layouts
                for scene_in_batch_idx in range(current_batch_size):
                    scene_id = global_scene_idx + scene_in_batch_idx
                    dataset_idx = int(batch_dataset_indices[scene_in_batch_idx])
                    
                    if scene_id not in results:
                        results[scene_id] = {}
                    
                    results[scene_id][step_idx] = {
                        "reward": float(rewards[scene_in_batch_idx]),
                        "intermediate_sample": xt_intermediate[scene_in_batch_idx].cpu(),
                        "quick_denoised_sample": x0_denoised[scene_in_batch_idx].cpu(),
                    }
                    
                    print(f"[INFO] Scene {scene_id}, Step {step_idx}: Reward = {rewards[scene_in_batch_idx]:.4f}")
                    
                    # Post-process to get bbox params for this scene at this step
                    save_scenes = cfg.get("save_scenes", False)
                    if save_scenes:
                        scene_denoised = x0_denoised[scene_in_batch_idx].detach().cpu().numpy()  # (N, V)
                        
                        # Extract bbox parameters
                        class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
                        
                        for j in range(scene_denoised.shape[0]):
                            class_label_idx = np.argmax(scene_denoised[j, 8:8+n_classes])
                            if class_label_idx != n_classes - 1:  # Not empty
                                ohe = np.zeros(n_classes - 1)
                                ohe[class_label_idx] = 1
                                class_labels.append(ohe)
                                translations.append(scene_denoised[j, :3])
                                sizes.append(scene_denoised[j, 3:6])
                                angles.append(scene_denoised[j, 6:8])
                                try:
                                    objfeats_32.append(scene_denoised[j, 8+n_classes:n_classes+8+32])
                                except Exception as e:
                                    objfeats_32 = None
                        
                        if len(class_labels) > 0:
                            bbox_params_dict = {
                                "class_labels": np.array(class_labels)[None, :],
                                "translations": np.array(translations)[None, :],
                                "sizes": np.array(sizes)[None, :],
                                "angles": np.array(angles)[None, :],
                                "objfeats_32": np.array(objfeats_32)[None, :] if objfeats_32 is not None else None,
                            }
                            
                            try:
                                # Post-process to get unnormalized bbox parameters
                                boxes = encoded_dataset.post_process(bbox_params_dict)
                                bbox_params = {k: v[0] for k, v in boxes.items()}
                                
                                # Store layout with step index and reward for later
                                scene_layouts[scene_in_batch_idx].append({
                                    "step_idx": step_idx,
                                    "layout": bbox_params,
                                    "reward": float(rewards[scene_in_batch_idx]),
                                    "dataset_idx": dataset_idx,
                                })
                                
                            except Exception as e:
                                print(f"[WARNING] Failed to process scene {scene_id}, step {step_idx}: {e}")
            
            # Now save one pkl per scene containing all intermediate steps
            if cfg.get("save_scenes", False):
                for scene_in_batch_idx in range(current_batch_size):
                    scene_id = global_scene_idx + scene_in_batch_idx
                    
                    if not scene_layouts[scene_in_batch_idx]:
                        continue
                    
                    # Get dataset index (same for all steps of this scene)
                    dataset_idx = scene_layouts[scene_in_batch_idx][0]["dataset_idx"]
                    
                    # Extract just the layouts in order
                    layout_list = [item["layout"] for item in scene_layouts[scene_in_batch_idx]]
                    indices_list = [dataset_idx] * len(layout_list)
                    
                    # Create ThreedFrontResults with all timesteps
                    threed_front_results = ThreedFrontResults(
                        raw_train_dataset, 
                        raw_dataset, 
                        config, 
                        indices_list,
                        layout_list
                    )
                    
                    # Save as single pkl file for this scene
                    pkl_dir = batch_output_dir / "scene_pkls"
                    pkl_dir.mkdir(exist_ok=True, parents=True)
                    pkl_path = pkl_dir / f"scene{scene_id}_all_steps.pkl"
                    
                    with open(pkl_path, "wb") as f:
                        pickle.dump(threed_front_results, f)
                    
                    # Also save metadata about steps and rewards
                    metadata = {
                        "scene_id": scene_id,
                        "dataset_idx": dataset_idx,
                        "steps": [item["step_idx"] for item in scene_layouts[scene_in_batch_idx]],
                        "rewards": [item["reward"] for item in scene_layouts[scene_in_batch_idx]],
                    }
                    metadata_path = pkl_dir / f"scene{scene_id}_metadata.json"
                    import json
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"[SUCCESS] Saved scene {scene_id} with {len(layout_list)} steps to: {pkl_path}")
        
        global_scene_idx += current_batch_size
    
    # Save results
    results_path = batch_output_dir / "intermediate_rewards_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[SUCCESS] Saved results to: {results_path}")
    
    # Analyze correlation between intermediate and final rewards
    print(f"\n[INFO] Analyzing correlation between intermediate and final rewards...")
    plot_dir = batch_output_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr, spearmanr
    
    # Get all steps and final step
    all_steps = sorted(list(results[0].keys()))
    final_step = all_steps[-1]
    intermediate_steps_list = all_steps[:-1]
    
    # Extract final rewards for all scenes
    final_rewards = np.array([results[sid][final_step]["reward"] for sid in results.keys()])
    
    # Storage for metrics
    correlations = {"pearson": [], "spearman": [], "r2": [], "mae": [], "rmse": []}
    regression_params = []  # (step, slope, intercept)
    
    # Create scatter plots for each intermediate step vs final reward
    n_intermediate = len(intermediate_steps_list)
    n_cols = 3
    n_rows = int(np.ceil(n_intermediate / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_intermediate > 1 else [axes]
    
    for idx, step in enumerate(intermediate_steps_list):
        # Get intermediate rewards
        intermediate_rewards = np.array([results[sid][step]["reward"] for sid in results.keys()])
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(intermediate_rewards, final_rewards)
        spearman_corr, spearman_p = spearmanr(intermediate_rewards, final_rewards)
        
        # Fit linear regression
        X = intermediate_rewards.reshape(-1, 1)
        y = final_rewards
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        # Compute metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Store metrics
        correlations["pearson"].append(pearson_corr)
        correlations["spearman"].append(spearman_corr)
        correlations["r2"].append(r2)
        correlations["mae"].append(mae)
        correlations["rmse"].append(rmse)
        regression_params.append((step, reg.coef_[0], reg.intercept_))
        
        # Plot scatter with regression line
        ax = axes[idx]
        ax.scatter(intermediate_rewards, final_rewards, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        
        # Plot regression line
        x_line = np.linspace(intermediate_rewards.min(), intermediate_rewards.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {reg.coef_[0]:.3f}x + {reg.intercept_:.3f}')
        
        # Add diagonal reference line (perfect prediction)
        min_val = min(intermediate_rewards.min(), final_rewards.min())
        max_val = max(intermediate_rewards.max(), final_rewards.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='Perfect prediction')
        
        ax.set_xlabel(f'Reward at Step {step}', fontsize=10)
        ax.set_ylabel(f'Final Reward (Step {final_step})', fontsize=10)
        ax.set_title(f'Step {step}\nPearson: {pearson_corr:.3f}, R²: {r2:.3f}, MAE: {mae:.3f}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_intermediate, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_intermediate_vs_final.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved scatter plots: {plot_dir / 'scatter_intermediate_vs_final.png'}")
    
    # Plot correlation bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pearson correlation
    ax = axes[0, 0]
    ax.bar(range(len(intermediate_steps_list)), correlations["pearson"], color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(intermediate_steps_list)))
    ax.set_xticklabels([f'Step {s}' for s in intermediate_steps_list], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation', fontsize=11)
    ax.set_title('Pearson Correlation: Intermediate vs Final Reward', fontsize=12, fontweight='bold')
    ax.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='0.8 threshold')
    ax.axhline(y=0.9, color='g', linestyle='--', linewidth=1, alpha=0.5, label='0.9 threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # R² score
    ax = axes[0, 1]
    ax.bar(range(len(intermediate_steps_list)), correlations["r2"], color='darkgreen', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(intermediate_steps_list)))
    ax.set_xticklabels([f'Step {s}' for s in intermediate_steps_list], rotation=45, ha='right')
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Regression R²: Intermediate vs Final Reward', fontsize=12, fontweight='bold')
    ax.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='0.8 threshold')
    ax.axhline(y=0.9, color='g', linestyle='--', linewidth=1, alpha=0.5, label='0.9 threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # MAE
    ax = axes[1, 0]
    ax.bar(range(len(intermediate_steps_list)), correlations["mae"], color='coral', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(intermediate_steps_list)))
    ax.set_xticklabels([f'Step {s}' for s in intermediate_steps_list], rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Prediction Error (MAE): Intermediate → Final Reward', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE
    ax = axes[1, 1]
    ax.bar(range(len(intermediate_steps_list)), correlations["rmse"], color='purple', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(intermediate_steps_list)))
    ax.set_xticklabels([f'Step {s}' for s in intermediate_steps_list], rotation=45, ha='right')
    ax.set_ylabel('Root Mean Squared Error', fontsize=11)
    ax.set_title('Prediction Error (RMSE): Intermediate → Final Reward', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_metrics_bar_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved correlation bar chart: {plot_dir / 'correlation_metrics_bar_chart.png'}")
    
    # Print summary statistics
    print(f"\n[INFO] ========== CORRELATION ANALYSIS SUMMARY ==========")
    print(f"Total scenes tested: {len(results)}")
    print(f"Intermediate steps tested: {intermediate_steps_list}")
    print(f"Final step: {final_step}")
    print(f"\n{'Step':<10} {'Pearson':<10} {'Spearman':<10} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 60)
    for i, step in enumerate(intermediate_steps_list):
        print(f"{step:<10} {correlations['pearson'][i]:<10.4f} {correlations['spearman'][i]:<10.4f} "
              f"{correlations['r2'][i]:<10.4f} {correlations['mae'][i]:<10.4f} {correlations['rmse'][i]:<10.4f}")
    
    print(f"\n[INFO] Regression Equations (y = final_reward, x = intermediate_reward):")
    for step, slope, intercept in regression_params:
        print(f"  Step {step:3d}: y = {slope:.4f} * x + {intercept:.4f}")
    
    print(f"\n[INFO] Final Reward Statistics:")
    print(f"  Mean: {np.mean(final_rewards):.4f}")
    print(f"  Std:  {np.std(final_rewards):.4f}")
    print(f"  Min:  {np.min(final_rewards):.4f}")
    print(f"  Max:  {np.max(final_rewards):.4f}")
    
    print(f"\n[COMPLETE] Results saved to: {batch_output_dir}")
    print(f"[INFO] You can load and analyze results with:")
    print(f"       import pickle")
    print(f"       results = pickle.load(open('{results_path}', 'rb'))")
    if cfg.get("save_scenes", False):
        print(f"\n[INFO] Render all intermediate steps for a scene with:")
        print(f"       python ../ThreedFront/scripts/render_results.py --retrieve_by_size --no_texture {batch_output_dir}/scene_pkls/scene0_all_steps.pkl")
        print(f"\n[INFO] Each pkl contains all {len(intermediate_steps)+1} timesteps for that scene in sequence")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
