"""
Script to compute reward statistics from baseline model samples.

This script samples scenes from a baseline model and computes statistics 
for multiple reward functions with enhanced analysis for curriculum design.
"""

import json
import os
import pickle
import subprocess

from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import numbers
import torch

from omegaconf import DictConfig
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans

from steerable_scene_generation.datasets.custom_scene import CustomDataset
from universal_constraint_rewards.commons import parse_and_descale_scenes
from universal_constraint_rewards.not_out_of_bound_reward import (
    SDFCache,
    precompute_sdf_cache,
)
from universal_constraint_rewards.accessibility_reward import AccessibilityCache

def analyze_distribution_shape(rewards_array):
    """
    Classify distribution shape for better curriculum insights.
    """
    # Compute min-based concentration
    min_val = float(np.min(rewards_array))
    # Exact min rate (within numerical tolerance)
    min_rate = np.mean(np.isclose(rewards_array, min_val, rtol=0.0, atol=1e-8))
    # Near-min rate: within +0.05 of the minimum (rewards are expected in [0,1])
    near_min_rate = np.mean(rewards_array <= (min_val + 0.05))

    # Compute max-based concentration
    max_val = float(np.max(rewards_array))
    # Exact max rate (within numerical tolerance)
    max_rate = np.mean(np.isclose(rewards_array, max_val, rtol=0.0, atol=1e-8))
    # Near-max rate: within -0.05 of the maximum
    near_max_rate = np.mean(rewards_array >= (max_val - 0.05))

    # (Replaced absolute threshold with relative threshold near the actual maximum)

    # Compute skewness and kurtosis
    skewness = float(scipy_stats.skew(rewards_array))
    kurtosis = float(scipy_stats.kurtosis(rewards_array))



    return {
        "min_rate": float(min_rate),
        "near_min_rate": float(near_min_rate),
        "max_rate": float(max_rate),
        "near_max_rate": float(near_max_rate),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

def compute_enhanced_stats(rewards_array, threshold, reward_name):
    """
    Compute comprehensive statistics for a single reward function.
    """
    # Basic statistics
    basic_stats = {
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "mean": float(np.mean(rewards_array)),
        "median": float(np.median(rewards_array)),
        "stddev": float(np.std(rewards_array)),
        "percentiles": {
            "p1": float(np.percentile(rewards_array, 1)),
            "p5": float(np.percentile(rewards_array, 5)),
            "p25": float(np.percentile(rewards_array, 25)),
            "p75": float(np.percentile(rewards_array, 75)),
            "p95": float(np.percentile(rewards_array, 95)),
            "p99": float(np.percentile(rewards_array, 99)),
        },
        "num_scenes": len(rewards_array),
    }

    # Success metrics
    num_success = int(np.sum(rewards_array >= threshold))
    success_rate = num_success / len(rewards_array)
    success_metrics = {
        "threshold": float(
            threshold
        ),
        "num_success": num_success,
        "success_rate": float(success_rate),
    }

    # Distribution analysis
    distribution = analyze_distribution_shape(rewards_array)

    return {
        "reward_name": reward_name,
        "basic_stats": basic_stats,
        "success_metrics": success_metrics,
        "distribution": distribution,
    }


def format_llm_summary(stats):
    """
    Create an LLM-friendly, comprehensive summary of reward analysis.
    """
    percentiles = stats['basic_stats'].get('percentiles', {})
    summary = f"""
=== REWARD ANALYSIS: {stats['reward_name']} ===

PERFORMANCE METRICS:
  • Success Rate: {stats['success_metrics']['success_rate']:.1%} \
({stats['success_metrics']['num_success']}/{stats['basic_stats']['num_scenes']} scenes)
  • Mean Reward: {stats['basic_stats']['mean']:.4f}
  • Median Reward: {stats['basic_stats']['median']:.4f}
  • Range: [{stats['basic_stats']['min']:.4f}, {stats['basic_stats']['max']:.4f}]
  • Std Dev: {stats['basic_stats']['stddev']:.4f}
"""
    # Add percentiles if available
    if percentiles:
        summary += "  • Percentiles:\n"
        for key in ['p1', 'p5', 'p25', 'p75', 'p95', 'p99']:
            if key in percentiles:
                summary += f"      - {key.upper()}: {percentiles[key]:.4f}\n"

    summary += f"""
DISTRIBUTION CHARACTERISTICS:

  • Skewness: {stats['distribution']['skewness']:.2f}
  • Kurtosis: {stats['distribution'].get('kurtosis', float('nan')):.2f}
    • Min Rate: {stats['distribution']['min_rate']:.1%}
    • Near Min Rate: {stats['distribution'].get('near_min_rate', 0):.1%}
    • Max Rate: {stats['distribution'].get('max_rate', 0):.1%}
    • Near Max Rate: {stats['distribution'].get('near_max_rate', 0):.1%}
"""
    summary += "\n" + "=" * 60 + "\n"
    return summary


def get_reward_stats_from_baseline(
    reward_functions: Dict[str, Callable],
    load: str = "rrudae6n",
    dataset: str = "custom_scene",
    config: DictConfig = None,
    dataset_processed_scene_data_path: str = "data/metadatas/custom_scene_metadata.json",
    dataset_max_num_objects_per_scene: int = 12,
    num_scenes: int = 1000,
    algorithm: str = "scene_diffuser_midiffusion",
    algorithm_trainer: str = "ddpm",
    experiment_find_unused_parameters: bool = True,
    algorithm_classifier_free_guidance_use: bool = False,
    algorithm_classifier_free_guidance_use_floor: bool = True,
    algorithm_classifier_free_guidance_weight: int = 0,
    algorithm_custom_loss: bool = True,
    algorithm_ema_use: bool = True,
    algorithm_noise_schedule_scheduler: str = "ddim",
    algorithm_noise_schedule_ddim_num_inference_timesteps: int = 150,
    algorithm_custom_old: bool = True,
    inpaint_masks=None,
    threshold_dict=None,
) -> Dict[str, Dict[str, float]]:
    """
    Sample scenes from baseline model and compute reward statistics.

    Args:
        reward_functions: Dict mapping reward function names to callable functions
                         Each function should take a scene dict and return a scalar reward [0, 1]
        load: Model checkpoint to load
        dataset: Dataset configuration name
        dataset_processed_scene_data_path: Path to scene metadata
        dataset_max_num_objects_per_scene: Maximum objects per scene
        num_scenes: Number of scenes to sample
        algorithm: Algorithm configuration name
        algorithm_trainer: Trainer type
        experiment_find_unused_parameters: Whether to find unused parameters
        algorithm_classifier_free_guidance_use: Use classifier-free guidance
        algorithm_classifier_free_guidance_weight: Guidance weight
        algorithm_custom_loss: Use custom loss
        algorithm_ema_use: Use EMA
        algorithm_noise_schedule_scheduler: Noise scheduler type
        algorithm_noise_schedule_ddim_num_inference_timesteps: Number of inference timesteps

    Returns:
        Dict mapping reward function names to statistics:
        {"reward_function_name": {"min": float, "max": float, "mean": float, "stddev": float}}
    """
    try:
        custom_dataset = CustomDataset(
            cfg=config.dataset,
            split=config.dataset.validation.get("splits", ["test"]),
            ckpt_path=None,
        )
    except Exception as e:
        print(f"Error creating CustomDataset: {e}")
        raise
    if inpaint_masks is not None:
        cmd = [
            "python",
            "scripts/inpaint.py",
            f"load={load}",
            f"dataset={dataset}",
            f"dataset.processed_scene_data_path={dataset_processed_scene_data_path}",
            f"dataset._name=custom_scene",
            f"+num_scenes={num_scenes}",
            f"algorithm={algorithm}",
            f"algorithm.trainer={algorithm_trainer}",
            f"experiment.find_unused_parameters={experiment_find_unused_parameters}",
            f"algorithm.classifier_free_guidance.use={algorithm_classifier_free_guidance_use}",
            f"algorithm.classifier_free_guidance.use_floor={algorithm_classifier_free_guidance_use_floor}",
            f"algorithm.classifier_free_guidance.weight={algorithm_classifier_free_guidance_weight}",
            f"algorithm.custom.loss={str(algorithm_custom_loss).lower()}",
            f"algorithm.ema.use={algorithm_ema_use}",
            f"algorithm.noise_schedule.scheduler={algorithm_noise_schedule_scheduler}",
            f"algorithm.noise_schedule.ddim.num_inference_timesteps={algorithm_noise_schedule_ddim_num_inference_timesteps}",
            f"algorithm.predict.inpaint_masks={inpaint_masks}",
            f"dataset.data.room_type={config.dataset.data.room_type}",
            f"dataset.model_path_vec_len=30",
            f"dataset.data.path_to_processed_data={config.dataset.data.path_to_processed_data}",
            f"dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm",
        ]


    else:
        # Build command to run custom_sample_and_render.py
        cmd = [
            "python",
            "scripts/custom_sample_and_render.py",
            f"load={load}",
            f"dataset={dataset}",
            f"dataset.processed_scene_data_path={dataset_processed_scene_data_path}",
            f"dataset.max_num_objects_per_scene={dataset_max_num_objects_per_scene}",
            f"dataset._name=custom_scene",
            f"+num_scenes={num_scenes}",
            f"algorithm={algorithm}",
            f"algorithm.trainer={algorithm_trainer}",
            f"experiment.find_unused_parameters={experiment_find_unused_parameters}",
            f"algorithm.classifier_free_guidance.use={algorithm_classifier_free_guidance_use}",
            f"algorithm.classifier_free_guidance.use_floor={algorithm_classifier_free_guidance_use_floor}",
            f"algorithm.classifier_free_guidance.weight={algorithm_classifier_free_guidance_weight}",
            f"algorithm.custom.loss={str(algorithm_custom_loss).lower()}",
            f"algorithm.ema.use={algorithm_ema_use}",
            f"algorithm.noise_schedule.scheduler={algorithm_noise_schedule_scheduler}",
            f"algorithm.noise_schedule.ddim.num_inference_timesteps={algorithm_noise_schedule_ddim_num_inference_timesteps}",
            f"dataset.model_path_vec_len=30",
            f"dataset.data.path_to_processed_data={config.dataset.data.path_to_processed_data}",
            f"dataset.data.room_type={config.dataset.data.room_type}",
            f"dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm",
        ]
    
    if config.dataset.data.room_type == "livingroom":
        cmd.append("dataset.data.dataset_directory=livingroom")
        cmd.append("dataset.data.annotation_file=livingroom_threed_front_splits.csv")
        cmd.append(f"dataset.max_num_objects_per_scene=21")
        cmd.append(f"algorithm.custom.objfeat_dim=0")
        cmd.append(f"algorithm.custom.obj_vec_len=65")
        cmd.append(f"algorithm.custom.obj_diff_vec_len=65")
        cmd.append(f"algorithm.custom.num_classes=25")
    elif config.dataset.data.room_type == "bedroom":
        cmd.append("dataset.data.dataset_directory=bedroom")
        cmd.append("dataset.data.annotation_file=bedroom_threed_front_splits_original.csv")
        cmd.append(f"dataset.max_num_objects_per_scene=12")
        cmd.append(f"algorithm.custom.num_classes=22")
        cmd.append(f"algorithm.custom.objfeat_dim=0")
        cmd.append(f"algorithm.custom.old={algorithm_custom_old}")
    # NOTE OBJFEATS IS SET TO ZERO FOR BEDROOM AND LIVINGROOM TOO
    
    print(f"Command: {' '.join(cmd)}")

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    print(f"Running sampling command...")
    print(f"Command: {' '.join(cmd)}")

    # Run the sampling script
    #NOTE: uncomment to run the sampling script
    # Run the sampling script and print logs live to the console
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=Path(__file__).parent.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Print output line by line as it's produced
    try:
        for line in process.stdout:
            print(line, end="")
    except Exception as e:
        pass
    
    process.wait()

    print(f"Sampling script completed with return code {process.returncode}")
    if process.returncode != 0:
        raise RuntimeError(
            f"Sampling script failed with return code {process.returncode}"
        )
    print(f"Sampling completed successfully")

    # Find the output pickle file
    # The script saves to outputs/<date>/<timestamp>/sampled_scenes_results.pkl
    outputs_dir = Path(__file__).parent.parent / "outputs"

    # Recursively find all pkl files and get the most recent one
    pkl_files = list(outputs_dir.glob("**/raw_sampled_scenes.pkl"))

    if not pkl_files:
        raise FileNotFoundError(
            f"Could not find sampled_scenes_results.pkl in outputs directory: {outputs_dir}"
        )

    # Sort by modification time and get the most recent
    pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading sampled scenes from: {pkl_path}")

    # Load the pickle file containing ThreedFrontResults
    with open(pkl_path, "rb") as f:
        raw_results = pickle.load(f)

    dataset_size = len(custom_dataset)
    indices = [i % dataset_size for i in range(num_scenes)]
    # print(f"[Ashok] Raw results: {raw_results}")
    raw_results = torch.tensor(raw_results)
    room_type = config.dataset.data.room_type
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    from universal_constraint_rewards.commons import idx_to_labels
    idx_to_labels = idx_to_labels[room_type]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes_with_empty"]
    sdf_cache = SDFCache(config.dataset.sdf_cache_dir, split="test")
    accessibility_cache = AccessibilityCache(config.dataset.accessibility_cache_dir, split="test")

    floor_plan_args_list = [custom_dataset.get_floor_plan_args(idx) for idx in indices]
    # Stack each key across the batch for tensor conversion
    floor_plan_args = {
        key: [args[key] for args in floor_plan_args_list]
        for key in [
            "floor_plan_centroid",
            "floor_plan_vertices",
            "floor_plan_faces",
            "room_outer_box",
        ]
    }
    parsed_scenes = parse_and_descale_scenes(raw_results,num_classes=num_classes, room_type=room_type)

    
    reward_stats = {}
    reward_args = {
        "parsed_scenes": parsed_scenes,
        "idx_to_labels": idx_to_labels,
        "room_type": room_type,
        "max_objects": max_objects,
        "num_classes": num_classes,
        "floor_polygons": [
            torch.tensor(
                custom_dataset.get_floor_polygon_points(idx),
                device=parsed_scenes["device"],
            )
            for idx in indices
        ],
        "indices": indices,
        "is_val": True,
        "sdf_cache": sdf_cache,
        "accessibility_cache": accessibility_cache,
        "floor_plan_args": floor_plan_args,
    }
    # # Save arguments to npy file (use pickle for objects that are not arrays)
    # np.save(f"reward_func_args_for_first_10_scenes.npy", reward_args, allow_pickle=True)

    # Prepare output directories
    base_dir = os.path.dirname(__file__)
    user_query = config.algorithm.ddpo.dynamic_constraint_rewards.user_query
    txt_dir = os.path.join(base_dir, f"{user_query.replace(' ', '_').replace('.', '')}_reward_analysis_txt")
    os.makedirs(txt_dir, exist_ok=True)

    for reward_name, reward_func in reward_functions.items():
        print(f"Computing rewards for: {reward_name}")
        # Support for threshold: if reward_func is tuple/list, unpack
        reward_name_base = "_".join(reward_name.split("_")[1:])
        threshold = threshold_dict.get(reward_name_base)
        if threshold is None:
            print(f"Threshold for {reward_name_base} is not found in threshold_dict baseline")
            threshold = 0.0
        print(f"Threshold for {reward_name_base} baseline: {threshold}")

        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels,
            room_type=room_type,
            max_objects=max_objects,
            num_classes=num_classes,
            floor_polygons=[
                torch.tensor(
                    custom_dataset.get_floor_polygon_points(idx),
                    device=parsed_scenes["device"],
                )
                for idx in indices
            ],
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            accessibility_cache=accessibility_cache,
            floor_plan_args=floor_plan_args,
        )
        
        print(f"Reward {reward_name}: {rewards}")
        print(f"Threshold for {reward_name}: {threshold}")

        rewards_array = np.array(rewards)
        if reward_name == "room_layout_constraint":
            np.save("room_layout_rewards.npy", rewards_array)
            print("Saved room layout rewards to room_layout_rewards.npy")

        # Enhanced analysis
        stats = compute_enhanced_stats(rewards_array, threshold, reward_name)

        # Generate LLM-friendly summary
        llm_summary = format_llm_summary(stats)
        llm_summary_path = os.path.join(txt_dir, f"{reward_name}_llm_summary_baseline.txt")
        with open(llm_summary_path, "w") as f:
            f.write(llm_summary)
        stats["llm_summary_path"] = llm_summary_path

        # Print to console
        print(llm_summary)

        # Store in reward_stats dict
        reward_stats[reward_name] = stats

    return reward_stats


def get_reward_stats_from_dataset(
    reward_functions: Dict[str, Callable],
    config: DictConfig = None,
    num_scenes: int | None = None,
    threshold_dict=None,
) -> Dict[str, Dict[str, float]]:
    """
    Load scenes from ground truth dataset and compute reward statistics.

    Args:
        reward_functions: Dict mapping reward function names to callable functions
                         Each function should take a scene dict and return a scalar reward [0, 1]
        config: Configuration object containing dataset paths
        num_scenes: Number of scenes to load from dataset (None = all scenes)

    Returns:
        Dict mapping reward function names to statistics:
        {"reward_function_name": {"min": float, "max": float, "mean": float, "stddev": float}}
    """

    if config is None:
        raise ValueError("Config must be provided")

    print(f"Loading ground truth dataset using CustomDataset...")

    try:
        custom_dataset = CustomDataset(
            cfg=config.dataset,
            # split=config.dataset.validation.get("splits", ["test"]),
            split=["train", "val", "test"],
            ckpt_path=None,
        )
    except Exception as e:
        print(f"Error creating CustomDataset: {e}")
        raise

    total_scenes = len(custom_dataset)
    if num_scenes is None or num_scenes > total_scenes:
        num_scenes = total_scenes
    indices = list(range(num_scenes))
    print(f"Loading {num_scenes} scenes from dataset (total available: {total_scenes})")

    # Load scenes from dataset
    scenes_list = []
    for i in range(num_scenes):
        try:
            sample = custom_dataset[i]
            scene = sample["scenes"]
            scenes_list.append(scene)
        except Exception as e:
            print(f"Warning: Could not load scene {i}: {e}")
            continue

    if len(scenes_list) == 0:
        raise ValueError("No scenes could be loaded from the dataset")

    # Stack into batch tensor
    raw_results = torch.stack(scenes_list)
    print(f"Loaded scenes tensor shape: {raw_results.shape}")
    
        # Get room type and metadata from config
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    from universal_constraint_rewards.commons import idx_to_labels
    room_type = config.dataset.data.room_type
    idx_to_labels = idx_to_labels[room_type]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes_with_empty"]
    sdf_cache = SDFCache(config.dataset.sdf_cache_dir, split="test")
    accessibility_cache = AccessibilityCache(config.dataset.accessibility_cache_dir, split="test")
    reward_stats = {}
    floor_plan_args_list = [custom_dataset.get_floor_plan_args(idx) for idx in indices]
    # Stack each key across the batch for tensor conversion
    floor_plan_args = {
        key: [args[key] for args in floor_plan_args_list]
        for key in [
            "floor_plan_centroid",
            "floor_plan_vertices",
            "floor_plan_faces",
            "room_outer_box",
        ]
    }

    # Parse scenes
    
    parsed_scenes = parse_and_descale_scenes(raw_results, num_classes=num_classes, room_type=room_type)


    # Prepare output directories (mirror baseline behaviour)
    base_dir = os.path.dirname(__file__)
    user_query = config.algorithm.ddpo.dynamic_constraint_rewards.user_query
    txt_dir = os.path.join(base_dir, f"{user_query.replace(' ', '_').replace('.', '')}_reward_analysis_txt")
    os.makedirs(txt_dir, exist_ok=True)

        
    # Compute rewards for each function, and return enhanced statistics similar to baseline
    for reward_name, reward_func in reward_functions.items():
        print(f"Computing rewards for: {reward_name}")
        reward_name_base = "_".join(reward_name.split("_")[1:])
        threshold = threshold_dict.get(reward_name_base)
        if threshold is None:
            print(f"Threshold for {reward_name_base} is not found in threshold_dict dataset")
            threshold = 0.0
        print(f"Threshold for {reward_name_base} dataset: {threshold}")
        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels,
            room_type=room_type,
            max_objects=max_objects,
            num_classes=num_classes,
            floor_polygons=[
                torch.tensor(
                    custom_dataset.get_floor_polygon_points(idx),
                    device=parsed_scenes["device"],
                )
                for idx in indices
            ],
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            accessibility_cache=accessibility_cache,
            floor_plan_args=floor_plan_args,
        )

        # Convert to numpy array
        rewards_array = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else np.array(rewards)
        
        print(f"Reward {reward_name}: {rewards}")


        # Enhanced analysis identical to baseline helper
        stats = compute_enhanced_stats(rewards_array, threshold, reward_name)

        # Generate and save LLM-friendly summary to file (same as baseline)
        llm_summary = format_llm_summary(stats)
        llm_summary_path = os.path.join(txt_dir, f"{reward_name}_llm_summary_dataset.txt")
        with open(llm_summary_path, "w") as f:
            f.write(llm_summary)
        stats["llm_summary_path"] = llm_summary_path

        reward_stats[reward_name] = stats

    return reward_stats



def get_reward_stats_from_baseline_for_normalizer(
    reward_functions: Dict[str, Callable],
    load: str = "rrudae6n",
    dataset: str = "custom_scene",
    config: DictConfig = None,
    reward_functions_initial: Dict[str, Callable] = None,
    dataset_processed_scene_data_path: str = "data/metadatas/custom_scene_metadata.json",
    dataset_max_num_objects_per_scene: int = 12,
    num_scenes: int = 1000,
    algorithm: str = "scene_diffuser_midiffusion",
    algorithm_trainer: str = "ddpm",
    experiment_find_unused_parameters: bool = True,
    algorithm_classifier_free_guidance_use: bool = False,
    algorithm_classifier_free_guidance_use_floor: bool = True,
    algorithm_classifier_free_guidance_weight: int = 0,
    algorithm_custom_loss: bool = True,
    algorithm_ema_use: bool = True,
    algorithm_noise_schedule_scheduler: str = "ddim",
    algorithm_noise_schedule_ddim_num_inference_timesteps: int = 150,
    algorithm_custom_old: bool = True,
    inpaint_dict=None,
    threshold_dict=None,
) -> Dict[str, Dict[str, float]]:
    """
    Sample scenes from baseline model and compute reward statistics.

    Args:
        reward_functions: Dict mapping reward function names to callable functions
                         Each function should take a scene dict and return a scalar reward [0, 1]
        load: Model checkpoint to load
        dataset: Dataset configuration name
        dataset_processed_scene_data_path: Path to scene metadata
        dataset_max_num_objects_per_scene: Maximum objects per scene
        num_scenes: Number of scenes to sample
        algorithm: Algorithm configuration name
        algorithm_trainer: Trainer type
        experiment_find_unused_parameters: Whether to find unused parameters
        algorithm_classifier_free_guidance_use: Use classifier-free guidance
        algorithm_classifier_free_guidance_weight: Guidance weight
        algorithm_custom_loss: Use custom loss
        algorithm_ema_use: Use EMA
        algorithm_noise_schedule_scheduler: Noise scheduler type
        algorithm_noise_schedule_ddim_num_inference_timesteps: Number of inference timesteps

    Returns:
        Dict mapping reward function names to statistics:
        {"reward_function_name": {"min": float, "max": float, "mean": float, "stddev": float}}
    """
    try:
        custom_dataset = CustomDataset(
            cfg=config.dataset,
            split=config.dataset.validation.get("splits", ["test"]),
            ckpt_path=None,
        )
    except Exception as e:
        print(f"Error creating CustomDataset: {e}")
        raise
    if inpaint_dict is not None:
        cmd = [
            "python",
            "scripts/inpaint.py",
            f"load={load}",
            f"dataset={dataset}",
            f"dataset.processed_scene_data_path={dataset_processed_scene_data_path}",
            f"dataset._name=custom_scene",
            f"+num_scenes={num_scenes}",
            f"algorithm={algorithm}",
            f"algorithm.trainer={algorithm_trainer}",
            f"experiment.find_unused_parameters={experiment_find_unused_parameters}",
            f"algorithm.classifier_free_guidance.use={algorithm_classifier_free_guidance_use}",
            f"algorithm.classifier_free_guidance.use_floor={algorithm_classifier_free_guidance_use_floor}",
            f"algorithm.classifier_free_guidance.weight={algorithm_classifier_free_guidance_weight}",
            f"algorithm.custom.loss={str(algorithm_custom_loss).lower()}",
            f"algorithm.ema.use={algorithm_ema_use}",
            f"algorithm.noise_schedule.scheduler={algorithm_noise_schedule_scheduler}",
            f"algorithm.noise_schedule.ddim.num_inference_timesteps={algorithm_noise_schedule_ddim_num_inference_timesteps}",
            f"algorithm.predict.inpaint_masks={inpaint_dict}",
            f"dataset.data.room_type={config.dataset.data.room_type}",
            f"dataset.model_path_vec_len=30",
            f"dataset.data.path_to_processed_data={config.dataset.data.path_to_processed_data}",
            f"dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm",
        ]


    else:
        # Build command to run custom_sample_and_render.py
        cmd = [
            "python",
            "scripts/custom_sample_and_render.py",
            f"load={load}",
            f"dataset={dataset}",
            f"dataset.processed_scene_data_path={dataset_processed_scene_data_path}",
            f"dataset.max_num_objects_per_scene={dataset_max_num_objects_per_scene}",
            f"dataset._name=custom_scene",
            f"+num_scenes={num_scenes}",
            f"algorithm={algorithm}",
            f"algorithm.trainer={algorithm_trainer}",
            f"experiment.find_unused_parameters={experiment_find_unused_parameters}",
            f"algorithm.classifier_free_guidance.use={algorithm_classifier_free_guidance_use}",
            f"algorithm.classifier_free_guidance.use_floor={algorithm_classifier_free_guidance_use_floor}",
            f"algorithm.classifier_free_guidance.weight={algorithm_classifier_free_guidance_weight}",
            f"algorithm.custom.loss={str(algorithm_custom_loss).lower()}",
            f"algorithm.ema.use={algorithm_ema_use}",
            f"algorithm.noise_schedule.scheduler={algorithm_noise_schedule_scheduler}",
            f"algorithm.noise_schedule.ddim.num_inference_timesteps={algorithm_noise_schedule_ddim_num_inference_timesteps}",
            f"dataset.model_path_vec_len=30",
            f"dataset.data.path_to_processed_data={config.dataset.data.path_to_processed_data}",
            f"dataset.data.room_type={config.dataset.data.room_type}",
            f"dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm",
        ]
    
    if config.dataset.data.room_type == "livingroom":
        cmd.append("dataset.data.dataset_directory=livingroom")
        cmd.append("dataset.data.annotation_file=livingroom_threed_front_splits.csv")
        cmd.append(f"dataset.max_num_objects_per_scene=21")
        cmd.append(f"algorithm.custom.objfeat_dim=0")
        cmd.append(f"algorithm.custom.obj_vec_len=65")
        cmd.append(f"algorithm.custom.obj_diff_vec_len=65")
        cmd.append(f"algorithm.custom.num_classes=25")
    elif config.dataset.data.room_type == "bedroom":
        cmd.append("dataset.data.dataset_directory=bedroom")
        cmd.append("dataset.data.annotation_file=bedroom_threed_front_splits_original.csv")
        cmd.append(f"dataset.max_num_objects_per_scene=12")
        cmd.append(f"algorithm.custom.num_classes=22")
        cmd.append(f"algorithm.custom.objfeat_dim=0")
        cmd.append(f"algorithm.custom.old={algorithm_custom_old}")
    # NOTE OBJFEATS IS SET TO ZERO FOR BEDROOM AND LIVINGROOM TOO
    
    print(f"Command: {' '.join(cmd)}")

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    print(f"Running sampling command...")
    print(f"Command: {' '.join(cmd)}")

    # Run the sampling script
    #NOTE: uncomment to run the sampling script
    # Run the sampling script and print logs live to the console
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=Path(__file__).parent.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Print output line by line as it's produced
    try:
        for line in process.stdout:
            print(line, end="")
    except Exception as e:
        pass
    
    process.wait()

    print(f"Sampling script completed with return code {process.returncode}")
    if process.returncode != 0:
        raise RuntimeError(
            f"Sampling script failed with return code {process.returncode}"
        )
    print(f"Sampling completed successfully")

    # Find the output pickle file
    # The script saves to outputs/<date>/<timestamp>/sampled_scenes_results.pkl
    outputs_dir = Path(__file__).parent.parent / "outputs"

    # Recursively find all pkl files and get the most recent one
    pkl_files = list(outputs_dir.glob("**/raw_sampled_scenes.pkl"))

    if not pkl_files:
        raise FileNotFoundError(
            f"Could not find sampled_scenes_results.pkl in outputs directory: {outputs_dir}"
        )

    # Sort by modification time and get the most recent
    pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading sampled scenes from: {pkl_path}")

    # Load the pickle file containing ThreedFrontResults
    with open(pkl_path, "rb") as f:
        raw_results = pickle.load(f)


    dataset_size = len(custom_dataset)
    indices = [i % dataset_size for i in range(num_scenes)]
    
    from universal_constraint_rewards.commons import idx_to_labels
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    room_type = config.dataset.data.room_type
    idx_to_labels = idx_to_labels[room_type]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes_with_empty"]
    
    raw_results = torch.tensor(raw_results)
    raw_results = torch.nan_to_num(raw_results)
    
    parsed_scenes = parse_and_descale_scenes(raw_results, num_classes=num_classes, room_type=room_type)

    

    sdf_cache = SDFCache(config.dataset.sdf_cache_dir, split="test")
    accessibility_cache = AccessibilityCache(config.dataset.accessibility_cache_dir, split="test")

    floor_plan_args_list = [custom_dataset.get_floor_plan_args(idx) for idx in indices]
    # Stack each key across the batch for tensor conversion
    floor_plan_args = {
        key: [args[key] for args in floor_plan_args_list]
        for key in [
            "floor_plan_centroid",
            "floor_plan_vertices",
            "floor_plan_faces",
            "room_outer_box",
        ]
    }
    reward_stats = {}
    reward_stats_initial = {}
    user_query = config.algorithm.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    stats_json = os.path.join(config.algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir, f"{user_query}_stats.json")
    print(f"Stats JSON: {stats_json}")
    os.makedirs(os.path.dirname(stats_json), exist_ok=True)

    for reward_name, reward_func in reward_functions.items():
        print(f"Computing rewards for: {reward_name}")
        # Support for threshold: if reward_func is tuple/list, unpack

        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels,
            room_type=room_type,
            max_objects=max_objects,
            num_classes=num_classes,
            floor_polygons=[
                torch.tensor(
                    custom_dataset.get_floor_polygon_points(idx),
                    device=parsed_scenes["device"],
                )
                for idx in indices
            ],
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            accessibility_cache=accessibility_cache,
            floor_plan_args=floor_plan_args,
        )
        
        print(f"Reward {reward_name}: {rewards}")

        rewards_array = np.array(rewards)
    
        reward_stats[reward_name] = {
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "mean": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "stddev": float(np.std(rewards_array)),
            "percentile_1": float(np.percentile(rewards_array, 1)),
            "percentile_5": float(np.percentile(rewards_array, 5)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
            "percentile_95": float(np.percentile(rewards_array, 95)),
            "percentile_99": float(np.percentile(rewards_array, 99)),
            "num_scenes": len(rewards_array),
        }

    with open(stats_json, "w") as f:
        json.dump(reward_stats, f, indent=2)
        
    print(f"Saved reward stats to: {stats_json}")
    
    stats_json_initial = os.path.join(config.algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir, f"{user_query}_stats_initial.json")
    for reward_name, reward_func in reward_functions_initial.items():
        print(f"Computing rewards for: {reward_name}")
        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels,
            room_type=room_type,
            max_objects=max_objects,
            num_classes=num_classes,
            floor_polygons=[
                torch.tensor(
                    custom_dataset.get_floor_polygon_points(idx),
                    device=parsed_scenes["device"],
                )
                for idx in indices
            ],
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            accessibility_cache=accessibility_cache,
            floor_plan_args=floor_plan_args,
        )
        rewards_array = np.array(rewards)
        reward_stats_initial[reward_name] = {
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "mean": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "stddev": float(np.std(rewards_array)),
            "percentile_1": float(np.percentile(rewards_array, 1)),
            "percentile_5": float(np.percentile(rewards_array, 5)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
            "percentile_95": float(np.percentile(rewards_array, 95)),
            "percentile_99": float(np.percentile(rewards_array, 99)),
            "num_scenes": len(rewards_array),
        }
    with open(stats_json_initial, "w") as f:
        json.dump(reward_stats_initial, f, indent=2)
    print(f"Saved initial reward stats to: {stats_json_initial}")

    return reward_stats, reward_stats_initial


from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers
import hydra

@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    from universal_constraint_rewards.commons import get_all_universal_reward_functions
    from dynamic_constraint_rewards.commons import import_dynamic_reward_functions
    user_query = cfg.algorithm.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    reward_functions = get_all_universal_reward_functions()
    reward_functions_initial = get_all_universal_reward_functions()
    if cfg.algorithm.ddpo.dynamic_constraint_rewards.use:
        dynamic_rewards, _ = import_dynamic_reward_functions(reward_code_dir=f"{user_query}_dynamic_reward_functions_final")
        dynamic_rewards_initial, _ = import_dynamic_reward_functions(reward_code_dir=f"{user_query}_dynamic_reward_functions_initial")
        reward_functions.update(dynamic_rewards)
        reward_functions_initial.update(dynamic_rewards_initial)
    print(f"Reward functions to analyze: {list(reward_functions.keys())}")
    if cfg.algorithm.ddpo.use_inpaint:
        inpaint_path = cfg.algorithm.ddpo.dynamic_constraint_rewards.inpaint_path 
        with open(inpaint_path, "r") as f:
            inpaint_info = json.load(f)
        inpaint_dict = inpaint_info["inpaint"]
        inpaint_dict = str(inpaint_dict).replace("'", '')
        print(f"Loaded inpaint masks from: {inpaint_path}, inpaint dict {inpaint_dict}")
    else:
        inpaint_dict = None
    
    reward_stats, reward_stats_initial = get_reward_stats_from_baseline_for_normalizer(
        reward_functions=reward_functions,
        reward_functions_initial=reward_functions_initial,
        config=cfg,
        load=cfg.load,
        dataset_max_num_objects_per_scene=cfg.dataset.max_num_objects_per_scene,
        algorithm_custom_old=True,
        inpaint_dict=None,
        # num_scenes=300,
    )
    
if __name__ == "__main__":
    main()
    
    
# python dynamic_constraint_rewards/get_reward_stats.py load=python scripts/custom_sample_and_render.py \
    # load=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/model.ckpt \
    # dataset=custom_scene \
    # dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    # dataset.max_num_objects_per_scene=12 \
    # +num_scenes=1000 \
    # algorithm=scene_diffuser_midiffusion \
    # algorithm.trainer=ddpm \
    # experiment.find_unused_parameters=True \
    # algorithm.classifier_free_guidance.use=False \
    # algorithm.classifier_free_guidance.use_floor=True \
    # algorithm.classifier_free_guidance.weight=1 \
    # algorithm.custom.loss=true \
    # algorithm.ema.use=True \
    # algorithm.noise_schedule.scheduler=ddim \
    # algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    # algorithm.custom.objfeat_dim=0 \
    # algorithm.custom.old=True \
    # algorithm.custom.loss=True dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 dataset.sdf_cache_dir=./living_sdf_cache/ dataset.accessibility_cache_dir=./living_accessibility_cache/