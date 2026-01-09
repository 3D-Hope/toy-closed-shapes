"""
Script to compute success rates for reward functions with user-provided thresholds.

This script:
1. Loads reward functions from config
2. Prompts user for threshold for each reward
3. Samples scenes from baseline model
4. Computes rewards and success rates based on thresholds
"""

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Callable, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from steerable_scene_generation.datasets.custom_scene import CustomDataset
from steerable_scene_generation.utils.omegaconf import register_resolvers
from universal_constraint_rewards.accessibility_reward import AccessibilityCache
from universal_constraint_rewards.commons import (
    get_all_universal_reward_functions,
    parse_and_descale_scenes,
)
from universal_constraint_rewards.not_out_of_bound_reward import (
    SDFCache,
    precompute_sdf_cache,
)


def get_thresholds_from_user(reward_functions: Dict[str, Callable]) -> Dict[str, float]:
    """
    Prompt user to input threshold for each reward function.
    
    Args:
        reward_functions: Dict mapping reward function names to callable functions
        
    Returns:
        Dict mapping reward function base names to threshold values
    """
    print("\n" + "="*80)
    print("THRESHOLD INPUT")
    print("="*80)
    print("\nPlease provide a threshold for each reward function.")
    print("Success is counted when reward >= threshold.\n")
    
    threshold_dict = {}
    
    for reward_name in reward_functions.keys():

        
        if reward_name in threshold_dict:
            # Already got threshold for this base name
            continue
            
        while True:
            try:
                # threshold_input = input(f"Threshold for '{reward_name}' (0.0-1.0): ")
                threshold_input = 10.0  # NOTE: For testing purposes, set a default threshold
                threshold = float(threshold_input)
                
                threshold_dict[reward_name] = threshold
                break

            except ValueError:
                print("  Error: Invalid input. Please enter a number between 0.0 and 1.0.")
    
    print("\n" + "="*80)
    print("THRESHOLDS SUMMARY")
    print("="*80)
    for name, threshold in threshold_dict.items():
        print(f"  {name}: {threshold}")
    print("="*80 + "\n")
    
    return threshold_dict


def sample_scenes_from_baseline(
    config: DictConfig,
    num_scenes: int = 1000,
) -> tuple:
    """
    Sample scenes from baseline model.
    
    Returns:
        Tuple of (parsed_scenes, custom_dataset, indices, pkl_path)
    """
    custom_dataset = CustomDataset(
        cfg=config.dataset,
        split=config.dataset.validation.get("splits", ["test"]),
        ckpt_path=None,
    )
    
    # Build command to run sampling script
    cmd = [
        "python",
        "scripts/custom_sample_and_render.py",
        f"load={config.load}",
        f"checkpoint_version={config.checkpoint_version}",
        f"dataset=custom_scene",
        f"dataset.processed_scene_data_path={config.dataset.processed_scene_data_path}",
        f"dataset.max_num_objects_per_scene={config.dataset.max_num_objects_per_scene}",
        f"dataset._name=custom_scene",
        f"+num_scenes={num_scenes}",
        # f"algorithm=scene_diffuser_flux_transformer", # TODO: make this configurable
        # f"algorithm=scene_diffuser_flux_transformer",
        f"algorithm=scene_diffuser_midiffusion",
        f"algorithm.trainer={config.algorithm.trainer}",
        f"experiment.find_unused_parameters=True",
        f"algorithm.classifier_free_guidance.use=False",
        f"algorithm.classifier_free_guidance.use_floor={config.algorithm.classifier_free_guidance.use_floor}",
        f"algorithm.classifier_free_guidance.weight=0",
        f"algorithm.custom.loss=true",
        f"algorithm.ema.use={config.algorithm.ema.use}",
        f"algorithm.noise_schedule.scheduler=ddim",
        f"algorithm.noise_schedule.ddim.num_inference_timesteps={config.algorithm.noise_schedule.ddim.num_inference_timesteps}",
        f"dataset.model_path_vec_len=30",
        f"dataset.data.path_to_processed_data={config.dataset.data.path_to_processed_data}",
        f"dataset.data.room_type={config.dataset.data.room_type}",
        f"dataset.data.encoding_type={config.dataset.data.encoding_type}",
        f"algorithm.custom.old={config.algorithm.custom.get('old', False)}",
    ]
    
    if config.dataset.data.room_type == "livingroom":
        cmd.extend([
            "dataset.data.dataset_directory=livingroom",
            "dataset.data.annotation_file=livingroom_threed_front_splits.csv",
            f"dataset.max_num_objects_per_scene=21",
            f"algorithm.custom.objfeat_dim=0",
            f"algorithm.custom.obj_vec_len=65",
            f"algorithm.custom.obj_diff_vec_len=65",
            f"algorithm.custom.num_classes=25",
        ])
    elif config.dataset.data.room_type == "bedroom":
        cmd.extend([
            "dataset.data.dataset_directory=bedroom",
            "dataset.data.annotation_file=bedroom_threed_front_splits_original.csv",
            f"dataset.max_num_objects_per_scene=12",
            f"algorithm.custom.num_classes=22",
            f"algorithm.custom.objfeat_dim=0",
        ])
    
    print("\n" + "="*80)
    print("SAMPLING SCENES FROM BASELINE MODEL")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the sampling script
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
    
    # Print output line by line
    try:
        for line in process.stdout:
            print(line, end="")
    except Exception:
        pass
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(
            f"Sampling script failed with return code {process.returncode}"
        )
    
    print("\nSampling completed successfully!")
    
    # Find the output pickle file
    outputs_dir = Path(__file__).parent.parent / "outputs"
    pkl_files = list(outputs_dir.glob("**/raw_sampled_scenes.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(
            f"Could not find raw_sampled_scenes.pkl in outputs directory: {outputs_dir}"
        )
    
    # Get the most recent file
    pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading sampled scenes from: {pkl_path}")
    
    # Load the pickle file
    with open(pkl_path, "rb") as f:
        raw_results = pickle.load(f)
    
    dataset_size = len(custom_dataset)
    indices = [i % dataset_size for i in range(num_scenes)]
    
    #
    
    from universal_constraint_rewards.commons import idx_to_labels
    room_type = config.dataset.data.room_type
    idx_to_labels_dict = idx_to_labels[room_type]
    num_classes = 22 if config.dataset.data.room_type == "bedroom" else 25
    
    # Parse scenes
    raw_results = torch.tensor(raw_results)
    raw_results = torch.nan_to_num(raw_results)
    parsed_scenes = parse_and_descale_scenes(
        raw_results, num_classes=num_classes, room_type=room_type
    )
    
    return parsed_scenes, custom_dataset, indices, pkl_path, idx_to_labels_dict


def compute_success_rates(
    reward_functions: Dict[str, Callable],
    threshold_dict: Dict[str, float],
    parsed_scenes: dict,
    custom_dataset: CustomDataset,
    indices: list,
    config: DictConfig,
    idx_to_labels_dict: dict,
    num_classes: int,
    max_objects: int = 12,
) -> Dict[str, Dict]:
    """
    Compute rewards and success rates for all reward functions.
    
    Returns:
        Dict mapping reward function names to statistics including success rate
    """
    # Load room info
    # all_rooms_info = json.load(
    #     open(
    #         os.path.join(
    #             config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
    #         )
    #     )
    # )
    
    room_type = config.dataset.data.room_type
    max_objects = 12 if room_type == "bedroom" else 22
    # Initialize caches
    # sdf_cache = SDFCache(config.dataset.sdf_cache_dir, split="test")
    # accessibility_cache = AccessibilityCache(
    #     config.dataset.accessibility_cache_dir, split="test"
    # )
    
    # # Prepare floor plan args
    # floor_plan_args_list = [custom_dataset.get_floor_plan_args(idx) for idx in indices]
    # floor_plan_args = {
    #     key: [args[key] for args in floor_plan_args_list]
    #     for key in [
    #         "floor_plan_centroid",
    #         "floor_plan_vertices",
    #         "floor_plan_faces",
    #         "room_outer_box",
    #     ]
    # }
    
    results = {}
    
    print("\n" + "="*80)
    print("COMPUTING REWARDS AND SUCCESS RATES")
    print("="*80 + "\n")
    
    for reward_name, reward_func in reward_functions.items():
        print(f"Processing: {reward_name}")
        
        # Get threshold for this reward
        # reward_name = "_".join(reward_name.split("_")[1:])
        try:
            threshold = threshold_dict[reward_name]
        except:
            raise Exception(f"Threshold not found for reward: {reward_name}")
        
        
        # Compute rewards
        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels_dict,
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
            # is_val=True,
            # sdf_cache=sdf_cache,
            # accessibility_cache=accessibility_cache,
            # floor_plan_args=floor_plan_args,
        )
        
        # Convert to numpy array
        rewards_array = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else np.array(rewards)
        print(f"  Rewards computed for {len(rewards_array)} scenes.")
        # Compute success metrics
        num_success = int(np.sum(rewards_array >= threshold))
        success_rate = num_success / len(rewards_array)
        
        # Compute basic statistics
        stats = {
            "reward_name": reward_name,
            "threshold": float(threshold),
            "num_scenes": len(rewards_array),
            "num_success": num_success,
            "success_rate": float(success_rate),
            "mean": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "stddev": float(np.std(rewards_array)),
            "percentiles": {
                "p1": float(np.percentile(rewards_array, 1)),
                "p5": float(np.percentile(rewards_array, 5)),
                "p25": float(np.percentile(rewards_array, 25)),
                "p75": float(np.percentile(rewards_array, 75)),
                "p95": float(np.percentile(rewards_array, 95)),
                "p99": float(np.percentile(rewards_array, 99)),
            },
        }
        
        results[reward_name] = stats
        
        # Print summary
        print(f"  Success Rate: {success_rate:.2%} ({num_success}/{len(rewards_array)} scenes)")
        print(f"  Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
    
    return results


def save_results(results: Dict, config: DictConfig, threshold_dict: Dict[str, float]):
    """
    Save results to JSON file and print summary.
    """
    user_query = config.algorithm.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    
    # Create output directory
    output_dir = Path(__file__).parent / f"{user_query}_success_rates"
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    results_path = output_dir / "success_rates.json"
    output_data = {
        "thresholds": threshold_dict,
        "results": results,
    }
    
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nResults saved to: {results_path}\n")
    
    # Print summary table
    print(f"{'Reward Function':<40} {'Threshold':<12} {'Success Rate':<15} {'Mean':<10}")
    print("-" * 80)
    
    for reward_name, stats in results.items():
        reward_name = "_".join(reward_name.split("_")[1:])
        print(
            f"{reward_name:<40} "
            f"{stats['threshold']:<12.3f} "
            f"{stats['success_rate']:<15.2%} "
            f"{stats['mean']:<10.4f}"
        )
    
    print("="*80 + "\n")


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    
    from dynamic_constraint_rewards.commons import import_dynamic_reward_functions
    
    # Load reward functions
    user_query = cfg.algorithm.ddpo.dynamic_constraint_rewards.user_query
    user_query = user_query.replace(' ', '_').replace('.', '')
    
    # reward_functions = get_all_universal_reward_functions()
    reward_functions = {}
    
    if cfg.algorithm.ddpo.dynamic_constraint_rewards.use:
        dynamic_rewards, _ = import_dynamic_reward_functions(
            reward_code_dir=f"{user_query}_dynamic_reward_functions_final"
        )
        reward_functions.update(dynamic_rewards)
    
    print(f"Loaded {len(reward_functions)} reward functions: {list(reward_functions.keys())}")
    
    # Get thresholds from user
    threshold_dict = get_thresholds_from_user(reward_functions)
    print(f"Thresholds: {threshold_dict}")
    # Sample scenes from baseline
    num_scenes = cfg.get("num_scenes", 1000)
    parsed_scenes, custom_dataset, indices, pkl_path, idx_to_labels_dict = sample_scenes_from_baseline(
        cfg, num_scenes=num_scenes
    )
    
    # Compute success rates
    results = compute_success_rates(
        reward_functions=reward_functions,
        threshold_dict=threshold_dict,
        parsed_scenes=parsed_scenes,
        custom_dataset=custom_dataset,
        indices=indices,
        config=cfg,
        idx_to_labels_dict=idx_to_labels_dict,
        num_classes=22 if cfg.dataset.data.room_type == "bedroom" else 25,
        max_objects=12 if cfg.dataset.data.room_type == "bedroom" else 21,
    )
    
    # Save and display results
    save_results(results, cfg, threshold_dict)


if __name__ == "__main__":
    main()
