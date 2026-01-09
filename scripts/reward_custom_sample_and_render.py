# TODO: changed to torch.cat([translations, sizes, angles, class_labels, objfeat], dim=-1), update this to run
# """
# Script for sampling from a trained model using the custom scene dataset.
# """

# import logging
# import os
# import pickle
# import sys

# from pathlib import Path

# import hydra
# import numpy as np
# import torch
# import wandb

# from omegaconf import DictConfig, OmegaConf
# from omegaconf.omegaconf import open_dict
# from threed_front.datasets import get_raw_dataset
# from threed_front.evaluation import ThreedFrontResults

# from steerable_scene_generation.algorithms.scene_diffusion.scene_diffuser_base import (
#     SceneDiffuserBase,
# )
# from steerable_scene_generation.datasets.custom_scene import get_dataset_raw_and_encoded
# from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
#     CustomDataset,
#     update_data_file_paths,
# )
# from steerable_scene_generation.experiments import build_experiment
# from steerable_scene_generation.utils.ckpt_utils import (
#     download_latest_or_best_checkpoint,
#     download_version_checkpoint,
#     is_run_id,
# )
# from steerable_scene_generation.utils.distributed_utils import is_rank_zero
# from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
# from steerable_scene_generation.utils.omegaconf import register_resolvers

# # Add logging filters.
# filter_drake_vtk_warning()

# # Disable tokenizer parallelism.
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ[
#     "HF_HOME"
# ] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
# os.environ[
#     "HF_DATASETS_CACHE"
# ] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


# @hydra.main(version_base=None, config_path="../configurations", config_name="config")
# def main(cfg: DictConfig) -> None:
#     if not is_rank_zero:
#         raise ValueError(
#             "This script must be run on the main process. "
#             "Try export CUDA_VISIBLE_DEVICES=0."
#         )

#     # Set random seed.
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

#     # Resolve the config.
#     register_resolvers()
#     OmegaConf.resolve(cfg)
#     config = cfg.dataset

#     # Check if load path is provided.
#     if "load" not in cfg or cfg.load is None:
#         raise ValueError("Please specify a checkpoint to load with 'load=...'")

#     # Get configuration values with defaults.
#     num_scenes = cfg.get("num_scenes", 1)
#     print(f"[DEBUG] Number of scenes to sample: {num_scenes}")

#     # Set predict mode.
#     cfg.algorithm.predict.do_sample = True
#     cfg.algorithm.predict.do_inference_time_search = False
#     cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
#     cfg.algorithm.predict.do_rearrange = False
#     cfg.algorithm.predict.do_complete = False

#     # Get yaml names.
#     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
#     cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

#     with open_dict(cfg):
#         if cfg_choice["experiment"] is not None:
#             cfg.experiment._name = cfg_choice["experiment"]
#         if cfg_choice["dataset"] is not None:
#             cfg.dataset._name = cfg_choice["dataset"]
#         if cfg_choice["algorithm"] is not None:
#             cfg.algorithm._name = cfg_choice["algorithm"]

#     # Set up the output directory.
#     output_dir = Path(hydra_cfg.runtime.output_dir)
#     logging.info(f"Outputs will be saved to: {output_dir}")

#     # Initialize wandb.
#     if cfg.wandb.project is None:
#         cfg.wandb.project = str(Path(__file__).parent.parent.name)
#     load_id = cfg.load
#     name = f"custom_sampling_{load_id}"
#     wandb.init(
#         name=name,
#         dir=str(output_dir),
#         config=OmegaConf.to_container(cfg),
#         project=cfg.wandb.project,
#         mode=cfg.wandb.mode,
#     )

#     # Load the checkpoint.
#     if is_run_id(load_id):
#         # Download the checkpoint from wandb.
#         run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
#         download_dir = output_dir / "checkpoints"
#         version = cfg.get("checkpoint_version", None)
#         if version is not None and isinstance(version, int):
#             checkpoint_path = download_version_checkpoint(
#                 run_path=run_path, version=version, download_dir=download_dir
#             )
#         else:
#             checkpoint_path = download_latest_or_best_checkpoint(
#                 run_path=run_path,
#                 download_dir=download_dir,
#                 use_best=cfg.get("use_best", False),
#             )
#     else:
#         # Use local path.
#         checkpoint_path = Path(load_id)

#     raw_train_dataset = get_raw_dataset(
#         update_data_file_paths(config["data"], config),
#         # config["data"],
#         split=config["training"].get("splits", ["train", "val"]),
#         include_room_mask=config["network"].get("room_mask_condition", True),
#     )

#     # Get Scaled dataset encoding (without data augmentation)
#     raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
#         update_data_file_paths(config["data"], config),
#         # config["data"],
#         split=config["validation"].get("splits", ["test"]),
#         max_length=config["max_num_objects_per_scene"],
#         include_room_mask=config["network"].get("room_mask_condition", True),
#     )
#     print(
#         f"[Ashok] bounds sizes {encoded_dataset.bounds['sizes']}, translations {encoded_dataset.bounds['translations']}"
#     )

#     print(f"[Ashok] type of dataset {type(encoded_dataset)}")
#     # Create a CustomSceneDataset
#     custom_dataset = CustomDataset(
#         cfg=cfg.dataset,
#         split=config["validation"].get("splits", ["val"]),
#         ckpt_path=str(checkpoint_path),
#     )

#     # Save a ground truth sample from the dataset for comparison
#     gt_sample_idx = 0  # Get the first sample
#     gt_sample = custom_dataset[gt_sample_idx]

#     gt_sample_np = gt_sample["scenes"].detach().cpu().numpy()
#     gt_text_path = output_dir / "ground_truth_scene.txt"
#     with open(gt_text_path, "w") as f:
#         f.write(f"Ground truth scene shape: {gt_sample_np.shape}\n\n")
#         f.write(f"Sample index: {gt_sample_idx}\n\n")
#         f.write(np.array2string(gt_sample_np, threshold=np.inf, precision=6))

#     logging.info(f"Saved ground truth scene to {gt_text_path}")

#     # Limit dataset to num_scenes samples
#     dataset_size = len(custom_dataset)
#     num_scenes_to_sample = min(num_scenes, dataset_size)

#     # Create subset of dataset with only the first num_scenes samples
#     from torch.utils.data import Subset

#     indices = list(range(num_scenes_to_sample))
#     limited_dataset = Subset(custom_dataset, indices)

#     print(f"[DEBUG] Full dataset size: {dataset_size}")
#     print(f"[DEBUG] Sampling {num_scenes_to_sample} scenes")

#     # Use batch size from config, not num_scenes
#     batch_size = cfg.experiment.get("test", {}).get(
#         "batch_size", cfg.experiment.validation.batch_size
#     )
#     print(f"[DEBUG] Using batch size: {batch_size}")

#     # Create a dataloader for the limited dataset
#     dataloader = torch.utils.data.DataLoader(
#         limited_dataset,
#         batch_size=batch_size,
#         num_workers=4,
#         shuffle=False,
#         persistent_workers=False,
#         pin_memory=cfg.experiment.test.pin_memory,
#     )

#     print(f"[DEBUG] Created limited dataset with size: {len(limited_dataset)}")
#     # At line ~190, before model prediction
#     for batch in dataloader:
#         print(f"[DEBUG] Batch keys: {batch.keys()}")
#         print(f"[DEBUG] Has fpbpn: {'fpbpn' in batch}")
#         if "fpbpn" in batch:
#             print(f"[DEBUG] fpbpn shape: {batch['fpbpn'].shape}")
#             print(f"[DEBUG] fpbpn sample values: {batch['fpbpn'][0, :5]}")
#         else:
#             print("[ERROR] fpbpn MISSING - Floor conditioning won't work!")
#         break
#     # Build experiment with custom dataset
#     experiment = build_experiment(cfg, ckpt_path=checkpoint_path)

#     try:
#         print("[DEBUG] Starting to sample scenes...")
#         # Sample scenes from the model
#         sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
#         # TODO: get indices of sampled scenes
#         sampled_indices = list(range(len(sampled_scene_batches[0])))
#         sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

#         print(f"[DEBUG] Sampled scenes shape: {sampled_scenes.shape}")

#         sampled_scenes_np = sampled_scenes.detach().cpu().numpy()  # b, 12, 30

#         # Compute composite reward components for analysis
#         print("\n" + "=" * 80)
#         print("COMPOSITE REWARD ANALYSIS")
#         print("=" * 80)
#         try:
#             from universal_constraint_rewards.commons import (
#                 get_composite_reward,
#                 parse_and_descale_scenes,
#             )
#             from universal_constraint_rewards.gravity_following_reward import (
#                 compute_gravity_following_reward,
#             )

#             # Get composite reward with all components
#             total_rewards, reward_components = get_composite_reward(
#                 sampled_scenes,
#                 num_classes=22,
#                 importance_weights={
#                     "must_have_furniture": 1.5,
#                     "gravity": 1.0,
#                     "non_penetration": 1.0,
#                     "object_count": 0.7,
#                 },
#                 room_type="bedroom",
#             )

#             # Compute averages for each component
#             print(f"\nTotal Composite Reward (avg): {total_rewards.mean().item():.4f}")
#             print(f"Total Composite Reward (std): {total_rewards.std().item():.4f}")
#             print(f"\nIndividual Component Rewards (normalized to [0,1]):")
#             print("-" * 60)

#             for component_name, component_values in reward_components.items():
#                 avg_reward = component_values.mean().item()
#                 std_reward = component_values.std().item()
#                 min_reward = component_values.min().item()
#                 max_reward = component_values.max().item()

#                 print(
#                     f"{component_name:25s}: avg={avg_reward:7.4f}, std={std_reward:7.4f}, "
#                     f"min={min_reward:7.4f}, max={max_reward:7.4f}"
#                 )

#             print("=" * 80 + "\n")

#             # Detailed gravity violation analysis
#             print("\n" + "=" * 80)
#             print("DETAILED GRAVITY VIOLATION ANALYSIS")
#             print("=" * 80)

#             # Parse scenes to get descaled positions and sizes
#             parsed_scene = parse_and_descale_scenes(sampled_scenes, num_classes=22)

#             # Compute raw gravity reward (negative penalty)
#             gravity_raw = compute_gravity_following_reward(parsed_scene)

#             # Convert to numpy for statistics
#             r = gravity_raw.cpu().numpy()

#             # Basic descriptive statistics
#             stats = {
#                 "mean": np.mean(r),
#                 "median": np.median(r),
#                 "std_dev": np.std(r),
#                 "min": np.min(r),
#                 "max": np.max(r),
#                 "iqr": np.percentile(r, 75) - np.percentile(r, 25),
#                 "skewness": ((r - np.mean(r)) ** 3).mean() / (np.std(r) ** 3 + 1e-8),
#                 "kurtosis": ((r - np.mean(r)) ** 4).mean() / (np.std(r) ** 4 + 1e-8),
#             }

#             print("\n=== Raw Gravity Reward Statistics (Negative Penalty) ===")
#             print("-" * 60)
#             for k, v in stats.items():
#                 print(f"{k:20s}: {v:+.6f}")

#             print("\n=== Percentile Distribution ===")
#             print("-" * 60)
#             for p in [5, 25, 50, 75, 95, 99]:
#                 print(f"{p:2d}th percentile: {np.percentile(r, p):+.6f}")

#             print("\n=== Variation Metrics ===")
#             print("-" * 60)
#             batch_var = np.var(r)
#             batch_cv = np.std(r) / (abs(np.mean(r)) + 1e-8)
#             print(f"Variance              : {batch_var:.6f}")
#             print(f"Coefficient of Var    : {batch_cv:.6f}")

#             # Count scenes by severity
#             print("\n=== Violation Severity Breakdown ===")
#             print("-" * 60)
#             perfect_scenes = np.sum(r == 0.0)
#             minor_violations = np.sum((r < 0) & (r >= -0.1))
#             moderate_violations = np.sum((r < -0.1) & (r >= -0.5))
#             severe_violations = np.sum(r < -0.5)

#             total = len(r)
#             print(
#                 f"Perfect (r = 0)       : {perfect_scenes:4d} / {total} ({100*perfect_scenes/total:5.1f}%)"
#             )
#             print(
#                 f"Minor (-0.1 < r < 0)  : {minor_violations:4d} / {total} ({100*minor_violations/total:5.1f}%)"
#             )
#             print(
#                 f"Moderate (-0.5 < r < -0.1): {moderate_violations:4d} / {total} ({100*moderate_violations/total:5.1f}%)"
#             )
#             print(
#                 f"Severe (r < -0.5)     : {severe_violations:4d} / {total} ({100*severe_violations/total:5.1f}%)"
#             )

#             # Physical interpretation
#             print("\n=== Physical Interpretation ===")
#             print("-" * 60)
#             print("NOTE: Gravity reward is NEGATIVE of total floating area (m²)")
#             print(f"Average floating area : {-np.mean(r):.4f} m²")
#             print(f"Median floating area  : {-np.median(r):.4f} m²")
#             print(f"Max floating area     : {-np.min(r):.4f} m² (worst scene)")
#             print(f"Min floating area     : {-np.max(r):.4f} m² (best scene)")

#             print("\n=== Recommended Scale Parameter ===")
#             print("-" * 60)
#             # Recommend scale based on data distribution
#             recommended_scale = np.percentile(
#                 -r, 75
#             )  # Use 75th percentile of violation
#             print(f"Based on 75th percentile violation: {recommended_scale:.4f}")
#             print(
#                 f"This means violations around {recommended_scale:.4f} m² will get ~-0.76 normalized penalty"
#             )
#             print(
#                 f"Current scale in code: Check NORMALIZATION_CONFIG['gravity']['scale']"
#             )

#             print("=" * 80 + "\n")

#         except Exception as e:
#             print(f"Warning: Could not compute composite rewards: {e}")
#             import traceback

#             traceback.print_exc()

#         print(f"[DEBUG] Sampled scenes numpy shape: {sampled_scenes_np.shape}")
#         bbox_params_list = []
#         n_classes = 22  # TODO: make it configurable, it should include empty token
#         path_to_results = output_dir / "sampled_scenes_results.pkl"
#         # n_scenes_with_2_beds = 0
#         # n_scenes_with_sofa = 0
#         # number_of_sofas_in_scenes = []
#         for i in range(sampled_scenes_np.shape[0]):
#             class_labels, translations, sizes, angles = [], [], [], []
#             # n_beds = 0
#             # has_sofa = False
#             # number_of_sofas = 0
#             for j in range(sampled_scenes_np.shape[1]):
#                 # print(f"[Ashok] class probs {sampled_scenes_np[i, j, :n_classes]}")
#                 # if sampled_scenes_np[i, j, 17] > 0:
#                 #     has_sofa = True

#                 class_label_idx = np.argmax(sampled_scenes_np[i, j, :n_classes])
#                 if class_label_idx != n_classes - 1:  # ignore if empty token
#                     # # Note: CLEAN THIS MESS, i did this to quickly test the task specific reward such as must have a sofa reward
#                     # if class_label_idx in [8, 15, 11]:
#                     #     n_beds += 1
#                     # if class_label_idx == 17:
#                     #     number_of_sofas += 1
#                     # continue
#                     ohe = np.zeros(n_classes - 1)
#                     ohe[class_label_idx] = 1
#                     class_labels.append(ohe)
#                     translations.append(
#                         sampled_scenes_np[i, j, n_classes : n_classes + 3]
#                     )
#                     sizes.append(sampled_scenes_np[i, j, n_classes + 3 : n_classes + 6])
#                     angles.append(
#                         sampled_scenes_np[i, j, n_classes + 6 : n_classes + 8]
#                     )
#             # if n_beds == 2:
#             #     n_scenes_with_2_beds += 1
#             # if has_sofa:
#             #     n_scenes_with_sofa += 1
#             # number_of_sofas_in_scenes.append(number_of_sofas)
#             # continue
#             bbox_params_list.append(
#                 {
#                     "class_labels": np.array(class_labels)[None, :],
#                     "translations": np.array(translations)[None, :],
#                     "sizes": np.array(sizes)[None, :],
#                     "angles": np.array(angles)[None, :],
#                 }
#             )
#         # print(
#         #     f"[Ashok] x number of scenes with 2 beds {n_scenes_with_2_beds} out of {sampled_scenes_np.shape[0]}"
#         # )
#         # print(
#         #     f"[Ashok] number of scenes with sofa {n_scenes_with_sofa} out of {sampled_scenes_np.shape[0]}"
#         # )
#         # print(f"[Ashok] number of sofas in scenes {number_of_sofas_in_scenes}")
#         # print(
#         #     f"[Ashok] avg number of sofas in scenes {np.mean(number_of_sofas_in_scenes)}"
#         # )

#         # import sys;sys.exit(0)
#         # print("bbox param list", bbox_params_list)
#         print(f"[Ashok] type of dataset {type(encoded_dataset)}")
#         layout_list = []
#         for bbox_params_dict in bbox_params_list:
#             boxes = encoded_dataset.post_process(bbox_params_dict)
#             bbox_params = {k: v[0] for k, v in boxes.items()}
#             layout_list.append(bbox_params)

#         # print("final output: ", layout_list)
#         # layout_list [{"class_labels":[], "translations":[1,2,3], "sizes": [1,2,3,], "angles": [1]}, ...]
#         threed_front_results = ThreedFrontResults(
#             raw_train_dataset, raw_dataset, config, sampled_indices, layout_list
#         )

#         pickle.dump(threed_front_results, open(path_to_results, "wb"))
#         print("Saved result to:", path_to_results)

#         # TODO: fixme
#         kl_divergence = threed_front_results.kl_divergence()
#         print("object category kl divergence:", kl_divergence)
#         return path_to_results
#     ###---
#     # # Log to wandb and save locally
#     # pickle_path = output_dir / "sampled_scenes.pkl"
#     # with open(pickle_path, "wb") as f:
#     #     pickle.dump(scene_dict, f)

#     # # Also save as text file for easy inspection
#     # text_path = output_dir / "sampled_scenes.txt"
#     # with open(text_path, "w") as f:
#     #     f.write(f"Sampled scenes shape: {sampled_scenes_np.shape}\n\n")
#     #     f.write(np.array2string(sampled_scenes_np, threshold=np.inf, precision=6))

#     # logging.info(f"Saved sampled scenes to {pickle_path} and {text_path}")

#     # # Log to wandb
#     # wandb.save(str(pickle_path))
#     # wandb.save(str(text_path))

#     except Exception as e:
#         raise
#         logging.error(f"Error during sampling: {str(e)}")
#         # Still try to save any partial results
#         try:
#             if "sampled_scenes" in locals():
#                 sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
#                 text_path = output_dir / "partial_sampled_scenes.txt"
#                 with open(text_path, "w") as f:
#                     f.write(
#                         f"Partial sampled scenes shape: {sampled_scenes_np.shape}\n\n"
#                     )
#                     f.write(
#                         np.array2string(
#                             sampled_scenes_np, threshold=np.inf, precision=6
#                         )
#                     )
#                 logging.info(f"Saved partial sampled scenes to {text_path}")
#                 wandb.save(str(text_path))
#         except:
#             pass

#         raise e

#     logging.info("Done!")


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()
