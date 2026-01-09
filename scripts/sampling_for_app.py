"""
Script for sampling a single scene from a trained model using a custom floor plan.
"""
import argparse
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


def load_fpbpn_from_file(file_path: str) -> np.ndarray:
    """Load fpbpn from a numpy file or pickle file.
    
    Args:
        file_path: Path to the file containing fpbpn.
                  Expected shape: (nfpbpn, 4) where 4 = (x, y, nx, ny)
                  or (1, nfpbpn, 4) for batch dimension.
    
    Returns:
        numpy array of shape (nfpbpn, 4)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Floor plan file not found: {file_path}")
    
    if file_path.suffix == '.npy':
        fpbpn = np.load(file_path)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            fpbpn = pickle.load(f)
    else:
        # Try loading as numpy array
        try:
            fpbpn = np.load(file_path)
        except:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .npy or .pkl")
    
    # Ensure shape is (nfpbpn, 4)
    if fpbpn.ndim == 3:
        # Remove batch dimension if present
        fpbpn = fpbpn[0]
    elif fpbpn.ndim == 1:
        raise ValueError(f"fpbpn should be 2D (nfpbpn, 4), got shape {fpbpn.shape}")
    
    if fpbpn.shape[1] != 4:
        raise ValueError(f"fpbpn should have 4 features (x, y, nx, ny), got {fpbpn.shape[1]}")
    
    # Pad or truncate to 256 points if needed (standard size)
    nfpbpn = 256
    if fpbpn.shape[0] > nfpbpn:
        # Truncate to first 256 points
        fpbpn = fpbpn[:nfpbpn]
        logging.warning(f"Truncated fpbpn from {fpbpn.shape[0]} to {nfpbpn} points")
    elif fpbpn.shape[0] < nfpbpn:
        # Pad with last point
        padding = np.tile(fpbpn[-1:], (nfpbpn - fpbpn.shape[0], 1))
        fpbpn = np.vstack([fpbpn, padding])
        logging.warning(f"Padded fpbpn from {fpbpn.shape[0]} to {nfpbpn} points")
    
    return fpbpn.astype(np.float32)


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

    # Set random seed.
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

    # Get number of scenes to sample (default: 1)
    num_scenes = cfg.get("num_scenes", 1)
    print(f"[INFO] Number of scenes to sample: {num_scenes}")

    # Get custom floor plan path from config
    # custom_fpbpn_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/fpbpn.npy"
    custom_fpbpn_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/polygon_world_fpbpn.npy"
    # custom_fpbpn_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-17/06-52-15/fpbpn_sample_idx_0.npy"
    if custom_fpbpn_path is None:
        raise ValueError(
            "Please specify a custom floor plan with 'custom_fpbpn_path=path/to/fpbpn.npy'"
        )

    # Set predict mode.
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(f"[DEBUG] cfg_choice: {cfg_choice}")

    with open_dict(cfg):
        if cfg_choice.get("experiment") is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice.get("dataset") is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice.get("algorithm") is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")

    # Initialize wandb.
    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)
    load_id = cfg.load
    name = f"custom_floor_sampling_{load_id}"
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
        version = cfg.get("checkpoint_version")
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

    # Load raw dataset for post-processing (needed for ThreedFrontResults)
    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"], config),
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    # Get Scaled dataset encoding (needed for post-processing)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    print(
        f"[Ashok] bounds sizes {encoded_dataset.bounds['sizes']}, translations {encoded_dataset.bounds['translations']}"
    )

    # Create a CustomDataset instance (needed for normalizer and post-processing)
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    # Load custom floor plan
    print(f"[INFO] Loading custom floor plan from: {custom_fpbpn_path}")
    fpbpn = load_fpbpn_from_file(custom_fpbpn_path)
    print(f"[INFO] Loaded fpbpn with shape: {fpbpn.shape}")

    # Convert to torch tensor and add batch dimension
    fpbpn_tensor = torch.from_numpy(fpbpn).unsqueeze(0)  # Shape: (1, nfpbpn, 4)
    print(f"[INFO] fpbpn tensor shape: {fpbpn_tensor.shape}")
    print(f"[INFO] fpbpn tensor: {fpbpn_tensor}")

    # Build experiment
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)

    # Create a simple dataset wrapper that provides our custom fpbpn
    # We need to create a dummy scene tensor for the dataset item
    # Get the expected scene shape from the dataset
    dummy_idx = 0
    dummy_sample = custom_dataset[dummy_idx]
    dummy_scene = dummy_sample["scenes"]  # Shape: (N, V)
    
    # Create a custom dataset item with our fpbpn (repeated for num_scenes)
    class CustomFpbpnDataset(torch.utils.data.Dataset):
        def __init__(self, fpbpn_tensor, dummy_scene, num_scenes):
            self.fpbpn = fpbpn_tensor[0]  # Remove batch dimension, shape: (nfpbpn, 4)
            self.dummy_scene = dummy_scene
            self.num_scenes = num_scenes
        
        def __len__(self):
            return self.num_scenes
        
        def __getitem__(self, idx):
            return {
                "scenes": self.dummy_scene,  # Dummy scene (won't be used, but needed for shape)
                "idx": torch.tensor(idx),
                "fpbpn": self.fpbpn,  # Same fpbpn for all scenes
            }
    
    custom_fpbpn_dataset = CustomFpbpnDataset(fpbpn_tensor, dummy_scene, num_scenes)
    
    # Create dataloader with appropriate batch size
    batch_size = cfg.experiment.get("test", {}).get(
        "batch_size", cfg.experiment.validation.batch_size
    )
    # Don't exceed num_scenes
    batch_size = min(batch_size, num_scenes)
    
    dataloader = torch.utils.data.DataLoader(
        custom_fpbpn_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    
    print(f"[INFO] Created custom dataloader with {num_scenes} scenes, batch_size={batch_size}")
    print(f"[INFO] All scenes will use the same floor plan (fpbpn shape: {fpbpn_tensor.shape})")

    try:
        print(f"[DEBUG] Starting to sample {num_scenes} scene(s) with custom floor plan...")
        
        # Use exec_task("predict", ...) like custom_sample_and_render.py
        # This properly initializes the algo through PyTorch Lightning
        sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
        
        # Concatenate all batches
        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)  # Shape: (num_scenes, N, V)

        print(f"[DEBUG] Sampled scenes shape: {sampled_scenes.shape}")

        # Check for NaN
        mask = ~torch.any(torch.isnan(sampled_scenes), dim=(1, 2))
        sampled_scenes = sampled_scenes[mask]
        num_valid_scenes = sampled_scenes.shape[0]
        
        if num_valid_scenes == 0:
            raise ValueError("All sampled scenes contain NaN values!")
        
        if num_valid_scenes < num_scenes:
            print(f"[WARNING] Only {num_valid_scenes} out of {num_scenes} scenes are valid (NaN filtered)")

        # Save raw sampled scenes
        with open(output_dir / "raw_sampled_scenes.pkl", "wb") as f:
            pickle.dump(sampled_scenes, f)

        sampled_scenes_np = sampled_scenes.detach().cpu().numpy()  # (num_scenes, N, V)
        print(f"[INFO] Processed {num_valid_scenes} valid scene(s), shape: {sampled_scenes_np.shape}")

        # Determine number of classes
        if cfg.dataset.data.room_type == "livingroom":
            n_classes = 25
        else:
            n_classes = 22

        # Parse all scenes into bbox parameters
        bbox_params_list = []
        for i in range(sampled_scenes_np.shape[0]):
            class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
            for j in range(sampled_scenes_np.shape[1]):
                class_label_idx = np.argmax(sampled_scenes_np[i, j, 8 : 8 + n_classes])
                if class_label_idx != n_classes - 1:  # ignore if empty token
                    ohe = np.zeros(n_classes - 1)
                    ohe[class_label_idx] = 1
                    class_labels.append(ohe)
                    translations.append(sampled_scenes_np[i, j, 0:3])
                    sizes.append(sampled_scenes_np[i, j, 3:6])
                    angles.append(sampled_scenes_np[i, j, 6:8])
                    try:
                        objfeats_32.append(
                            sampled_scenes_np[i, j, 8 + n_classes : n_classes + 8 + 32]
                        )
                    except Exception as e:
                        objfeats_32 = None

            bbox_params_dict = {
                "class_labels": np.array(class_labels)[None, :] if class_labels else np.zeros((1, 0, n_classes - 1)),
                "translations": np.array(translations)[None, :] if translations else np.zeros((1, 0, 3)),
                "sizes": np.array(sizes)[None, :] if sizes else np.zeros((1, 0, 3)),
                "angles": np.array(angles)[None, :] if angles else np.zeros((1, 0, 2)),
                "objfeats_32": np.array(objfeats_32)[None, :] if objfeats_32 is not None else None,
            }
            bbox_params_list.append(bbox_params_dict)

        # Post-process all scenes
        layout_list = []
        successful_indices = []
        for idx, bbox_params_dict in enumerate(bbox_params_list):
            try:
                boxes = encoded_dataset.post_process(bbox_params_dict)
                bbox_params = {k: v[0] for k, v in boxes.items()}
                layout_list.append(bbox_params)
                successful_indices.append(idx)
            except Exception as e:
                print(f"[WARNING] Skipping scene {idx} due to post_process error: {e}")
                continue

        if len(layout_list) == 0:
            print(f"[WARNING] Post-processing failed for all scenes. Saving raw scenes...")
            # Save raw scene data
            raw_output = {
                "sampled_scenes": sampled_scenes_np,
                "bbox_params_list": bbox_params_list,
                "fpbpn": fpbpn,
            }
            path_to_results = output_dir / "sampled_scenes_raw.pkl"
            pickle.dump(raw_output, open(path_to_results, "wb"))
            print("Saved raw scenes to:", path_to_results)
            return path_to_results
        
        # Create ThreedFrontResults (using dummy indices since we don't have dataset indices)
        threed_front_results = ThreedFrontResults(
            raw_train_dataset, raw_dataset, config, successful_indices, layout_list
        )
        
        path_to_results = output_dir / "sampled_scenes_results.pkl"
        pickle.dump(threed_front_results, open(path_to_results, "wb"))
        print(f"Saved {len(layout_list)} scene(s) to: {path_to_results}")
        
        # Save custom floor info for rendering script
        # The render script expects this file to use the same custom floor plan for all scenes
        custom_floor_info = {
            "type": "fpbpn",  # Use fpbpn type for polygon floor
            "fpbpn": fpbpn_tensor.numpy(),  # Shape: (1, nfpbpn, 4)
            "fpbpn_path": str(custom_fpbpn_path),  # Store original path for reference
        }
        custom_floor_info_path = output_dir / "custom_floor_info.pkl"
        with open(custom_floor_info_path, "wb") as f:
            pickle.dump(custom_floor_info, f)
        print(f"Saved custom floor info to: {custom_floor_info_path}")
        print(f"  - Floor plan type: {custom_floor_info['type']}")
        print(f"  - fpbpn shape: {custom_floor_info['fpbpn'].shape}")
        print(f"  - Original path: {custom_floor_info['fpbpn_path']}")
        
        # Print summary with output path
        print("\n" + "=" * 80)
        print("✓ SAMPLING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Results file: {path_to_results}")
        print(f"\nTo render the scenes, run:")
        print(f"python ../ThreedFront/scripts/render_results_3d_custom_floor.py \\")
        print(f"  {path_to_results} \\")
        print(f"  --floor_plan_npy /path/to/floor_plan_world.npz \\")
        print(f"  --retrieve_by_size")
        print("=" * 80 + "\n")
        
        with open("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/output_path.txt", "w") as f:
            f.write(f"OUTPUT_DIR={output_dir}")
        return output_dir

    except Exception as e:
        logging.error(f"Error during sampling: {str(e)}")
        import traceback
        traceback.print_exc()
        
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
    
    finally:
        logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""
Input: polygon_world.npy

python scripts/sample_polygon_input.py

python scripts/generate_floor_plan_from_polygon.py tmp/polygon_world.npy --output tmp/floor_plan_world.npz

python ../ThreedFront/scripts/preprocess_floorplan_custom.py tmp/polygon_world.npy --output_fpbpn tmp/polygon_world_fpbpn.npy

python scripts/sampling_for_app.py +num_scenes=5 \
    load=gtjphzpb \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion\
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.classifier_free_guidance.use_floor=true \
    algorithm.custom.old=False \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    wandb.mode=disabled
    
python ../ThreedFront/scripts/render_results_3d_custom_floor.py \
  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-17/16-01-35/sampled_scenes_results.pkl \
  --floor_plan_npy /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/floor_plan_world.npz \
      --retrieve_by_size
"""