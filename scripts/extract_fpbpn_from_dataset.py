"""
Script to extract sample fpbpn (floor plan boundary points normals) from the dataset.

This script loads the CustomDataset and extracts fpbpn from one or more samples,
saving them to files for use with custom floor plan sampling.

Usage:
    # Extract first sample
    python scripts/extract_fpbpn_from_dataset.py dataset=custom_scene extract_num_samples=1

    # Extract specific indices
    python scripts/extract_fpbpn_from_dataset.py dataset=custom_scene extract_indices="0,5,10"

    # Extract random samples
    python scripts/extract_fpbpn_from_dataset.py dataset=custom_scene extract_indices="random:5"

    # Extract with metadata
    python scripts/extract_fpbpn_from_dataset.py dataset=custom_scene extract_num_samples=3 extract_save_metadata=true
"""

import logging
import json
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
    CustomDataset,
    update_data_file_paths,
)
from steerable_scene_generation.utils.omegaconf import register_resolvers

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
    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(f"[DEBUG] cfg_choice: {cfg_choice}")

    with open_dict(cfg):
        if cfg_choice.get("dataset") is not None:
            cfg.dataset._name = cfg_choice["dataset"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get extraction options from config (can be overridden via Hydra command line)
    # indices_str = cfg.get("extract_indices", None)
    # num_samples = cfg.get("extract_num_samples", 1)
    num_samples = 1
    indices_str = "0"
    split = "validation"
    output_prefix = "fpbpn_sample"
    save_metadata = False
    # split = cfg.get("extract_split", "validation")
    # output_prefix = cfg.get("extract_output_prefix", "fpbpn_sample")
    # save_metadata = cfg.get("extract_save_metadata", False)

    # Create CustomDataset
    print(f"[INFO] Loading dataset with split: {split}")
    try:
        custom_dataset = CustomDataset(
            cfg=cfg.dataset,
            split=config[split].get("splits", ["test"] if split == "validation" else ["train", "val"]),
            ckpt_path=None,  # No checkpoint needed for extraction
        )
    except Exception as e:
        logging.error(f"Error creating CustomDataset: {e}")
        raise

    dataset_size = len(custom_dataset)
    print(f"[INFO] Dataset size: {dataset_size}")

    # Determine which indices to extract
    if indices_str:
        if indices_str.startswith("random:"):
            # Extract random samples
            num_random = int(indices_str.split(":")[1])
            if num_random > dataset_size:
                print(f"[WARNING] Requested {num_random} random samples but dataset only has {dataset_size}. Using all samples.")
                num_random = dataset_size
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(dataset_size, size=num_random, replace=False).tolist()
            print(f"[INFO] Selected {len(indices)} random indices: {indices}")
        else:
            # Parse comma-separated indices
            indices = [int(i.strip()) for i in indices_str.split(",")]
            # Validate indices
            invalid = [i for i in indices if i < 0 or i >= dataset_size]
            if invalid:
                raise ValueError(f"Invalid indices: {invalid}. Dataset size is {dataset_size}")
    else:
        # Extract first N samples
        num_samples = min(num_samples, dataset_size)
        indices = list(range(num_samples))
        print(f"[INFO] Extracting first {num_samples} samples")

    # Extract fpbpn from each index
    extracted_fpbpn = {}
    metadata = {}

    print(f"[INFO] Extracting fpbpn from {len(indices)} samples...")
    for idx in indices:
        try:
            sample = custom_dataset[idx]
            
            # Get fpbpn
            fpbpn = sample["fpbpn"]
            
            # Convert to numpy if it's a torch tensor
            if isinstance(fpbpn, torch.Tensor):
                fpbpn_np = fpbpn.detach().cpu().numpy()
            else:
                fpbpn_np = np.array(fpbpn)
            
            # Handle different shapes
            if fpbpn_np.ndim == 3:
                # Remove batch dimension if present
                fpbpn_np = fpbpn_np[0]
            elif fpbpn_np.ndim == 1:
                raise ValueError(f"Unexpected fpbpn shape: {fpbpn_np.shape}")
            
            extracted_fpbpn[idx] = fpbpn_np.astype(np.float32)
            
            # Store metadata
            if save_metadata:
                metadata[idx] = {
                    "shape": fpbpn_np.shape,
                    "x_range": [float(fpbpn_np[:, 0].min()), float(fpbpn_np[:, 0].max())],
                    "y_range": [float(fpbpn_np[:, 1].min()), float(fpbpn_np[:, 1].max())],
                    "nx_range": [float(fpbpn_np[:, 2].min()), float(fpbpn_np[:, 2].max())],
                    "ny_range": [float(fpbpn_np[:, 3].min()), float(fpbpn_np[:, 3].max())],
                }
            
            print(f"[INFO] Extracted fpbpn from index {idx}: shape {fpbpn_np.shape}")
            
        except Exception as e:
            logging.warning(f"Error extracting fpbpn from index {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not extracted_fpbpn:
        raise ValueError("No fpbpn could be extracted from the dataset")

    # Save extracted fpbpn
    print(f"[INFO] Saving {len(extracted_fpbpn)} fpbpn samples...")
    
    for idx, fpbpn in extracted_fpbpn.items():
        # Save individual file
        output_file = output_dir / f"{output_prefix}_idx_{idx}.npy"
        np.save(output_file, fpbpn)
        print(f"[INFO] Saved: {output_file}")
        
        # Also save a visualization-friendly version (just x, y coordinates)
        viz_file = output_dir / f"{output_prefix}_idx_{idx}_viz.npy"
        np.save(viz_file, fpbpn[:, :2])
        print(f"[INFO] Saved visualization points: {viz_file}")

    # Save all fpbpn in a single file (dictionary)
    all_fpbpn_file = output_dir / f"{output_prefix}_all.pkl"
    with open(all_fpbpn_file, "wb") as f:
        pickle.dump(extracted_fpbpn, f)
    print(f"[INFO] Saved all fpbpn to: {all_fpbpn_file}")

    # Save metadata if requested
    if save_metadata and metadata:
        metadata_file = output_dir / f"{output_prefix}_metadata.json"
        import json
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Saved metadata to: {metadata_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Dataset split: {split}")
    print(f"Dataset size: {dataset_size}")
    print(f"Extracted indices: {sorted(extracted_fpbpn.keys())}")
    print(f"Output directory: {output_dir}")
    print("\nSample fpbpn statistics:")
    for idx in sorted(extracted_fpbpn.keys())[:5]:  # Show first 5
        fpbpn = extracted_fpbpn[idx]
        print(f"  Index {idx}: shape {fpbpn.shape}, "
              f"x: [{fpbpn[:, 0].min():.3f}, {fpbpn[:, 0].max():.3f}], "
              f"y: [{fpbpn[:, 1].min():.3f}, {fpbpn[:, 1].max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Usage example:")
    print("=" * 60)
    first_idx = sorted(extracted_fpbpn.keys())[0]
    first_file = output_dir / f"{output_prefix}_idx_{first_idx}.npy"
    print(f"python scripts/sampling_for_app.py \\")
    print(f"    load=your_checkpoint_id \\")
    print(f"    custom_fpbpn_path={first_file} \\")
    print(f"    dataset=custom_scene \\")
    print(f"    algorithm=scene_diffuser_midiffusion")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""
python scripts/extract_fpbpn_from_dataset.py dataset=custom_scene
"""