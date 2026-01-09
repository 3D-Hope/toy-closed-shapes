"""
Script to extract and save floor plan mesh data (vertices/faces) from the dataset.

This script loads the CustomDataset and extracts floor-plan-related geometry from
one or more samples, saving them to disk so they can be reused when sampling or
visualizing floor plans.

The script mirrors the behavior of `extract_fpbpn_from_dataset.py`, but instead
of boundary points + normals, it saves the full floor mesh information exposed
by `CustomDataset.get_floor_plan_args`.

Usage examples (Hydra overrides are optional and can be added later):
    # Extract first sample from the validation split
    python scripts/save_floor_plan_from_dataset.py dataset=custom_scene

    # You can adapt this script similarly to `extract_fpbpn_from_dataset.py`
    # to support command-line overrides for indices, splits, etc.
"""

import logging
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

# Disable tokenizer parallelism and set HuggingFace cache paths as in
# `extract_fpbpn_from_dataset.py` so behavior is consistent.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ[
    "HF_HOME"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
os.environ[
    "HF_DATASETS_CACHE"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for extracting floor-plan mesh data from the dataset.

    The current implementation mirrors the hard-coded settings used in
    `extract_fpbpn_from_dataset.py` for simplicity:
    - Uses the `validation` split
    - Extracts index 0 by default
    - Saves per-sample `.npz` files with vertices/faces/etc.
    - Also saves an aggregated `.pkl` file with all extracted floor plans

    You can extend this later to read indices/splits from the config or
    command-line overrides if desired.
    """

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    dataset_cfg = cfg.dataset

    # Get yaml choices from Hydra runtime config.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(f"[DEBUG] cfg_choice: {cfg_choice}")

    # Ensure `cfg.dataset._name` matches the chosen dataset, as in
    # `extract_fpbpn_from_dataset.py`.
    with open_dict(cfg):
        if cfg_choice.get("dataset") is not None:
            cfg.dataset._name = cfg_choice["dataset"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraction options (kept simple and parallel to `extract_fpbpn_from_dataset.py`).
    # You can later expose these via Hydra config if needed.
    num_samples = 1
    indices_str = "0"  # Extract only index 0 by default
    split = "validation"
    output_prefix = "floor_plan"
    save_metadata = False

    # Create CustomDataset.
    print(f"[INFO] Loading dataset with split: {split}")
    try:
        custom_dataset = CustomDataset(
            cfg=cfg.dataset,
            split=dataset_cfg[split].get(
                "splits", ["test"] if split == "validation" else ["train", "val"]
            ),
            ckpt_path=None,  # No checkpoint needed for extraction
        )
    except Exception as e:
        logging.error(f"Error creating CustomDataset: {e}")
        raise

    dataset_size = len(custom_dataset)
    print(f"[INFO] Dataset size: {dataset_size}")

    # Determine which indices to extract (copied pattern from `extract_fpbpn_from_dataset.py`).
    if indices_str:
        if indices_str.startswith("random:"):
            # Extract random samples
            num_random = int(indices_str.split(":")[1])
            if num_random > dataset_size:
                print(
                    f"[WARNING] Requested {num_random} random samples but dataset only has {dataset_size}. Using all samples."
                )
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
                raise ValueError(
                    f"Invalid indices: {invalid}. Dataset size is {dataset_size}"
                )
    else:
        # Extract first N samples
        num_samples = min(num_samples, dataset_size)
        indices = list(range(num_samples))
        print(f"[INFO] Extracting first {num_samples} samples")

    # Extract floor plan mesh data for each index.
    extracted_floor_plans: dict[int, dict[str, np.ndarray]] = {}
    metadata: dict[int, dict[str, float]] = {}

    print(f"[INFO] Extracting floor-plan mesh data from {len(indices)} samples...")
    for idx in indices:
        try:
            # Use the helper from `CustomDataset` to get all floor-plan-related tensors.
            floor_args = custom_dataset.get_floor_plan_args(idx)

            # Convert all tensors to numpy on CPU.
            floor_args_np: dict[str, np.ndarray] = {}
            for key, value in floor_args.items():
                if isinstance(value, torch.Tensor):
                    floor_args_np[key] = value.detach().cpu().numpy()
                else:
                    floor_args_np[key] = np.asarray(value)

            extracted_floor_plans[idx] = floor_args_np

            if save_metadata and "floor_plan_vertices" in floor_args_np:
                verts = floor_args_np["floor_plan_vertices"]
                # Simple bounding-box metadata for sanity checks / debugging.
                metadata[idx] = {
                    "num_vertices": int(verts.shape[0]),
                    "x_min": float(verts[:, 0].min()),
                    "x_max": float(verts[:, 0].max()),
                    "y_min": float(verts[:, 1].min()),
                    "y_max": float(verts[:, 1].max()),
                    "z_min": float(verts[:, 2].min()),
                    "z_max": float(verts[:, 2].max()),
                }

            print(
                f"[INFO] Extracted floor-plan data from index {idx}: "
                f"keys={list(floor_args_np.keys())}"
            )
        except Exception as e:
            logging.warning(f"Error extracting floor-plan data from index {idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not extracted_floor_plans:
        raise ValueError("No floor-plan data could be extracted from the dataset")

    # Save extracted floor plans.
    print(f"[INFO] Saving {len(extracted_floor_plans)} floor-plan samples...")

    for idx, floor_args_np in extracted_floor_plans.items():
        # Save individual file as a `.npz` containing vertices, faces, etc.
        output_file = output_dir / f"{output_prefix}_idx_{idx}.npz"
        np.savez(output_file, **floor_args_np)
        print(f"[INFO] Saved floor-plan mesh: {output_file}")

    # Save all floor plans in a single `.pkl` file (dictionary mapping idx -> dict).
    all_floor_plans_file = output_dir / f"{output_prefix}_all.pkl"
    with open(all_floor_plans_file, "wb") as f:
        pickle.dump(extracted_floor_plans, f)
    print(f"[INFO] Saved all floor-plan meshes to: {all_floor_plans_file}")

    # Optionally save metadata.
    if save_metadata and metadata:
        metadata_file = output_dir / f"{output_prefix}_metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        print(f"[INFO] Saved floor-plan metadata to: {metadata_file}")

    # Print brief summary.
    print("\n" + "=" * 60)
    print("Floor-Plan Extraction Summary")
    print("=" * 60)
    print(f"Dataset split: {split}")
    print(f"Dataset size: {dataset_size}")
    print(f"Extracted indices: {sorted(extracted_floor_plans.keys())}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""
python scripts/save_floor_plan_from_dataset.py dataset=custom_scene
"""