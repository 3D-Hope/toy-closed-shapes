"""
Script to analyze variance and distribution of scene parameters.
Analyzes both normalized and unnormalized data for translations, sizes, and angles.
"""

import logging
import os
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from scipy import stats

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
from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
from steerable_scene_generation.utils.omegaconf import register_resolvers

# Add logging filters
filter_drake_vtk_warning()

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


def analyze_parameter_statistics(data, param_name, output_dir, normalized=True):
    """
    Analyze and visualize statistics for a parameter.
    
    Args:
        data: numpy array of shape (n_objects, n_dims)
        param_name: name of the parameter (e.g., 'translations', 'sizes', 'angles')
        output_dir: directory to save plots and statistics
        normalized: whether data is normalized or unnormalized
    """
    data_type = "normalized" if normalized else "unnormalized"
    
    # Calculate statistics
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    var = np.var(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    median = np.median(data, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"{param_name.upper()} - {data_type.upper()}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape}")
    print(f"\nMean: {mean}")
    print(f"Std: {std}")
    print(f"Variance: {var}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Median: {median}")
    print(f"25th percentile: {q25}")
    print(f"75th percentile: {q75}")
    
    # Save statistics to file
    stats_file = output_dir / f"{param_name}_{data_type}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"{param_name.upper()} - {data_type.upper()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Shape: {data.shape}\n\n")
        f.write(f"Mean: {mean}\n")
        f.write(f"Std: {std}\n")
        f.write(f"Variance: {var}\n")
        f.write(f"Min: {min_val}\n")
        f.write(f"Max: {max_val}\n")
        f.write(f"Median: {median}\n")
        f.write(f"25th percentile: {q25}\n")
        f.write(f"75th percentile: {q75}\n")
    
    # Create visualizations
    n_dims = data.shape[1]
    dim_names = {
        'translations': ['X', 'Y', 'Z'],
        'sizes': ['Width', 'Height', 'Depth'],
        'angles': ['Sin', 'Cos']
    }
    
    # Distribution plots
    fig, axes = plt.subplots(1, n_dims, figsize=(5*n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    
    for i in range(n_dims):
        ax = axes[i]
        dim_name = dim_names.get(param_name, [f'Dim{i}'])[i] if i < len(dim_names.get(param_name, [])) else f'Dim{i}'
        
        # Histogram with KDE
        ax.hist(data[:, i], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(data[:, i])
        x = np.linspace(data[:, i].min(), data[:, i].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={sigma:.3f}')
        
        ax.set_xlabel(dim_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{param_name} - {dim_name}\n({data_type})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"{param_name}_{data_type}_distributions.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plots
    fig, ax = plt.subplots(figsize=(max(8, n_dims*2), 6))
    box_data = [data[:, i] for i in range(n_dims)]
    labels = [dim_names.get(param_name, [f'Dim{i}'])[i] if i < len(dim_names.get(param_name, [])) else f'Dim{i}' for i in range(n_dims)]
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{param_name} - Box Plot ({data_type})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    boxplot_file = output_dir / f"{param_name}_{data_type}_boxplot.png"
    plt.savefig(boxplot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap (if multiple dimensions)
    if n_dims > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = np.corrcoef(data.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax.set_xticks(range(n_dims))
        ax.set_yticks(range(n_dims))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Add correlation values
        for i in range(n_dims):
            for j in range(n_dims):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=12)
        
        ax.set_title(f'{param_name} - Correlation Matrix ({data_type})', fontsize=14)
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        corr_file = output_dir / f"{param_name}_{data_type}_correlation.png"
        plt.savefig(corr_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved plots to {output_dir}")
    
    return {
        'mean': mean,
        'std': std,
        'var': var,
        'min': min_val,
        'max': max_val,
        'median': median,
        'q25': q25,
        'q75': q75
    }


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set random seed
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
    
    # Get configuration values
    num_scenes = cfg.get("num_scenes", 100)
    print(f"[INFO] Analyzing {num_scenes} scenes")
    
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
    
    # Set up output directory
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")
    
    # Load checkpoint
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
    
    # Get encoded dataset for post-processing
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    
    print(f"[INFO] Dataset bounds:")
    print(f"  Sizes: {encoded_dataset.bounds['sizes']}")
    print(f"  Translations: {encoded_dataset.bounds['translations']}")
    
    # Create CustomDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )
    
    # Sample scenes
    dataset_size = len(custom_dataset)
    num_scenes_to_sample = min(num_scenes, dataset_size)
    indices = list(range(num_scenes_to_sample))
    
    from torch.utils.data import Subset
    limited_dataset = Subset(custom_dataset, indices)
    
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
    
    print(f"[INFO] Sampling {num_scenes_to_sample} scenes...")
    
    # Build experiment and sample
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
    sampled_scenes = torch.cat(sampled_scene_batches, dim=0)
    
    # Remove NaN samples
    mask = ~torch.any(torch.isnan(sampled_scenes), dim=(1, 2))
    sampled_scenes = sampled_scenes[mask]
    
    print(f"[INFO] Valid samples: {sampled_scenes.shape[0]}")
    
    sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
    
    # Determine number of classes
    if cfg.dataset.data.room_type == "livingroom":
        n_classes = 25
    else:
        n_classes = 22
    
    # Extract normalized parameters
    normalized_translations = []
    normalized_sizes = []
    normalized_angles = []
    
    print(f"[INFO] Extracting normalized parameters...")
    for i in range(sampled_scenes_np.shape[0]):
        for j in range(sampled_scenes_np.shape[1]):
            class_label_idx = np.argmax(sampled_scenes_np[i, j, 8:8+n_classes])
            if class_label_idx != n_classes - 1:  # ignore empty token
                normalized_translations.append(sampled_scenes_np[i, j, 0:3])
                normalized_sizes.append(sampled_scenes_np[i, j, 3:6])
                normalized_angles.append(sampled_scenes_np[i, j, 6:8])
    
    normalized_translations = np.array(normalized_translations)
    normalized_sizes = np.array(normalized_sizes)
    normalized_angles = np.array(normalized_angles)
    
    print(f"[INFO] Extracted {len(normalized_translations)} objects")
    
    # Analyze normalized data
    print("\n" + "="*70)
    print("ANALYZING NORMALIZED DATA")
    print("="*70)
    
    norm_stats = {}
    norm_stats['translations'] = analyze_parameter_statistics(
        normalized_translations, 'translations', output_dir, normalized=True
    )
    norm_stats['sizes'] = analyze_parameter_statistics(
        normalized_sizes, 'sizes', output_dir, normalized=True
    )
    norm_stats['angles'] = analyze_parameter_statistics(
        normalized_angles, 'angles', output_dir, normalized=True
    )
    
    # Unnormalize data using encoded_dataset.post_process
    print("\n" + "="*70)
    print("UNNORMALIZING DATA")
    print("="*70)
    
    unnormalized_translations = []
    unnormalized_sizes = []
    unnormalized_angles = []
    
    # Process in batches
    for i in range(sampled_scenes_np.shape[0]):
        class_labels, translations, sizes, angles = [], [], [], []
        
        for j in range(sampled_scenes_np.shape[1]):
            class_label_idx = np.argmax(sampled_scenes_np[i, j, 8:8+n_classes])
            if class_label_idx != n_classes - 1:  # ignore empty token
                ohe = np.zeros(n_classes - 1)
                ohe[class_label_idx] = 1
                class_labels.append(ohe)
                translations.append(sampled_scenes_np[i, j, 0:3])
                sizes.append(sampled_scenes_np[i, j, 3:6])
                angles.append(sampled_scenes_np[i, j, 6:8])
        
        if len(class_labels) == 0:
            continue
        
        bbox_params_dict = {
            "class_labels": np.array(class_labels)[None, :],
            "translations": np.array(translations)[None, :],
            "sizes": np.array(sizes)[None, :],
            "angles": np.array(angles)[None, :],
        }
        
        try:
            # Post-process to unnormalize
            boxes = encoded_dataset.post_process(bbox_params_dict)
            
            # Extract unnormalized values
            unnormalized_translations.extend(boxes['translations'][0])
            unnormalized_sizes.extend(boxes['sizes'][0])
            unnormalized_angles.extend(boxes['angles'][0])
            
        except Exception as e:
            print(f"[WARNING] Skipping scene {i} due to post_process error: {e}")
            continue
    
    unnormalized_translations = np.array(unnormalized_translations)
    unnormalized_sizes = np.array(unnormalized_sizes)
    unnormalized_angles = np.array(unnormalized_angles)
    
    print(f"[INFO] Unnormalized {len(unnormalized_translations)} objects")
    
    # Analyze unnormalized data
    print("\n" + "="*70)
    print("ANALYZING UNNORMALIZED DATA")
    print("="*70)
    
    unnorm_stats = {}
    unnorm_stats['translations'] = analyze_parameter_statistics(
        unnormalized_translations, 'translations', output_dir, normalized=False
    )
    unnorm_stats['sizes'] = analyze_parameter_statistics(
        unnormalized_sizes, 'sizes', output_dir, normalized=False
    )
    unnorm_stats['angles'] = analyze_parameter_statistics(
        unnormalized_angles, 'angles', output_dir, normalized=False
    )
    
    # Save all statistics to pickle
    all_stats = {
        'normalized': {
            'translations': normalized_translations,
            'sizes': normalized_sizes,
            'angles': normalized_angles,
            'stats': norm_stats
        },
        'unnormalized': {
            'translations': unnormalized_translations,
            'sizes': unnormalized_sizes,
            'angles': unnormalized_angles,
            'stats': unnorm_stats
        }
    }
    
    stats_pkl = output_dir / "parameter_statistics.pkl"
    with open(stats_pkl, 'wb') as f:
        pickle.dump(all_stats, f)
    
    print(f"\n[INFO] All statistics saved to {stats_pkl}")
    print(f"[INFO] All plots saved to {output_dir}")
    print("\n[INFO] Analysis complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
