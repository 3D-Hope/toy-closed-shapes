#!/usr/bin/env python3
"""
Standalone script to analyze dataset quality and find problematic scenes.

Usage:
    python analyze_dataset_quality.py --dataset living_room --splits train val test
    python analyze_dataset_quality.py --dataset bedroom --output my_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from steerable_scene_generation.datasets.custom_scene.custom_scene_final import CustomDataset
from steerable_scene_generation.utils.omegaconf import register_resolvers


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file."""
    cfg = OmegaConf.load(config_path)
    register_resolvers()
    OmegaConf.resolve(cfg)
    return cfg


def analyze_dataset(
    cfg: DictConfig,
    splits: list,
    verbose: bool = False,
    output_file: Optional[str] = None,
) -> dict:
    """
    Analyze a dataset for quality issues.
    
    Args:
        cfg: Configuration dictionary
        splits: List of splits to analyze ['train', 'val', 'test']
        verbose: If True, print detailed progress
        output_file: Path to save JSON report (optional)
    
    Returns:
        Analysis report dictionary
    """
    print("\n" + "="*80)
    print("INITIALIZING DATASET ANALYSIS")
    print("="*80)
    print(f"Dataset configuration: {cfg.dataset}")
    print(f"Splits to analyze: {splits}")
    
    # Create dataset
    dataset = CustomDataset(
        cfg=cfg.dataset,
        split=splits,
        ckpt_path=None,
    )
    
    dataset_size = len(dataset)
    print(f"Total scenes to analyze: {dataset_size}\n")
    
    # Analyze all scenes
    print("="*80)
    print("ANALYZING SCENES")
    print("="*80)
    
    checkpoint_interval = max(1, dataset_size // 20)  # 20 progress updates
    
    for i in range(dataset_size):
        # Progress indicator
        if i % checkpoint_interval == 0:
            percentage = 100.0 * i / dataset_size if dataset_size > 0 else 0
            print(f"Progress: {i:6d}/{dataset_size:6d} ({percentage:5.1f}%) analyzed...")
        
        # Load and analyze scene
        try:
            _ = dataset[i]
        except Exception as e:
            logging.warning(f"Error loading scene {i}: {e}")
    
    # Final progress
    print(f"Progress: {dataset_size}/{dataset_size} (100.0%) analyzed...")
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    report = dataset.get_analysis_report()
    
    # Print summary
    dataset.print_analysis_summary()
    
    # Save report if requested
    if output_file:
        dataset.save_analysis_report(output_path=output_file)
        print(f"Report saved to: {output_file}")
    
    return report


def compare_datasets(
    cfg: DictConfig,
    splits: list,
    output_file: Optional[str] = None,
) -> dict:
    """
    Compare quality issues across different dataset configurations.
    """
    print("\n" + "="*80)
    print("COMPARING DATASETS")
    print("="*80)
    
    comparison = {}
    
    for dataset_name in ["bedroom", "living_room"]:
        print(f"\nAnalyzing {dataset_name}...")
        # Assumes config has both bedroom and living_room configs
        try:
            report = analyze_dataset(cfg, splits, output_file=f"{dataset_name}_analysis.json")
            comparison[dataset_name] = report
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for dataset_name, report in comparison.items():
        num_issues = report.get('num_problematic_scenes', 0)
        percentage = report.get('percentage_problematic', 0)
        print(f"{dataset_name:20s}: {num_issues:5d} issues ({percentage:6.2f}%)")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset quality and find problematic scenes"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configurations/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to analyze'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scene_quality_analysis.json',
        help='Output file for analysis report'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple datasets (requires multiple configs)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    cfg = load_config(args.config)
    
    # Run analysis
    try:
        report = analyze_dataset(
            cfg=cfg,
            splits=args.splits,
            verbose=args.verbose,
            output_file=args.output,
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        # Print key statistics
        num_issues = report.get('num_problematic_scenes', 0)
        percentage = report.get('percentage_problematic', 0)
        
        if num_issues > 0:
            print(f"⚠️  Found {num_issues} problematic scenes ({percentage:.2f}% of dataset)")
            print(f"See {args.output} for details")
        else:
            print("✓ No issues found!")
        
        return 0
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
