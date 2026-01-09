#!/usr/bin/env python
"""
Simple script to download the latest model checkpoint from a wandb run.
Usage: python download_checkpoint.py --run_id bgdrozky [--entity 078bct021-ashok-d] [--project 3dhope_rl] [--output_dir ./checkpoints]
"""

import argparse
import logging

from pathlib import Path

import wandb

from steerable_scene_generation.utils.ckpt_utils import (
    download_latest_or_best_checkpoint,
    is_run_id,
)


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("checkpoint-downloader")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Download the latest checkpoint from a wandb run"
    )
    parser.add_argument("--run_id", type=str, required=True, help="The wandb run ID")
    parser.add_argument(
        "--entity", type=str, default="078bct021-ashok-d", help="The wandb entity"
    )
    parser.add_argument(
        "--project", type=str, default="3dhope_rl", help="The wandb project"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the checkpoint",
    )
    parser.add_argument(
        "--use_best",
        action="store_true",
        help="Download the best checkpoint instead of the latest",
    )
    args = parser.parse_args()

    # Validate run_id format
    if not is_run_id(args.run_id):
        logger.error(f"Invalid run ID format: {args.run_id}")
        return

    # Create the run path
    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    logger.info(f"Downloading checkpoint for run: {run_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Download the checkpoint
    try:
        checkpoint_path = download_latest_or_best_checkpoint(
            run_path=run_path, download_dir=output_dir, use_best=args.use_best
        )
        logger.info(f"Successfully downloaded checkpoint to: {checkpoint_path}")

        # Create a symlink to the latest checkpoint
        latest_link = output_dir / "latest.ckpt"
        latest_link.unlink(missing_ok=True)
        latest_link.symlink_to(checkpoint_path)
        logger.info(f"Created symlink: {latest_link} -> {checkpoint_path}")

    except Exception as e:
        logger.error(f"Error downloading checkpoint: {str(e)}")
        return


if __name__ == "__main__":
    main()
