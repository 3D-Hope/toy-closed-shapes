#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic reward functions for scene generation.
These functions can be passed to mutable_reward in ddpo_helpers.py.

Usage:
    from dynamic_rewards import double_bed_reward_function
    rewards = mutable_reward(scenes, scene_vec_desc, double_bed_reward_function, cfg)

Author: GitHub Copilot
Date: October 9, 2025
"""

import numpy as np
import torch


def object_count_target_reward_function(
    scene_data, scene_vec_desc=None, cfg=None, target_count=8
):
    """
    Creates a reward based on how close the scene is to having exactly 'target_count' objects.
    Reward = -|n_objects - target_count|²

    Args:
        scene_data (dict): Dictionary with scene information
        scene_vec_desc: Not used but included for API compatibility
        cfg: Optional configuration
        target_count (int): Target number of objects

    Returns:
        torch.Tensor: Rewards for each scene in the batch
    """
    class_labels = scene_data["class_labels"]
    batch_size = class_labels.shape[0]
    num_classes = scene_data["num_classes"]

    rewards = torch.zeros(batch_size, device=class_labels.device)

    for i in range(batch_size):
        # Count non-empty objects (assuming last class is 'empty')
        non_empty_mask = torch.argmax(class_labels[i, :, :num_classes], dim=1) != (
            num_classes - 1
        )
        object_count = torch.sum(non_empty_mask).item()

        # Calculate squared difference from target count (negative because higher reward is better)
        rewards[i] = -((object_count - target_count) ** 2)

    return rewards


def room_balance_reward_function(scene_data, scene_vec_desc=None, cfg=None):
    """
    Creates a reward based on how well objects are balanced across the room
    (not clustered on one side).

    Args:
        scene_data (dict): Dictionary with scene information
        scene_vec_desc: Not used but included for API compatibility
        cfg: Optional configuration

    Returns:
        torch.Tensor: Rewards for each scene in the batch
    """
    positions = scene_data["positions"]  # Shape: B x N x 3
    class_labels = scene_data["class_labels"]  # Shape: B x N x num_classes
    num_classes = scene_data["num_classes"]
    batch_size = positions.shape[0]

    rewards = torch.zeros(batch_size, device=positions.device)

    for i in range(batch_size):
        # Get positions of non-empty objects
        non_empty_mask = torch.argmax(class_labels[i, :, :num_classes], dim=1) != (
            num_classes - 1
        )
        valid_positions = positions[i, non_empty_mask, :]

        if valid_positions.shape[0] <= 1:
            # Not enough objects to calculate balance
            rewards[i] = 0
            continue

        # Calculate centroid of the room (average position of all objects)
        centroid = torch.mean(valid_positions, dim=0)

        # Calculate the standard deviation of distances from centroid
        # (measure of how spread out objects are)
        distances = torch.norm(valid_positions - centroid, dim=1)
        std_distance = torch.std(distances)

        # We want a moderate spread - not too tight, not too far
        # Use a Gaussian-like function centered at an ideal standard deviation
        ideal_std = 1.5  # Can be tuned
        spread_score = torch.exp(-0.5 * ((std_distance - ideal_std) / 0.5) ** 2)

        rewards[i] = spread_score

    return rewards


def double_bed_reward_function(scene_data, scene_vec_desc=None, cfg=None):
    """
    Creates a reward for scenes that have exactly one double bed positioned
    against a wall and at least two nightstands nearby.

    Args:
        scene_data (dict): Dictionary with scene information
        scene_vec_desc: Not used but included for API compatibility
        cfg: Optional configuration

    Returns:
        torch.Tensor: Rewards for each scene in the batch
    """
    positions = scene_data["positions"]  # Shape: B x N x 3
    class_labels = scene_data["class_labels"]  # Shape: B x N x num_classes
    batch_size = positions.shape[0]
    class_to_name = scene_data["class_to_name_map"]

    # Find indices of important objects
    double_bed_idx = next(
        idx for idx, name in class_to_name.items() if name == "double_bed"
    )
    nightstand_idx = next(
        idx for idx, name in class_to_name.items() if name == "nightstand"
    )

    rewards = torch.zeros(batch_size, device=positions.device)

    for i in range(batch_size):
        # Find double beds
        double_bed_mask = torch.argmax(class_labels[i], dim=1) == double_bed_idx
        double_bed_count = torch.sum(double_bed_mask).item()

        # Find nightstands
        nightstand_mask = torch.argmax(class_labels[i], dim=1) == nightstand_idx
        nightstand_count = torch.sum(nightstand_mask).item()

        # Base reward starts at 0
        scene_reward = 0.0

        # If exactly one double bed
        if double_bed_count == 1:
            scene_reward += 1.0

            # Check if bed is positioned against a wall (using X or Z coordinates near bounds)
            bed_position = positions[i, double_bed_mask, :][0]
            room_bounds = torch.tensor(
                [
                    [-2.7, 2.7],  # X min/max
                    [0.0, 3.6],  # Y min/max
                    [-2.7, 2.8],  # Z min/max
                ],
                device=positions.device,
            )

            # Check if the bed is close to any wall (within 0.3 units)
            wall_threshold = 0.3
            near_wall_x = (
                torch.abs(bed_position[0] - room_bounds[0, 0]) < wall_threshold
                or torch.abs(bed_position[0] - room_bounds[0, 1]) < wall_threshold
            )
            near_wall_z = (
                torch.abs(bed_position[2] - room_bounds[2, 0]) < wall_threshold
                or torch.abs(bed_position[2] - room_bounds[2, 1]) < wall_threshold
            )

            if near_wall_x or near_wall_z:
                scene_reward += 1.0

            # Check if there are nightstands near the bed
            if nightstand_count >= 1:
                # Get positions of all nightstands
                nightstand_positions = positions[i, nightstand_mask, :]
                bed_position = positions[i, double_bed_mask, :][0]

                # Calculate distances from nightstands to bed
                distances = torch.norm(nightstand_positions - bed_position, dim=1)

                # Count how many nightstands are close to the bed
                close_nightstands = torch.sum(distances < 1.0).item()

                # Reward based on number of close nightstands (up to 2)
                scene_reward += min(close_nightstands, 2) * 0.5

        # Penalty for having more than one double bed
        elif double_bed_count > 1:
            scene_reward -= 1.0

        rewards[i] = scene_reward

    return rewards


def bedroom_composition_reward_function(scene_data, scene_vec_desc=None, cfg=None):
    """
    Comprehensive bedroom composition reward based on:
    1. Having exactly one bed (single or double)
    2. Having 1-2 nightstands near the bed
    3. Having a wardrobe
    4. Proper spacing between objects (no overlap)
    5. Bed positioned against a wall

    Args:
        scene_data (dict): Dictionary with scene information
        scene_vec_desc: Not used but included for API compatibility
        cfg: Optional configuration

    Returns:
        torch.Tensor: Rewards for each scene in the batch
    """
    positions = scene_data["positions"]  # B x N x 3
    sizes = scene_data["sizes"]  # B x N x 3
    class_labels = scene_data["class_labels"]  # B x N x num_classes
    batch_size = positions.shape[0]
    class_to_name = scene_data["class_to_name_map"]

    # Find indices of important objects
    bed_indices = [
        idx
        for idx, name in class_to_name.items()
        if "bed" in name and name != "dressing_table"
    ]
    nightstand_idx = next(
        (idx for idx, name in class_to_name.items() if name == "nightstand"), None
    )
    wardrobe_idx = next(
        (idx for idx, name in class_to_name.items() if name == "wardrobe"), None
    )

    rewards = torch.zeros(batch_size, device=positions.device)

    for i in range(batch_size):
        scene_reward = 0.0
        object_classes = torch.argmax(class_labels[i], dim=1)

        # Count beds
        bed_mask = torch.zeros_like(object_classes, dtype=torch.bool)
        for bed_idx in bed_indices:
            bed_mask = bed_mask | (object_classes == bed_idx)

        bed_count = torch.sum(bed_mask).item()

        # 1. Reward for exactly one bed
        if bed_count == 1:
            scene_reward += 2.0
        elif bed_count > 1:
            scene_reward -= 1.0  # Penalty for multiple beds
        else:
            scene_reward -= 2.0  # Penalty for no bed

        if bed_count >= 1:
            # Get the bed position and size
            bed_position = positions[i, bed_mask, :][0]
            bed_size = sizes[i, bed_mask, :][0]

            # 2. Check if bed is against a wall
            room_bounds = torch.tensor(
                [
                    [-2.7, 2.7],  # X min/max
                    [0.0, 3.6],  # Y min/max
                    [-2.7, 2.8],  # Z min/max
                ],
                device=positions.device,
            )

            wall_threshold = 0.3
            near_wall_x = (
                torch.abs(bed_position[0] - room_bounds[0, 0]) < wall_threshold
                or torch.abs(bed_position[0] - room_bounds[0, 1]) < wall_threshold
            )
            near_wall_z = (
                torch.abs(bed_position[2] - room_bounds[2, 0]) < wall_threshold
                or torch.abs(bed_position[2] - room_bounds[2, 1]) < wall_threshold
            )

            if near_wall_x or near_wall_z:
                scene_reward += 1.0

            # 3. Check for nightstands near the bed
            if nightstand_idx is not None:
                nightstand_mask = object_classes == nightstand_idx
                nightstand_count = torch.sum(nightstand_mask).item()

                if nightstand_count > 0:
                    # Get positions of all nightstands
                    nightstand_positions = positions[i, nightstand_mask, :]

                    # Calculate distances from nightstands to bed
                    distances = torch.norm(nightstand_positions - bed_position, dim=1)

                    # Count how many nightstands are close to the bed
                    close_nightstands = torch.sum(distances < 1.0).item()

                    # Reward based on number of close nightstands (up to 2)
                    scene_reward += min(close_nightstands, 2) * 0.5

                    # Penalty for too many nightstands
                    if nightstand_count > 2:
                        scene_reward -= (nightstand_count - 2) * 0.2

        # 4. Check for wardrobe
        if wardrobe_idx is not None:
            wardrobe_mask = object_classes == wardrobe_idx
            wardrobe_count = torch.sum(wardrobe_mask).item()

            if wardrobe_count == 1:
                scene_reward += 1.0
            elif wardrobe_count > 1:
                scene_reward += 0.5  # Less reward for multiple wardrobes

        # 5. Check for overlapping objects (rough approximation)
        non_empty_mask = object_classes != (scene_data["num_classes"] - 1)
        valid_positions = positions[i, non_empty_mask, :]
        valid_sizes = sizes[i, non_empty_mask, :]

        if len(valid_positions) > 1:
            # Check for each pair of objects
            overlap_penalty = 0
            for j in range(len(valid_positions)):
                for k in range(j + 1, len(valid_positions)):
                    # Check if bounding boxes overlap
                    pos1, size1 = valid_positions[j], valid_sizes[j]
                    pos2, size2 = valid_positions[k], valid_sizes[k]

                    # Calculate distances between centers in each dimension
                    distances = torch.abs(pos1 - pos2)

                    # Calculate minimum distances needed to avoid overlap
                    min_distances = (size1 + size2) / 2

                    # Check if there's overlap in all dimensions
                    overlap = torch.all(distances < min_distances).item()

                    if overlap:
                        overlap_penalty += 0.3

            scene_reward -= min(overlap_penalty, 2.0)  # Cap the penalty

        rewards[i] = scene_reward

    return rewards


def custom_object_distribution_reward_function(
    scene_data, scene_vec_desc=None, cfg=None, expected_counts=None
):
    """
    Creates a reward based on how well the scene matches an expected distribution of objects.
    Reward = -∑|count(object_type) - expected_count(object_type)|²

    Args:
        scene_data (dict): Dictionary with scene information
        scene_vec_desc: Not used but included for API compatibility
        cfg: Optional configuration
        expected_counts (dict): Dictionary mapping object class indices to expected counts
                              If None, uses a default bedroom distribution

    Returns:
        torch.Tensor: Rewards for each scene in the batch
    """
    class_labels = scene_data["class_labels"]
    batch_size = class_labels.shape[0]

    # Default expected counts for a standard bedroom if none provided
    if expected_counts is None:
        expected_counts = {
            8: 1,  # double_bed: 1
            12: 2,  # nightstand: 2
            20: 1,  # wardrobe: 1
            0: 1,  # armchair: 1
            4: 1,  # chair: 1
        }

    rewards = torch.zeros(batch_size, device=class_labels.device)

    for i in range(batch_size):
        # Get class distribution for this scene
        object_classes = torch.argmax(class_labels[i], dim=1)

        # Count non-empty objects by class
        class_counts = {}
        for cls in expected_counts.keys():
            class_counts[cls] = torch.sum(object_classes == cls).item()

        # Calculate penalty as sum of squared differences from expected counts
        penalty = 0
        for cls, expected in expected_counts.items():
            actual = class_counts.get(cls, 0)
            penalty += (actual - expected) ** 2

        # Higher reward for closer match to expected distribution
        rewards[i] = -penalty

    return rewards


# Example usage in your code:
"""
from steerable_scene_generation.algorithms.scene_diffusion.dynamic_rewards import (
    double_bed_reward_function,
    bedroom_composition_reward_function,
    custom_object_distribution_reward_function
)

# In your reward calculation code:
rewards = mutable_reward(
    scenes, 
    scene_vec_desc, 
    bedroom_composition_reward_function,
    cfg
)

# Or with custom object distribution:
expected_counts = {8: 1, 12: 2, 20: 1}  # 1 double_bed, 2 nightstands, 1 wardrobe
rewards = mutable_reward(
    scenes, 
    scene_vec_desc,
    lambda scene_data, scene_vec_desc, cfg: 
        custom_object_distribution_reward_function(scene_data, scene_vec_desc, cfg, expected_counts),
    cfg
)
"""
