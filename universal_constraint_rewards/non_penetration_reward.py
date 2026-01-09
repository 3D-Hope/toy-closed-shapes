"""
Reward for non-penetration constraint using penetration depth.

Uses simplified penetration depth calculation based on AABB (Axis-Aligned Bounding Box) overlap.
Following the approach from the original authors who use sum of negative signed distances.

Penetration depth is better than overlap area or IoU because:
1. Provides magnitude information (how much to separate objects)
2. Physical meaning - directly actionable for corrections
3. Better gradient information for RL optimization
4. Handles different object scales consistently
"""

import torch

from universal_constraint_rewards.commons import ceiling_objects, idx_to_labels


def compute_aabb_penetration_depth(centers1, sizes1, centers2, sizes2):
    """
    Compute penetration depth between two sets of AABBs.

    Penetration depth is defined as the minimum translation distance needed to
    separate two overlapping objects. For AABBs, this is the minimum overlap
    across all axes.

    Args:
        centers1: (B, N1, 3) - centers of first set of boxes
        sizes1: (B, N1, 3) - sizes of first set of boxes
        centers2: (B, N2, 3) - centers of second set of boxes
        sizes2: (B, N2, 3) - sizes of second set of boxes

    Returns:
        penetration_depth: (B, N1, N2) - penetration depth (0 if no overlap)
    """
    batch_size = centers1.shape[0]
    device = centers1.device

    # Expand dimensions for pairwise comparison
    c1 = centers1.unsqueeze(2)  # (B, N1, 1, 3)
    s1 = sizes1.unsqueeze(2)  # (B, N1, 1, 3)
    c2 = centers2.unsqueeze(1)  # (B, 1, N2, 3)
    s2 = sizes2.unsqueeze(1)  # (B, 1, N2, 3)

    # NOTE: sizes are already half-extents (sx/2, sy/2, sz/2)
    # So we use them directly
    half1 = s1  # (B, N1, 1, 3) - already half-extents
    half2 = s2  # (B, 1, N2, 3) - already half-extents

    # Compute center distance for each axis
    center_dist = torch.abs(c1 - c2)  # (B, N1, N2, 3)

    # Compute sum of half-extents for each axis
    sum_half_extents = half1 + half2  # (B, N1, N2, 3)

    # Overlap on each axis = sum_of_half_extents - center_distance
    # If positive, there's overlap; if negative or zero, no overlap
    overlap_per_axis = sum_half_extents - center_dist  # (B, N1, N2, 3)

    # For AABBs to overlap, they must overlap on ALL axes
    # Check if overlapping on all axes
    is_overlapping_all_axes = (overlap_per_axis > 0).all(dim=3)  # (B, N1, N2)

    # Penetration depth is the MINIMUM overlap across all axes
    # (the smallest amount needed to separate the objects)
    min_overlap_per_pair = overlap_per_axis.min(dim=3)[0]  # (B, N1, N2)

    # Only consider penetration where objects actually overlap on all axes
    penetration_depth = torch.where(
        is_overlapping_all_axes,
        min_overlap_per_pair,
        torch.zeros_like(min_overlap_per_pair),
    )

    # Clamp to ensure non-negative
    penetration_depth = torch.clamp(penetration_depth, min=0.0)

    return penetration_depth


def compute_non_penetration_reward(parsed_scenes, **kwargs):
    """
    Calculate reward based on non-penetration constraint using penetration depth.

    Following the approach from original authors: reward = sum of negative signed distances.
    When objects overlap, we get positive penetration depth, so reward is negative.

    Args:
        parsed_scenes: Dict returned by parse_and_descale_scenes()

    Returns:
        rewards: Tensor of shape (B,) with non-penetration rewards for each scene
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]
    sizes = parsed_scenes["sizes"]
    object_indices = parsed_scenes["object_indices"]
    is_empty = parsed_scenes["is_empty"]
    batch_size = positions.shape[0]
    device = positions.device
    
    # print(f"Parsed scene: pos {positions[:10]} sizes: {sizes[:10]}")
    # print(f"Parsed scene: {parsed_scenes}")

    # Identify ceiling objects (they don't participate in ground-level collisions)
    ceiling_indices = [
        idx for idx, label in idx_to_labels[room_type].items() if label in ceiling_objects
    ]
    is_ceiling = torch.zeros_like(is_empty, dtype=torch.bool)
    for ceiling_idx in ceiling_indices:
        is_ceiling |= object_indices == ceiling_idx

    # Create mask for ground objects (non-empty, non-ceiling)
    is_ground_object = ~is_empty & ~is_ceiling

    # Compute pairwise penetration depths
    penetration_depth = compute_aabb_penetration_depth(
        positions, sizes, positions, sizes
    )  # (B, N, N)

    # Create mask to ignore self-overlaps (diagonal), empty objects, and ceiling objects
    mask = is_ground_object.unsqueeze(2) & is_ground_object.unsqueeze(1)  # (B, N, N)

    # Remove self-overlaps (diagonal)
    eye = torch.eye(positions.shape[1], device=device, dtype=torch.bool)
    eye = eye.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)
    mask = mask & ~eye

    # Apply mask to penetration depths
    masked_penetration = torch.where(
        mask, penetration_depth, torch.zeros_like(penetration_depth)
    )

    # Sum total penetration depth per scene
    # Divide by 2 because each pair is counted twice (i,j) and (j,i)
    total_penetration = masked_penetration.sum(dim=[1, 2]) / 2.0  # (B,)

    # Convert to reward: +1 if no penetration, else -penetration_depth
    # This provides a clear positive signal for valid scenes and negative for invalid ones
    rewards = torch.where(
        total_penetration == 0,
        torch.ones_like(total_penetration),
        -total_penetration
    )
    return rewards


def test_non_penetration_reward():
    """Test cases for non-penetration reward."""
    print("\n" + "=" * 60)
    print("Testing Non-Penetration Reward (Penetration Depth)")
    print("=" * 60)

    device = "cpu"
    num_classes = 22
    num_objects = 12

    # Helper to create a scene with specific positions and sizes
    def create_scene_with_positions(objects_data):
        """
        objects_data: list of tuples (obj_type, position_norm, size_norm)
        """
        scene = torch.zeros(num_objects, 30, device=device)

        for i, (obj_type, pos, size) in enumerate(objects_data):
            if i >= num_objects:
                break
            scene[i, obj_type] = 1.0  # One-hot
            scene[i, num_classes : num_classes + 3] = torch.tensor(pos, device=device)
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                size, device=device
            )
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        # Fill remaining with empty
        for i in range(len(objects_data), num_objects):
            scene[i, 21] = 1.0  # Empty class

        return scene

    # Test Case 1: No overlap - objects well separated
    print("\nTest 1: No overlap - objects well separated")
    scene1 = create_scene_with_positions(
        [
            (4, [-0.5, 0.0, -0.5], [0.1, 0.1, 0.1]),  # Chair at one corner
            (4, [0.5, 0.0, 0.5], [0.1, 0.1, 0.1]),  # Chair at opposite corner
        ]
    )

    # Test Case 2: Objects completely overlapping (same position, same size)
    print("Test 2: Objects completely overlapping")
    scene2 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.5, 0.1, 0.5]),  # Chair
            (18, [0.0, 0.0, 0.0], [0.5, 0.1, 0.5]),  # Table at exact same position
        ]
    )

    # Test Case 3: Ceiling lamp should be ignored (even if it overlaps in XZ projection)
    print("Test 3: Ceiling lamp overlapping chair in XZ (should be ignored)")
    scene3 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.3, 0.1, 0.3]),  # Chair on ground
            (3, [0.0, 0.8, 0.0], [0.3, 0.1, 0.3]),  # Ceiling lamp above (no Y overlap!)
        ]
    )

    # Test Case 4: Partial overlap (small penetration)
    print("Test 4: Partial overlap")
    scene4 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.4, 0.1, 0.4]),  # Chair
            (18, [0.3, 0.0, 0.0], [0.4, 0.1, 0.4]),  # Table slightly overlapping in X
        ]
    )

    # Test Case 5: Large penetration (deeply overlapping)
    print("Test 5: Large penetration")
    scene5 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.6, 0.2, 0.6]),  # Large chair
            (18, [0.1, 0.0, 0.1], [0.6, 0.2, 0.6]),  # Large table, mostly overlapping
        ]
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3, scene4, scene5], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Compute rewards
    rewards = compute_non_penetration_reward(parsed)

    print(f"\nResults:")
    print(f"Scene 1 (no overlap): {rewards[0].item():.4f} (should be 0)")
    print(
        f"Scene 2 (complete overlap): {rewards[1].item():.4f} (should be very negative)"
    )
    print(f"Scene 3 (ceiling lamp ignored): {rewards[2].item():.4f} (should be 0)")
    print(
        f"Scene 4 (partial overlap): {rewards[3].item():.4f} (should be moderately negative)"
    )
    print(
        f"Scene 5 (large penetration): {rewards[4].item():.4f} (should be very negative)"
    )

    # Verify
    assert abs(rewards[0].item()) < 0.01, "No overlap should have reward 0"
    assert rewards[1].item() < -0.5, "Complete overlap should have very negative reward"
    assert abs(rewards[2].item()) < 0.01, "Ceiling lamp should be ignored"
    assert rewards[3].item() < -0.05, "Partial overlap should have negative reward"
    # Note: penetration depth is minimum separation distance, so complete and partial
    # overlaps can have similar depths depending on geometry
    assert (
        rewards[4].item() < -0.5
    ), "Large penetration should have very negative reward"

    print("\n✓ All non-penetration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_non_penetration_reward()
