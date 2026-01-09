import numpy as np
import torch

from universal_constraint_rewards.commons import ceiling_objects, idx_to_labels


def compute_gravity_following_reward(parsed_scene, tolerance=0.01, **kwargs):
    """
    Calculate gravity-following reward based on how close objects are to the ground.

    Objects should rest on the floor (y_min ≈ 0), except for ceiling objects.
    Only penalizes objects that are MORE than tolerance away from the floor(both sinking and floating cases).

    Args:
        parsed_scene: Dict returned by parse_and_descale_scenes()
        tolerance: Distance threshold in meters (default 0.01m = 1cm)

    Returns:
        rewards: Tensor of shape (B,) with gravity-following rewards
    """


    room_type = kwargs["room_type"]
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]


    # Identify ceiling objects
    ceiling_indices = [
        idx for idx, label in idx_to_labels[room_type].items() if label in ceiling_objects
    ]
    is_ceiling = torch.zeros_like(is_empty, dtype=torch.bool)
    for ceiling_idx in ceiling_indices:
        is_ceiling |= object_indices == ceiling_idx

    # Mask: objects that should follow gravity (non-empty, non-ceiling)
    should_follow_gravity = ~is_empty & ~is_ceiling

    # Calculate y_min for each object (bottom of bounding box)
    y_centers = positions[:, :, 1]  # (B, N)
    y_half_extents = sizes[:, :, 1]  # (B, N) - already half-extents
    y_min = y_centers - y_half_extents  # (B, N)

    # Distance from floor (should be ~0 for objects on ground)
    floor_distance = torch.abs(y_min)  # (B, N)

    # Calculate violations: distance beyond tolerance
    # If object is 0.005m off ground and tolerance is 0.01m: no violation (0)
    # If object is 0.05m off ground and tolerance is 0.01m: violation of 0.04m
    violations = torch.clamp(floor_distance - tolerance, min=0.0)  # (B, N)

    # Apply mask: only consider gravity-following objects
    masked_violations = torch.where(
        should_follow_gravity, violations, torch.zeros_like(violations)
    )

    # Sum violations per scene
    total_violation = masked_violations.sum(dim=1)  # (B,)

    # Additional check: if all objects are empty, set reward to zero for that scene
    # (avoiding negative zero if there are no objects to sum over)
    all_empty = is_empty.all(dim=1)  # (B,)
    reward = -total_violation
    reward = torch.where(all_empty, torch.zeros_like(reward), reward)


    return reward


def test_gravity_following_reward():
    """Test cases for gravity following reward."""
    print("\n" + "=" * 60)
    print("Testing Gravity Following Reward")
    print("=" * 60)

    device = "cpu"
    num_classes = 22
    batch_size = 3
    num_objects = 12

    # Helper to create a scene tensor
    def create_scene(object_types, positions_normalized, sizes_normalized):
        """
        object_types: list of object indices (length N)
        positions_normalized: list of [x, y, z] in normalized coords (length N)
        sizes_normalized: list of [x, y, z] in normalized coords (length N)
        """
        scene = torch.zeros(num_objects, 30, device=device)

        for i, (obj_type, pos, size) in enumerate(
            zip(object_types, positions_normalized, sizes_normalized)
        ):
            # One-hot encoding
            scene[i, obj_type] = 1.0
            # Position (normalized -1 to 1)
            scene[i, num_classes : num_classes + 3] = torch.tensor(pos, device=device)
            # Size (normalized -1 to 1)
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                size, device=device
            )
            # Orientation (cos, sin)
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        return scene

    # Test Case 1: Perfect scene - all objects on ground
    print("\nTest 1: All objects perfectly on ground")
    # Chair at y=0 (normalized y=-1 gives actual y≈0.045)
    # Use y=-0.95 to get closer to y=0
    scene1 = create_scene(
        object_types=[
            4,
            4,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # 2 chairs, rest empty
        positions_normalized=[
            [0.0, -0.98, 0.0],  # Chair 1 - very close to ground
            [0.5, -0.98, 0.5],  # Chair 2 - very close to ground
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],  # Small chair
            [0.0, 0.0, 0.0],  # Small chair
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Test Case 2: Objects floating above ground
    print("Test 2: Objects floating above ground")
    scene2 = create_scene(
        object_types=[
            4,
            16,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # chair, sofa, rest empty
        positions_normalized=[
            [0.0, 0.5, 0.0],  # Chair floating high
            [0.5, 0.3, 0.5],  # Sofa floating
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Test Case 3: Mix of ceiling objects and ground objects
    print("Test 3: Ceiling lamp (ignored) + chair on ground")
    scene3 = create_scene(
        object_types=[
            3,
            4,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # ceiling_lamp, chair, rest empty
        positions_normalized=[
            [0.0, 0.8, 0.0],  # Ceiling lamp (should be ignored)
            [0.5, -0.98, 0.5],  # Chair on ground
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Compute rewards
    rewards = compute_gravity_following_reward(parsed)

    print(f"\nResults:")
    print(
        f"Scene 1 (objects on ground): {rewards[0].item():.4f} (should be close to 0)"
    )
    print(f"Scene 2 (floating objects): {rewards[1].item():.4f} (should be negative)")
    print(
        f"Scene 3 (ceiling lamp ignored): {rewards[2].item():.4f} (should be close to 0)"
    )

    # Verify
    assert (
        rewards[0] > rewards[1]
    ), "Scene with grounded objects should have higher reward than floating"
    assert rewards[0] > -0.5, "Grounded scene should have reward close to 0"
    assert rewards[1] < -0.5, "Floating scene should have significantly negative reward"

    print("\n✓ All gravity tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gravity_following_reward()
