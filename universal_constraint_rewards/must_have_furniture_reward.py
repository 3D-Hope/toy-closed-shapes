"""
A bedroom must have at least one bed of any type (single_bed, double_bed, kids_bed).
"""

import torch

from universal_constraint_rewards.commons import idx_to_labels


def compute_must_have_furniture_reward(parsed_scene, **kwargs):
    """
    Calculate reward based on whether the scene contains required furniture for the room type.

    For bedrooms: must have at least one bed (single_bed, double_bed, or kids_bed)

    Args:
        parsed_scene: Dict returned by parse_and_descale_scenes()
        room_type: Type of room (default: 'bedroom'). Currently only 'bedroom' is supported.

    Returns:
        rewards: Tensor of shape (B,) with must-have furniture rewards for each scene
    """
    room_type = kwargs["room_type"]
    if room_type != "bedroom":
        raise NotImplementedError(
            f"Room type '{room_type}' is not supported yet. Only 'bedroom' is implemented."
        )

    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]
    batch_size = object_indices.shape[0]
    device = object_indices.device

    # Define required furniture for bedroom
    bed_types = ["single_bed", "double_bed", "kids_bed"]
    bed_indices = [
        int(idx) for idx, label in idx_to_labels[room_type].items() if label in bed_types
    ]

    # Check if each scene has at least one bed
    has_bed = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for bed_idx in bed_indices:
        # Check if any non-empty slot contains this bed type
        has_this_bed = (object_indices == bed_idx) & (~is_empty)
        # If any slot has this bed, mark scene as having a bed
        has_bed |= has_this_bed.any(dim=1)

    # Convert boolean to reward
    # Scenes with bed get reward 0, scenes without bed get large penalty
    reward = torch.where(
        has_bed,
        torch.zeros(batch_size, device=device),  # Has bed: reward = 0
        torch.full((batch_size,), -10.0, device=device),  # No bed: reward = -10
    )

    return reward


def test_must_have_furniture_reward():
    """Test cases for must-have furniture reward."""
    print("\n" + "=" * 60)
    print("Testing Must-Have Furniture Reward (Bedroom)")
    print("=" * 60)

    device = "cpu"
    num_classes = 22
    num_objects = 12

    # Helper to create a scene with specific objects
    def create_scene_with_objects(object_types):
        """Create a scene with specific object types."""
        scene = torch.zeros(num_objects, 30, device=device)

        for i, obj_type in enumerate(object_types):
            if i >= num_objects:
                break
            scene[i, obj_type] = 1.0  # One-hot
            scene[i, num_classes : num_classes + 3] = torch.tensor(
                [0.0, 0.0, 0.0], device=device
            )
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                [0.0, 0.0, 0.0], device=device
            )
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        # Fill remaining with empty
        for i in range(len(object_types), num_objects):
            scene[i, 21] = 1.0  # Empty class

        return scene

    # Test Case 1: Bedroom with single_bed
    print("\nTest 1: Bedroom with single_bed")
    scene1 = create_scene_with_objects([15, 12, 20])  # single_bed, nightstand, wardrobe

    # Test Case 2: Bedroom with double_bed
    print("Test 2: Bedroom with double_bed")
    scene2 = create_scene_with_objects(
        [8, 12, 10]
    )  # double_bed, nightstand, dressing_table

    # Test Case 3: Bedroom with kids_bed
    print("Test 3: Bedroom with kids_bed")
    scene3 = create_scene_with_objects([11, 5, 17])  # kids_bed, children_cabinet, stool

    # Test Case 4: Bedroom WITHOUT any bed (should get penalty)
    print("Test 4: Bedroom WITHOUT bed (invalid)")
    scene4 = create_scene_with_objects(
        [4, 18, 12]
    )  # chair, table, nightstand (no bed!)

    # Test Case 5: Empty bedroom (should get penalty)
    print("Test 5: Empty bedroom (invalid)")
    scene5 = create_scene_with_objects([])  # All empty

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3, scene4, scene5], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Compute rewards
    rewards = compute_must_have_furniture_reward(parsed, room_type="bedroom")

    print(f"\nResults:")
    print(f"Scene 1 (has single_bed): {rewards[0].item():.4f} (should be 0)")
    print(f"Scene 2 (has double_bed): {rewards[1].item():.4f} (should be 0)")
    print(f"Scene 3 (has kids_bed): {rewards[2].item():.4f} (should be 0)")
    print(f"Scene 4 (no bed): {rewards[3].item():.4f} (should be -10)")
    print(f"Scene 5 (empty): {rewards[4].item():.4f} (should be -10)")

    # Verify
    assert rewards[0].item() == 0, "Scene with single_bed should have reward 0"
    assert rewards[1].item() == 0, "Scene with double_bed should have reward 0"
    assert rewards[2].item() == 0, "Scene with kids_bed should have reward 0"
    assert rewards[3].item() == -10, "Scene without bed should have reward -10"
    assert rewards[4].item() == -10, "Empty scene should have reward -10"

    print("\n✓ All must-have furniture tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_must_have_furniture_reward()
