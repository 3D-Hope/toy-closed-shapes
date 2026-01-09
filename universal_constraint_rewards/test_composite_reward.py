"""
Test the composite reward system with all implemented rewards.
"""

import torch

from universal_constraint_rewards.commons import get_composite_reward


def test_composite_reward_system():
    """Test the composite reward system with multiple reward functions."""
    print("\n" + "=" * 70)
    print("Testing Composite Reward System")
    print("=" * 70)

    device = "cpu"
    num_classes = 22
    num_objects = 12

    # Helper to create a scene
    def create_bedroom_scene(objects_data):
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

    # Test Case 1: Good bedroom - has bed, objects on ground, no overlap, good count
    print("\nTest 1: Good bedroom scene")
    scene1 = create_bedroom_scene(
        [
            (8, [-0.6, -0.98, -0.5], [0.0, 0.0, 0.0]),  # double_bed on ground, far left
            (
                12,
                [0.6, -0.98, -0.5],
                [0.0, 0.0, 0.0],
            ),  # nightstand on ground, far right
            (20, [0.0, -0.98, 0.6], [0.0, 0.0, 0.0]),  # wardrobe on ground, back
            (10, [-0.6, -0.98, 0.6], [0.0, 0.0, 0.0]),  # dressing_table on ground
            (
                3,
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.0],
            ),  # ceiling_lamp (ignored in collision)
        ]
    )

    # Test Case 2: Bad bedroom - NO bed (major penalty)
    print("Test 2: Bad bedroom - missing bed")
    scene2 = create_bedroom_scene(
        [
            (4, [-0.5, -0.98, -0.5], [0.0, 0.0, 0.0]),  # chair (not a bed!)
            (18, [0.5, -0.98, 0.5], [0.0, 0.0, 0.0]),  # table
            (12, [0.0, -0.98, 0.0], [0.0, 0.0, 0.0]),  # nightstand
        ]
    )

    # Test Case 3: Bedroom with objects floating (gravity penalty)
    print("Test 3: Bedroom with floating furniture")
    scene3 = create_bedroom_scene(
        [
            (15, [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]),  # single_bed floating!
            (12, [0.5, 0.3, 0.5], [0.0, 0.0, 0.0]),  # nightstand floating!
        ]
    )

    # Test Case 4: Bedroom with overlapping objects (penetration penalty)
    print("Test 4: Bedroom with overlapping furniture")
    scene4 = create_bedroom_scene(
        [
            (8, [0.0, -0.98, 0.0], [0.4, 0.0, 0.4]),  # double_bed
            (12, [0.0, -0.98, 0.0], [0.3, 0.0, 0.3]),  # nightstand at same position!
            (20, [0.0, -0.98, 0.0], [0.3, 0.0, 0.3]),  # wardrobe at same position!
        ]
    )

    # Test Case 5: Empty bedroom (multiple penalties)
    print("Test 5: Empty bedroom")
    scene5 = create_bedroom_scene([])

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3, scene4, scene5], dim=0)

    # Define reward weights (BALANCED configuration from scale analysis)
    reward_weights = {
        "gravity": 5.0,  # Scale up small penalties
        "non_penetration": 0.5,  # Scale DOWN large penalties
        "must_have_furniture": 1.0,  # Keep as-is
        "object_count": 1.0,  # Keep as-is
    }

    print(f"\nUsing BALANCED weights (normalized to ~10 penalty range):")
    print(
        f"  gravity: {reward_weights['gravity']} (raw range: -2.7 to 0 → weighted: -13.5 to 0)"
    )
    print(
        f"  non_penetration: {reward_weights['non_penetration']} (raw range: -25 to 0 → weighted: -12.5 to 0)"
    )
    print(
        f"  must_have: {reward_weights['must_have_furniture']} (raw range: -10 to 0 → weighted: -10 to 0)"
    )
    print(
        f"  object_count: {reward_weights['object_count']} (raw range: -9 to -1 → weighted: -9 to -1)"
    )

    # Compute composite rewards
    total_rewards, components = get_composite_reward(
        scenes,
        num_classes=num_classes,
        reward_weights=reward_weights,
        room_type="bedroom",
    )

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)

    scene_names = [
        "Good bedroom",
        "Missing bed",
        "Floating furniture",
        "Overlapping furniture",
        "Empty bedroom",
    ]

    for i, name in enumerate(scene_names):
        print(f"\n{name}:")
        print(f"  Total Reward: {total_rewards[i].item():.4f}")
        for component_name, component_values in components.items():
            weight = reward_weights[component_name]
            print(
                f"    {component_name:20s}: {component_values[i].item():8.4f} (weight: {weight:.1f}, weighted: {component_values[i].item() * weight:8.4f})"
            )

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)

    # Verify expected orderings
    print(f"\nScene Rankings (by total reward):")
    sorted_indices = torch.argsort(total_rewards, descending=True)
    for rank, idx in enumerate(sorted_indices):
        print(
            f"  {rank+1}. {scene_names[idx]:25s} (reward: {total_rewards[idx].item():8.4f})"
        )

    # Assertions
    assert total_rewards[0] > total_rewards[1], "Good bedroom should beat missing bed"
    assert (
        total_rewards[0] > total_rewards[2]
    ), "Good bedroom should beat floating furniture"
    assert (
        total_rewards[0] > total_rewards[3]
    ), "Good bedroom should beat overlapping furniture"
    assert total_rewards[0] > total_rewards[4], "Good bedroom should beat empty bedroom"

    print("\n✓ All composite reward tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_composite_reward_system()
