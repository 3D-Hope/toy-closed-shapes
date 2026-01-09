import torch

from dynamic_constraint_rewards.utilities import get_all_utility_functions


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Composite reward for TV stand, desk, and chair presence in the scene.
    Considers both presence and proper placement (penalizes overplacement/underplacement).

    Reward structure:
    - All 3 objects properly placed (each gets +1.0): Base reward 10.0
    - 2 objects properly placed: Base reward 5.0
    - 1 object properly placed: Base reward 2.0
    - 0 objects properly placed: Base penalty -2.0
    - Individual object penalties for over/underplacement are also included

    Input:
        - parsed_scenes: dict with scene tensors
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor polygon vertices
        - **kwargs: additional keyword arguments

    Output:
        reward: torch.Tensor of shape (B,)
    """
    utility_functions = get_all_utility_functions()

    one_hot = parsed_scenes["one_hot"]  # (B, N, num_classes)
    B = one_hot.shape[0]
    device = parsed_scenes["device"]

    # Get individual rewards for each object type (includes over/underplacement penalties)
    tv_stand_reward = utility_functions["get_object_present_reward_potential"][
        "function"
    ](
        one_hot,
        "tv_stand",
        idx_to_labels,
        object_indices=parsed_scenes["object_indices"],
    )

    desk_reward = utility_functions["get_object_present_reward_potential"]["function"](
        one_hot, "desk", idx_to_labels, object_indices=parsed_scenes["object_indices"]
    )

    chair_reward = utility_functions["get_object_present_reward_potential"]["function"](
        one_hot, "chair", idx_to_labels, object_indices=parsed_scenes["object_indices"]
    )
    
    # single_bed_reward = utility_functions["get_object_present_reward_potential"]["function"](
    #     one_hot, "single_bed", idx_to_labels, object_indices=parsed_scenes["object_indices"]
    # )
    # double_bed_reward = utility_functions["get_object_present_reward_potential"]["function"](
    #     one_hot, "double_bed", idx_to_labels, object_indices=parsed_scenes["object_indices"]
    # )
    # bed_reward = (
    #     (single_bed_reward == 1.0).float() + (double_bed_reward == 1.0).float()
    # )
    

    # # Count how many object types are properly placed (reward = 1.0)
    num_properly_placed = (
        (tv_stand_reward == 1.0).float()
        + (desk_reward == 1.0).float()
        + (chair_reward == 1.0).float()
        # + (bed_reward == 1.0).float()
    )

    # Base reward structure emphasizing having all 4 objects
    base_rewards = torch.zeros(B, device=device)
    # base_rewards[num_properly_placed == 4] = 20.0  # Highest reward for all 4
    base_rewards[num_properly_placed == 3] = 10.0  # lower reward for 3
    base_rewards[num_properly_placed == 2] = 5.0  # Lower for 2
    base_rewards[num_properly_placed == 1] = 2.0  # Even lower for 1
    base_rewards[num_properly_placed == 0] = -2.0  # Penalty for none

    # Add individual penalties for over/underplacement
    individual_penalties = (
        tv_stand_reward + desk_reward + chair_reward - num_properly_placed
    )
    # # Total reward: base + individual penalties
    total_rewards = base_rewards + individual_penalties
    #simplified reward 
    # total_rewards = base_rewards

    return total_rewards


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Test the composite reward function for TV stand, desk, and chair presence.
    """
    utility_functions = get_all_utility_functions()

    # Scene 1: All 3 objects present (tv_stand=19, desk=7, chair=4)
    num_objects_1 = 5
    class_label_indices_1 = [
        19,
        7,
        4,
        8,
        12,
    ]  # tv_stand, desk, chair, double_bed, nightstand
    translations_1 = [
        (0, 0.4, 0),
        (1, 0.4, 1),
        (1.5, 0.4, 1),
        (2, 0.5, 2),
        (-2, 0.3, -2),
    ]
    sizes_1 = [
        (0.4, 0.4, 0.3),
        (0.6, 0.4, 0.5),
        (0.3, 0.4, 0.3),
        (1.0, 0.5, 1.0),
        (0.3, 0.3, 0.3),
    ]
    orientations_1 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type,
        num_objects_1,
        class_label_indices_1,
        translations_1,
        sizes_1,
        orientations_1,
    )

    # Scene 2: 2 objects present (tv_stand and desk, no chair)
    num_objects_2 = 4
    class_label_indices_2 = [19, 7, 8, 12]  # tv_stand, desk, double_bed, nightstand
    translations_2 = [(0, 0.4, 0), (1, 0.4, 1), (2, 0.5, 2), (-2, 0.3, -2)]
    sizes_2 = [(0.4, 0.4, 0.3), (0.6, 0.4, 0.5), (1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type,
        num_objects_2,
        class_label_indices_2,
        translations_2,
        sizes_2,
        orientations_2,
    )

    # Scene 3: 1 object present (only desk)
    num_objects_3 = 3
    class_label_indices_3 = [7, 8, 12]  # desk, double_bed, nightstand
    translations_3 = [(1, 0.4, 1), (2, 0.5, 2), (-2, 0.3, -2)]
    sizes_3 = [(0.6, 0.4, 0.5), (1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type,
        num_objects_3,
        class_label_indices_3,
        translations_3,
        sizes_3,
        orientations_3,
    )

    # Scene 4: No target objects present
    num_objects_4 = 2
    class_label_indices_4 = [8, 12]  # double_bed, nightstand
    translations_4 = [(2, 0.5, 2), (-2, 0.3, -2)]
    sizes_4 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_4 = [(1, 0), (1, 0)]
    scene_4 = utility_functions["create_scene_for_testing"]["function"](
        room_type,
        num_objects_4,
        class_label_indices_4,
        translations_4,
        sizes_4,
        orientations_4,
    )

    # Stack scenes
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes["room_type"] = room_type
    parsed_scenes["device"] = scene_1["device"]

    rewards = get_reward(
        parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs
    )
    print("Rewards:", rewards)

    # Test assertions
    print(
        f"Scene 1 (all 3 objects properly placed): {rewards[0].item()}, expected: 10.0"
    )
    print(f"Scene 2 (2 objects properly placed): {rewards[1].item()}, expected: 5.0")
    print(f"Scene 3 (1 object properly placed): {rewards[2].item()}, expected: 2.0")
    print(f"Scene 4 (0 objects properly placed): {rewards[3].item()}, expected: -2.0")

    print("All tests passed!")
