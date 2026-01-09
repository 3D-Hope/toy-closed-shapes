import torch

idx_to_labels_bedroom = {
    0: "armchair",
    1: "bookshelf",
    2: "cabinet",
    3: "ceiling_lamp",
    4: "chair",
    5: "children_cabinet",
    6: "coffee_table",
    7: "desk",
    8: "double_bed",
    9: "dressing_chair",
    10: "dressing_table",
    11: "kids_bed",
    12: "nightstand",
    13: "pendant_lamp",
    14: "shelf",
    15: "single_bed",
    16: "sofa",
    17: "stool",
    18: "table",
    19: "tv_stand",
    20: "wardrobe",
}

idx_to_labels_livingroom = {
    0: "armchair",
    1: "bookshelf",
    2: "cabinet",
    3: "ceiling_lamp",
    4: "chaise_longue_sofa",
    5: "chinese_chair",
    6: "coffee_table",
    7: "console_table",
    8: "corner_side_table",
    9: "desk",
    10: "dining_chair",
    11: "dining_table",
    12: "l_shaped_sofa",
    13: "lazy_sofa",
    14: "lounge_chair",
    15: "loveseat_sofa",
    16: "multi_seat_sofa",
    17: "pendant_lamp",
    18: "round_end_table",
    19: "shelf",
    20: "stool",
    21: "tv_stand",
    22: "wardrobe",
    23: "wine_cabinet",
}

idx_to_labels = {
    "bedroom": idx_to_labels_bedroom,
    "livingroom": idx_to_labels_livingroom,
}


ceiling_objects = ["ceiling_lamp", "pendant_lamp"]


def descale_to_origin(x, minimum, maximum):
    """
    x shape : BxNx3
    minimum, maximum shape: 3
    """
    x = (x + 1) / 2
    x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
    return x


def descale_pos(
    positions, pos_min=None, pos_max=None, device="cuda", room_type="livingroom"
):
    """
    Descale positions to original coordinates.

    Args:
        positions: Tensor of shape BxNx3
        pos_min: Minimum position values (optional)
        pos_max: Maximum position values (optional)
        device: Device for tensors

    Returns:
        Descaled positions
    """
    if pos_min is None:
        if room_type == "bedroom":
            pos_min = torch.tensor([-2.7625005, 0.045, -2.75275], device=device)
        elif room_type == "livingroom":  # TODO: Update these values
            pos_min = torch.tensor(
                [-5.672918693230125, 0.0375, -5.716401580065309], device=device
            )
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    if pos_max is None:
        if room_type == "bedroom":
            pos_max = torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device)
        elif room_type == "livingroom":
            pos_max = torch.tensor(
                [5.09667921844729, 3.3577405149437496, 5.4048500000000015],
                device=device,
            )
        else:
            raise ValueError(f"Unknown room type: {room_type}")

    return descale_to_origin(positions, pos_min, pos_max)


def descale_size(
    sizes, size_min=None, size_max=None, device="cuda", room_type="livingroom"
):
    """
    Descale sizes to original dimensions.

    IMPORTANT: The returned sizes are HALF-EXTENTS (sx/2, sy/2, sz/2), not full dimensions.
    This means:
    - For a box centered at (x, y, z) with returned size (sx, sy, sz):
      - The box extends from (x-sx, y-sy, z-sz) to (x+sx, y+sy, z+sz)
      - Full dimensions would be (2*sx, 2*sy, 2*sz)
    - When computing bounding boxes: use size directly, DO NOT divide by 2 again
    - When computing object bottom: y_min = y_center - y_size (not y_center - y_size/2)

    Args:
        sizes: Tensor of shape BxNx3 (normalized)
        size_min: Minimum size values (optional)
        size_max: Maximum size values (optional)
        device: Device for tensors

    Returns:
        Descaled sizes (HALF-EXTENTS)
    """
    if size_min is None:
        if room_type == "bedroom":
            size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
        elif room_type == "livingroom":  # TODO: Update these values
            size_min = torch.tensor(
                [
                    0.03998999999999997,
                    0.020000020334800084,
                    0.0328434999999998,
                ],
                device=device,
            )
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    if size_max is None:
        if room_type == "bedroom":
            size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)
        elif room_type == "livingroom":
            size_max = torch.tensor(
                [2.3802699999999994, 1.7700649999999998, 1.3224289999999996],
                device=device,
            )
        else:
            raise ValueError(f"Unknown room type: {room_type}")

    return descale_to_origin(sizes, size_min, size_max)


def parse_and_descale_scenes(
    scenes, num_classes=22, parse_only=False, room_type="livingroom"
):
    """
    Parse scene tensor and descale positions/sizes to world coordinates.

    IMPORTANT: Sizes are HALF-EXTENTS (sx/2, sy/2, sz/2), not full dimensions!
    - For bounding boxes: min = center - size, max = center + size
    - DO NOT divide sizes by 2 again in reward calculations

    Args:
        scenes: Tensor of shape (B, N, 30)
        num_classes: Number of object classes (default: 22)

    Returns:
        dict with keys:
            - one_hot: (B, N, num_classes)
            - positions: (B, N, 3) - world coordinates
            - sizes: (B, N, 3) - world coordinates (HALF-EXTENTS!)
            - orientations: (B, N, 2) - [cos_theta, sin_theta]
            - object_indices: (B, N) - argmax of one_hot
            - is_empty: (B, N) - boolean mask for empty slots
            - device: device of input tensor
    """
    device = scenes.device

    # Parse scene representation
    positions_normalized = scenes[:, :, 0:3]
    sizes_normalized = scenes[:, :, 3:6]
    orientations = scenes[:, :, 6:8]  # [cos_theta, sin_theta]
    one_hot = scenes[:, :, 8 : 8 + num_classes]

    # Descale to world coordinates
    if not parse_only:
        positions = descale_pos(
            positions_normalized, device=device, room_type=room_type
        )
        sizes = descale_size(sizes_normalized, device=device, room_type=room_type)
    else:
        positions = positions_normalized
        sizes = sizes_normalized

    # Get object categories
    object_indices = torch.argmax(one_hot, dim=-1)

    # Identify empty slots
    empty_class_idx = num_classes - 1
    is_empty = object_indices == empty_class_idx

    return {
        "one_hot": one_hot,
        "positions": positions,
        "sizes": sizes,
        "orientations": orientations,
        "object_indices": object_indices,
        "is_empty": is_empty,
        "device": device,
    }


def get_all_universal_reward_functions():
    """
    Returns a dictionary of all universal (hand-designed) reward functions.

    This is a centralized place to define which universal rewards exist,
    so they don't need to be listed in multiple places.

    Returns:
        Dict mapping reward names to reward functions
    """
    # Import here to avoid circular imports
    from universal_constraint_rewards.accessibility_reward import (
        compute_accessibility_reward,
    )
    from universal_constraint_rewards.axis_alignment_reward import (
        compute_axis_alignment_reward,
    )
    from universal_constraint_rewards.furniture_against_wall_reward import (
        compute_wall_proximity_reward,
    )
    from universal_constraint_rewards.gravity_following_reward import (
        compute_gravity_following_reward,
    )
    from universal_constraint_rewards.must_have_furniture_reward import (
        compute_must_have_furniture_reward,
    )
    from universal_constraint_rewards.night_tables_on_head_side_reward import (
        compute_nightstand_placement_reward,
    )
    from universal_constraint_rewards.non_penetration_reward import (
        compute_non_penetration_reward,
    )
    from universal_constraint_rewards.not_out_of_bound_reward import (
        compute_boundary_violation_reward,
    )

    # from universal_constraint_rewards.object_count_reward import (
    #     compute_object_count_reward,
    # )

    return {
        # "must_have_furniture": compute_must_have_furniture_reward if ,
        # "object_count": compute_object_count_reward,
        "not_out_of_bound": compute_boundary_violation_reward,
        "accessibility": compute_accessibility_reward,
        "non_penetration": compute_non_penetration_reward,
        # "gravity_following": compute_gravity_following_reward,
        # "night_tables_on_head_side": compute_nightstand_placement_reward,
        # "axis_alignment": compute_axis_alignment_reward,
        # "furniture_against_wall": compute_wall_proximity_reward,
    }


def get_universal_reward(
    parsed_scenes,
    reward_normalizer,
    num_classes=22,
    importance_weights=None,
    get_reward_functions=None,
    **kwargs,
):
    """
    Entry point for computing universal reward from multiple reward functions.

    This function computes predefined universal reward functions and combines them.

    Args:
        parsed_scenes: Dict returned by parse_and_descale_scenes()
        num_classes: Number of object classes (default: 22)
        importance_weights: Dict mapping reward names to importance weights
        reward_normalizer:  normalizer to scale rewards to [0, 1] range
        get_reward_functions: Dict of reward functions to compute (if None, uses defaults)
        **kwargs: Additional arguments passed to individual reward functions

    Returns:
        total_reward: Combined reward normalized by sum of importance weights
        reward_components: Dict with individual reward values for analysis
    """
    # print("[Ashok] importance_weights in universal reward:", importance_weights)
    # importance_weights = importance_weights["importance_weights"]
    # rewards = {}
    # print(f"[Ashok] Computing universal rewards kwargs has keys: {list(kwargs.keys())}")
    # Define default universal reward functions if not provided
    # if get_reward_functions is None:
    get_reward_functions = get_all_universal_reward_functions()
    rewards_sum = 0.0
    reward_components = {}

    # Compute rewards for each function
    for key, value in get_reward_functions.items():
        # if key not in importance_weights or importance_weights[key] == 0:
        #     continue  # Skip rewards with zero importance weight
        reward = value(parsed_scenes, **kwargs)
        # rewards[key] = reward
        reward_components[key] = reward
        rewards_sum += reward
        print(f"[Ashok] Raw reward for {key}: {reward}")
        # rewards_sum += importance_weights[key] * reward
    # Normalize rewards if normalizer is provided
    # reward_normalizer = None
    # if reward_normalizer is not None:
    #     for key, value in rewards.items():
    #         reward_components[
    #             key
    #         ] = value  # viz raw values to avoid weird normalized values in curves
    #         rewards[key] = reward_normalizer.normalize(key, torch.tensor(value))
    #         print(f"[Ashok] Normalized reward for {key}: {rewards[key]}")
    # else:
    #     for key, value in rewards.items():
    #         reward_components[key] = value
    # rewards_sum = 0

    # for key, value in rewards.items():
    #     importance = importance_weights[key]
    #     rewards_sum += importance * value

    return rewards_sum, reward_components
