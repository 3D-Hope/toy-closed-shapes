import torch

from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Reward function for bedrooms with a wardrobe and a stool.

    - Both a wardrobe and a stool must be present.
    - The distance from the stool center to the nearest wardrobe boundary must be less than 0.5m (inclusive).
    - Reward:
        - Both present and distance below threshold: 10.0
        - Both present, but too far: 2.0
        - Only one present: 0.0
        - Neither present: -2.0
        - Overplacement penalty: -2.0 per extra wardrobe or stool beyond the first
    """
    # utility_functions = get_all_utility_functions()
    
    one_hot = parsed_scenes["one_hot"]  # (B, N, num_classes)
    positions = parsed_scenes["positions"]  # (B, N, 3)
    sizes = parsed_scenes["sizes"]      # (B, N, 3)
    idx_to_label = idx_to_labels

    B, N, num_classes = one_hot.shape[0], one_hot.shape[1], one_hot.shape[2]
    device = one_hot.device

    # Find class indices
    wardrobe_names = ["wardrobe"]
    stool_names = ["stool"]

    label_to_idx = {v: k for k, v in idx_to_label.items()}

    wardrobe_indices = [label_to_idx[name] for name in wardrobe_names if name in label_to_idx]
    stool_indices = [label_to_idx[name] for name in stool_names if name in label_to_idx]

    wardrobe_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for idx in wardrobe_indices:
        wardrobe_mask |= (parsed_scenes["object_indices"] == idx)
    stool_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for idx in stool_indices:
        stool_mask |= (parsed_scenes["object_indices"] == idx)

    has_wardrobe = wardrobe_mask.any(dim=1)  # (B,)
    has_stool = stool_mask.any(dim=1)        # (B,)
    both_present = has_wardrobe & has_stool

    num_wardrobes = wardrobe_mask.sum(dim=1)
    num_stools = stool_mask.sum(dim=1)

    # Overplacement penalty (expected: 1 wardrobe, 1 stool)
    over_wardrobes = torch.clamp(num_wardrobes - 1, min=0)
    over_stools = torch.clamp(num_stools - 1, min=0)
    overplacement_penalty = (over_wardrobes + over_stools) * -2.0

    # Default reward is -2
    rewards = torch.full((B,), -2.0, device=device)

    # For scenes with both present, check the min dist from stool center to any wardrobe boundary
    for b in range(B):
        if both_present[b]:
            # Get all wardrobe(s) in scene
            wardrobe_indices_b = torch.where(wardrobe_mask[b])[0]
            stool_indices_b = torch.where(stool_mask[b])[0]

            # There could be multiple, but per reward function just need min
            # For all stools and all wardrobes: compute min distance from stool center to wardrobe boundary
            min_dist = torch.tensor(float('inf'), device=device)
            for si in stool_indices_b:
                stool_pos = positions[b, si]  # (3,)
                # Only X and Z matter (ignore Y)
                stool_xz = stool_pos[[0, 2]]
                for wi in wardrobe_indices_b:
                    wd_pos = positions[b, wi]  # (3,)
                    wd_size = sizes[b, wi]     # (3,)
                    wd_center = wd_pos[[0, 2]]
                    wd_hw = wd_size[[0, 2]]

                    wd_min = wd_center - wd_hw
                    wd_max = wd_center + wd_hw
                    # Clamp stool_xz to wardrobe box, then L2 dist from stool to this closest boundary point
                    closest = torch.clamp(stool_xz, wd_min, wd_max)
                    dist = torch.norm(stool_xz - closest, p=2)
                    if dist < min_dist:
                        min_dist = dist

            # Assign reward
            if min_dist <= 0.5:
                rewards[b] = 10.0
            else:
                rewards[b] = 2.0

    # If only 1 present (exclusive or), get 0.0
    only_one_present = (has_wardrobe ^ has_stool)
    rewards[only_one_present] = 0.0

    # Add overplacement penalty
    rewards = rewards + overplacement_penalty

    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Test the wardrobe-stool reward function.
    """
    utility_functions = get_all_utility_functions()

    def lblidx(lbl):
        for k, v in idx_to_labels.items():
            if v == lbl:
                return k
        return -1

    # Scene 1: Both present, close together (should reward 10.0)
    wardrobe_idx = lblidx("wardrobe") if lblidx("wardrobe") != -1 else (lblidx("clothes_wardrobe") if lblidx("clothes_wardrobe") != -1 else lblidx("closet"))
    stool_idx = lblidx("stool")
    num_objects_1 = 2
    class_label_indices_1 = [wardrobe_idx, stool_idx]
    translations_1 = [(0, 0, 0), (0.45, 0, 0)]  # within 0.5m boundary
    sizes_1 = [(1, 2, 0.6), (0.3, 0.5, 0.3)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1,
        sizes_1, orientations_1
    )

    # Scene 2: Both present, too far (should reward 2.0)
    translations_2 = [(0, 0, 0), (2.0, 0, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_2,
        sizes_1, orientations_1
    )

    # Scene 3: Only wardrobe present (should reward 0.0)
    num_objects_3 = 1
    class_label_indices_3 = [wardrobe_idx]
    translations_3 = [(0, 0, 0)]
    sizes_3 = [(1, 2, 0.6)]
    orientations_3 = [(1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_3, class_label_indices_3, translations_3,
        sizes_3, orientations_3
    )

    # Scene 4: Only stool present (should reward 0.0)
    num_objects_4 = 1
    class_label_indices_4 = [stool_idx]
    translations_4 = [(1, 0, 1)]
    sizes_4 = [(0.3, 0.5, 0.3)]
    orientations_4 = [(1, 0)]
    scene_4 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_4, class_label_indices_4, translations_4,
        sizes_4, orientations_4
    )

    # Scene 5: None present (should reward -2.0)
    num_objects_5 = 0
    class_label_indices_5 = []
    translations_5 = []
    sizes_5 = []
    orientations_5 = []
    scene_5 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_5, class_label_indices_5, translations_5,
        sizes_5, orientations_5
    )

    # Scene 6: Two wardrobes and one stool, arrangements close (should reward 8.0: 10.0 - 2.0)
    num_objects_6 = 3
    class_label_indices_6 = [wardrobe_idx, wardrobe_idx, stool_idx]
    translations_6 = [(0, 0, 0), (2.0, 0, 0), (0.49, 0, 0)]
    sizes_6 = [(1, 2, 0.6), (1, 2, 0.6), (0.3, 0.5, 0.3)]
    orientations_6 = [(1, 0), (1, 0), (1, 0)]
    scene_6 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_6, class_label_indices_6, translations_6,
        sizes_6, orientations_6
    )

    # Scene 7: One wardrobe, two stools, close, (reward 8.0: 10 - 2)
    num_objects_7 = 3
    class_label_indices_7 = [wardrobe_idx, stool_idx, stool_idx]
    translations_7 = [(0, 0, 0), (0.48, 0, 0), (0.45, 0, 0.2)]
    sizes_7 = [(1, 2, 0.6), (0.3, 0.5, 0.3), (0.3, 0.5, 0.3)]
    orientations_7 = [(1, 0), (1, 0), (1, 0)]
    scene_7 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_7, class_label_indices_7, translations_7,
        sizes_7, orientations_7
    )

    # Combine
    scenes = [scene_1, scene_2, scene_3, scene_4, scene_5, scene_6, scene_7]
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes_all = {
        k: torch.cat([scene[k] for scene in scenes], dim=0)
        for k in tensor_keys
    }
    parsed_scenes_all["room_type"] = room_type
    parsed_scenes_all["device"] = scene_1["device"]

    rewards = get_reward(
        parsed_scenes_all, idx_to_labels, room_type, floor_polygons, **kwargs
    )
    print("Rewards:", rewards.tolist())
    print(f"Scene 1 (both present, close): {rewards[0].item()}, expected: 10.0")
    print(f"Scene 2 (both present, too far): {rewards[1].item()}, expected: 2.0")
    print(f"Scene 3 (only wardrobe): {rewards[2].item()}, expected: 0.0")
    print(f"Scene 4 (only stool): {rewards[3].item()}, expected: 0.0")
    print(f"Scene 5 (neither): {rewards[4].item()}, expected: -2.0")
    print(f"Scene 6 (two wardrobes, one stool, close): {rewards[5].item()}, expected: 8.0")
    print(f"Scene 7 (one wardrobe, two stools, close): {rewards[6].item()}, expected: 8.0")
    print("All tests done.")

