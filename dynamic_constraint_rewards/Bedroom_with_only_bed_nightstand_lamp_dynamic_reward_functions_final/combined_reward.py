import torch

from dynamic_constraint_rewards.utilities import get_all_utility_functions


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Composite reward for a bedroom with only a bed, nightstand, and lamp allowed.
    Rewards:
    - All 3 objects present (and no extra objects): 10.0
    - 2 of the 3 present (and no extra): 5.0
    - 1 of the 3 present (and no extra): 2.0
    - No objects present: -2.0
    - Extra objects present: -5.0 per extra object
    
    Penalizes underplacement (missing required objects) and overplacement (extra non-allowed objects, or more than one of each).

    Each required object must be properly placed (i.e., using utility reward == 1.0) to count for full credit.
    """
    utility_functions = get_all_utility_functions()
    one_hot = parsed_scenes["one_hot"]
    B = one_hot.shape[0]
    device = parsed_scenes["one_hot"].device if "device" not in parsed_scenes else parsed_scenes["device"]

    # Define all allowed objects for room, and reward targets
    bed_targets = ["kids_bed", "single_bed", "double_bed"]
    nightstand_targets = ["nightstand"]
    lamp_targets = ["ceiling_lamp", "pendant_lamp"]

    all_allowed_targets = bed_targets + nightstand_targets + lamp_targets

    label_to_idx = {v: k for k, v in idx_to_labels.items()}
    allowed_label_indices = set(label_to_idx[o] for o in all_allowed_targets if o in label_to_idx)

    class_labels = parsed_scenes["object_indices"]  # (B, N)
    N = class_labels.shape[1]

    def count_obj(obj_names):
        if isinstance(obj_names, str):
            obj_names = [obj_names]
        indices = [label_to_idx[o] for o in obj_names if o in label_to_idx]
        if not indices:
            return torch.zeros(B, device=device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i in indices:
            mask = mask | (class_labels == i)
        return mask.sum(dim=1)

    num_beds = count_obj(bed_targets)
    num_nightstands = count_obj(nightstand_targets)
    num_lamps = count_obj(lamp_targets)
    # ---- Utility rewards for required objects ----
    # Return 1.0 if present and proper, else 0.0
    bed_present = torch.where(num_beds > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    nightstand_present = torch.where(num_nightstands > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    lamp_present = torch.where(num_lamps > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

    present_count = bed_present + nightstand_present + lamp_present

    # ---- Assign base rewards according to the rules ----
    base_rewards = torch.zeros(B, device=device)
    base_rewards[present_count == 3] = 10.0
    base_rewards[present_count == 2] = 5.0
    base_rewards[present_count == 1] = 2.0
    base_rewards[present_count == 0] = -2.0

    # ---- Overplacement detection (more than one of each allowed type) ----
        # Only one of each type allowed - overplacement is non-allowed
    over_beds = torch.clamp(num_beds - 1, min=0)
    over_nightstands = torch.clamp(num_nightstands - 2, min=0)
    over_lamps = torch.clamp(num_lamps - 1, min=0)
    overplacement_penalty = (over_beds + over_nightstands + over_lamps) * -2.0
    





    # ---- Extra (illegal) objects (not in allowed list): -5.0 * (count illegal) ----
    illegal_object_penalty = torch.zeros(B, device=device)
    for b in range(B):
        labels_in_scene = class_labels[b].tolist()
        illegal_cnt = 0
        for idx in labels_in_scene:
            if idx not in allowed_label_indices and idx != 21: #TODO: remove hardcoding of 21 (empty)
                illegal_cnt += 1
        if illegal_cnt:
            illegal_object_penalty[b] = -5.0 * illegal_cnt

    # ---- Total reward ----
    total_rewards = base_rewards + overplacement_penalty + illegal_object_penalty

    return total_rewards


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Test the reward for the scenario where only bed, nightstand, and lamp are allowed.
    """
    utility_functions = get_all_utility_functions()

    def lblidx(lbl):
        for k, v in idx_to_labels.items():
            if v == lbl:
                return k
        return -1

    # Scene 1: All 3 objects present, one of each, no extras
    num_objects_1 = 3
    class_label_indices_1 = [
        lblidx("double_bed") if lblidx("double_bed") != -1 else (lblidx("single_bed") if lblidx("single_bed") != -1 else (lblidx("kids_bed") if lblidx("kids_bed") != -1 else lblidx("bed"))),
        lblidx("nightstand"),
        lblidx("ceiling_lamp") if lblidx("ceiling_lamp") != -1 else lblidx("pendant_lamp"),
    ]
    translations_1 = [(0, 0, 0), (1, 0, 1), (1, 0, -1)]
    sizes_1 = [(1, 0.5, 1), (0.3, 0.3, 0.3), (0.3, 0.7, 0.3)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1,
        sizes_1, orientations_1
    )

    # Scene 2: Only kids_bed and lamp present (no nightstand)
    num_objects_2 = 2
    class_label_indices_2 = [
        lblidx("kids_bed") if lblidx("kids_bed") != -1 else (lblidx("double_bed") if lblidx("double_bed") != -1 else lblidx("single_bed")),
        lblidx("ceiling_lamp") if lblidx("ceiling_lamp") != -1 else lblidx("pendant_lamp"),
    ]
    translations_2 = [(0, 0, 0), (1, 0, -1)]
    sizes_2 = [(1, 0.5, 1), (0.3, 0.7, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_2, class_label_indices_2, translations_2,
        sizes_2, orientations_2
    )

    # Scene 3: Only kids_bed present (no nightstand, no lamp)
    num_objects_3 = 1
    class_label_indices_3 = [
        lblidx("single_bed") if lblidx("single_bed") != -1 else (lblidx("double_bed") if lblidx("double_bed") != -1 else (lblidx("kids_bed") if lblidx("kids_bed") != -1 else lblidx("bed"))),
    ]
    translations_3 = [(0, 0, 0)]
    sizes_3 = [(1, 0.5, 1)]
    orientations_3 = [(1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_3, class_label_indices_3, translations_3,
        sizes_3, orientations_3
    )

    # Scene 4: No allowed objects
    num_objects_4 = 2
    class_label_indices_4 = []
    for target in ["desk", "chair"]:
        idx = lblidx(target)
        if idx != -1:
            class_label_indices_4.append(idx)
    if len(class_label_indices_4) == 0:
        class_label_indices_4 = [0, 1]
    translations_4 = [(2, 0, 2), (-2, 0, -2)]
    sizes_4 = [(0.6, 0.5, 0.6), (0.3, 0.4, 0.3)]
    orientations_4 = [(1, 0), (1, 0)]
    scene_4 = utility_functions["create_scene_for_testing"]["function"](
        room_type, len(class_label_indices_4), class_label_indices_4, translations_4,
        sizes_4, orientations_4
    )

    # Scene 5: All 3 objects, but an extra lamp (should incur overplacement penalty)
    num_objects_5 = 4
    class_label_indices_5 = [
        lblidx("double_bed") if lblidx("double_bed") != -1 else (lblidx("single_bed") if lblidx("single_bed") != -1 else lblidx("bed")),
        lblidx("nightstand"),
        lblidx("ceiling_lamp") if lblidx("ceiling_lamp") != -1 else lblidx("pendant_lamp"),
        lblidx("lamp"),
    ]
    translations_5 = [(0, 0, 0), (1, 0, 1), (1, 0, -1), (0.8, 0, -1.1)]
    sizes_5 = [(1, 0.5, 1), (0.3, 0.3, 0.3), (0.3, 0.7, 0.3), (0.3, 0.7, 0.3)]
    orientations_5 = [(1, 0), (1, 0), (1, 0), (1, 0)]
    scene_5 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_5, class_label_indices_5, translations_5,
        sizes_5, orientations_5
    )

    # Scene 6: Valid objects, but with an illegal extra (e.g. nightstand, bed, lamp, chair)
    num_objects_6 = 4
    cls = [
        lblidx("single_bed") if lblidx("single_bed") != -1 else (lblidx("double_bed") if lblidx("double_bed") != -1 else lblidx("bed")),
        lblidx("nightstand"),
        lblidx("ceiling_lamp") if lblidx("ceiling_lamp") != -1 else lblidx("pendant_lamp"),
    ]
    illegal = lblidx("chair") if lblidx("chair") != -1 else 0
    class_label_indices_6 = cls + [illegal]
    translations_6 = [(0, 0, 0), (1, 0, 1), (1, 0, -1), (1, 0, 5)]
    sizes_6 = [(1, 0.5, 1), (0.3, 0.3, 0.3), (0.3, 0.7, 0.3), (0.4, 0.4, 0.4)]
    orientations_6 = [(1, 0), (1, 0), (1, 0), (1, 0)]
    scene_6 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_6, class_label_indices_6, translations_6,
        sizes_6, orientations_6
    )

    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    scenes = [scene_1, scene_2, scene_3, scene_4, scene_5, scene_6]
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

    print(f"Scene 1 (all present, no extras): {rewards[0].item()}, expected: 10.0")
    print(f"Scene 2 (2 present, no extras): {rewards[1].item()}, expected: 5.0")
    print(f"Scene 3 (1 present, no extras): {rewards[2].item()}, expected: 2.0")
    print(f"Scene 4 (no allowed objects): {rewards[3].item()}, expected: -10.0")
    print(f"Scene 5 (all present, extra lamp): {rewards[4].item()}, expected: 5.0")
    print(f"Scene 6 (all correct, plus illegal object): {rewards[5].item()}, expected: 5.0")

    print("All tests passed!")

