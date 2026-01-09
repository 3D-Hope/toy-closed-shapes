
        User Prompt: A bedroom with ceiling lamp above each corner of the bed.
        Initial Constraints: {'constraints': [{'id': 'C1', 'name': 'ceiling_lamp_presence', 'description': 'Verifies that at least one ceiling lamp exists in the scene'}, {'id': 'C2', 'name': 'bed_presence', 'description': 'Verifies that at least one bed (double_bed, single_bed, or kids_bed) exists in the scene'}, {'id': 'C3', 'name': 'ceiling_lamps_above_bed_corners', 'description': 'Checks that ceiling lamps are positioned directly above the four corners of the bed in the XZ plane, with appropriate vertical positioning near the ceiling height'}, {'id': 'C4', 'name': 'four_ceiling_lamps_required', 'description': 'Verifies that exactly four ceiling lamps are present (one for each corner of the bed)'}]}
        Initial Reward Functions: {'rewards': [{'id': 'R1', 'constraint_id': 'C1', 'name': 'ceiling_lamp_presence', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward for having at least one ceiling lamp in the scene.\n    Returns 0 if at least one ceiling lamp exists, -1 otherwise.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]\n    \n    B = parsed_scenes[\'positions\'].shape[0]\n    rewards = torch.zeros(B, device=parsed_scenes[\'device\'])\n    \n    for i in range(B):\n        scene_one_hot = parsed_scenes[\'one_hot\'][i:i+1]\n        count = get_object_count(scene_one_hot, "ceiling_lamp", idx_to_labels)\n        if count < 1:\n            rewards[i] = -1.0\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    # Scene 1: Has 1 ceiling lamp (satisfies)\n    scene_1 = create_scene(\n        room_type=room_type,\n        num_objects=2,\n        class_label_indices=[3, 8],  # ceiling_lamp, double_bed\n        translations=[(0, 2.8, 0), (1, 0.5, 1)],\n        sizes=[(0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],\n        orientations=[(1, 0), (1, 0)]\n    )\n    \n    # Scene 2: Has 4 ceiling lamps (satisfies)\n    scene_2 = create_scene(\n        room_type=room_type,\n        num_objects=5,\n        class_label_indices=[3, 3, 3, 3, 8],  # 4 ceiling_lamps, double_bed\n        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 2.8, 1), (3, 2.8, 1), (2, 0.5, 2)],\n        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],\n        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]\n    )\n    \n    # Scene 3: No ceiling lamp (fails)\n    scene_3 = create_scene(\n        room_type=room_type,\n        num_objects=2,\n        class_label_indices=[8, 12],  # double_bed, nightstand\n        translations=[(1, 0.5, 1), (2, 0.3, 2)],\n        sizes=[(1.0, 0.4, 1.5), (0.4, 0.3, 0.4)],\n        orientations=[(1, 0), (1, 0)]\n    )\n    \n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print("Expected: [0.0, 0.0, -1.0]")\n    assert rewards.shape[0] == 3\n    assert rewards[0] == 0.0, f"Scene 1 should have reward 0.0, got {rewards[0]}"\n    assert rewards[1] == 0.0, f"Scene 2 should have reward 0.0, got {rewards[1]}"\n    assert rewards[2] == -1.0, f"Scene 3 should have reward -1.0, got {rewards[2]}"', 'success_threshold': 0.0}, {'id': 'R2', 'constraint_id': 'C2', 'name': 'bed_presence', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward for having at least one bed (double_bed, single_bed, or kids_bed) in the scene.\n    Returns 0 if at least one bed exists, -1 otherwise.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]\n    \n    B = parsed_scenes[\'positions\'].shape[0]\n    rewards = torch.zeros(B, device=parsed_scenes[\'device\'])\n    \n    bed_types = ["double_bed", "single_bed", "kids_bed"]\n    \n    for i in range(B):\n        scene_one_hot = parsed_scenes[\'one_hot\'][i:i+1]\n        total_beds = 0\n        for bed_type in bed_types:\n            total_beds += get_object_count(scene_one_hot, bed_type, idx_to_labels)\n        \n        if total_beds < 1:\n            rewards[i] = -1.0\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    # Scene 1: Has double_bed (satisfies)\n    scene_1 = create_scene(\n        room_type=room_type,\n        num_objects=2,\n        class_label_indices=[8, 12],  # double_bed, nightstand\n        translations=[(1, 0.5, 1), (2, 0.3, 2)],\n        sizes=[(1.0, 0.4, 1.5), (0.4, 0.3, 0.4)],\n        orientations=[(1, 0), (1, 0)]\n    )\n    \n    # Scene 2: Has single_bed (satisfies)\n    scene_2 = create_scene(\n        room_type=room_type,\n        num_objects=2,\n        class_label_indices=[15, 12],  # single_bed, nightstand\n        translations=[(1, 0.4, 1), (2, 0.3, 2)],\n        sizes=[(0.6, 0.3, 1.2), (0.4, 0.3, 0.4)],\n        orientations=[(1, 0), (1, 0)]\n    )\n    \n    # Scene 3: No bed (fails)\n    scene_3 = create_scene(\n        room_type=room_type,\n        num_objects=2,\n        class_label_indices=[12, 20],  # nightstand, wardrobe\n        translations=[(2, 0.3, 2), (3, 1.0, 3)],\n        sizes=[(0.4, 0.3, 0.4), (0.8, 1.0, 0.5)],\n        orientations=[(1, 0), (1, 0)]\n    )\n    \n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print("Expected: [0.0, 0.0, -1.0]")\n    assert rewards.shape[0] == 3\n    assert rewards[0] == 0.0, f"Scene 1 should have reward 0.0, got {rewards[0]}"\n    assert rewards[1] == 0.0, f"Scene 2 should have reward 0.0, got {rewards[1]}"\n    assert rewards[2] == -1.0, f"Scene 3 should have reward -1.0, got {rewards[2]}"', 'success_threshold': 0.0}, {'id': 'R3', 'constraint_id': 'C3', 'name': 'ceiling_lamps_above_bed_corners', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward for ceiling lamps being positioned above the four corners of the bed.\n    For each ceiling lamp, calculate the minimum distance to any bed corner.\n    Reward = -sum(min_distances) across all ceiling lamps.\n    Better alignment (smaller distances) gives higher reward (closer to 0).\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]\n    \n    B = parsed_scenes[\'positions\'].shape[0]\n    N = parsed_scenes[\'positions\'].shape[1]\n    rewards = torch.zeros(B, device=parsed_scenes[\'device\'])\n    \n    # Find class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    ceiling_lamp_idx = labels_to_idx["ceiling_lamp"]\n    bed_indices = [labels_to_idx["double_bed"], labels_to_idx["single_bed"], labels_to_idx["kids_bed"]]\n    \n    for i in range(B):\n        positions = parsed_scenes[\'positions\'][i]  # (N, 3)\n        sizes = parsed_scenes[\'sizes\'][i]  # (N, 3)\n        orientations = parsed_scenes[\'orientations\'][i]  # (N, 2)\n        object_indices = parsed_scenes[\'object_indices\'][i]  # (N,)\n        is_empty = parsed_scenes[\'is_empty\'][i]  # (N,)\n        \n        # Find ceiling lamps\n        lamp_mask = (object_indices == ceiling_lamp_idx) & (~is_empty)\n        lamp_positions = positions[lamp_mask]  # (num_lamps, 3)\n        \n        # Find beds\n        bed_mask = torch.zeros_like(is_empty, dtype=torch.bool)\n        for bed_idx in bed_indices:\n            bed_mask |= (object_indices == bed_idx)\n        bed_mask &= (~is_empty)\n        \n        if lamp_mask.sum() == 0 or bed_mask.sum() == 0:\n            rewards[i] = -10.0  # No lamps or beds\n            continue\n        \n        # Get bed info (use first bed if multiple)\n        bed_idx_tensor = torch.where(bed_mask)[0][0]\n        bed_pos = positions[bed_idx_tensor]  # (3,)\n        bed_size = sizes[bed_idx_tensor]  # (3,)\n        bed_orient = orientations[bed_idx_tensor]  # (2,)\n        \n        # Calculate bed corners in XZ plane\n        cos_theta = bed_orient[0]\n        sin_theta = bed_orient[1]\n        \n        # Half-sizes\n        half_x = bed_size[0]\n        half_z = bed_size[2]\n        \n        # Four corners relative to bed center (in local frame)\n        local_corners = torch.tensor([\n            [half_x, half_z],\n            [half_x, -half_z],\n            [-half_x, half_z],\n            [-half_x, -half_z]\n        ], device=parsed_scenes[\'device\'])\n        \n        # Rotate corners to world frame\n        rotation_matrix = torch.tensor([\n            [cos_theta, -sin_theta],\n            [sin_theta, cos_theta]\n        ], device=parsed_scenes[\'device\'])\n        \n        world_corners = torch.matmul(local_corners, rotation_matrix.T)  # (4, 2)\n        world_corners[:, 0] += bed_pos[0]  # Add bed x\n        world_corners[:, 1] += bed_pos[2]  # Add bed z\n        \n        # For each lamp, find minimum distance to any corner (in XZ plane)\n        total_distance = 0.0\n        for lamp_pos in lamp_positions:\n            lamp_xz = torch.tensor([lamp_pos[0], lamp_pos[2]], device=parsed_scenes[\'device\'])\n            distances = torch.norm(world_corners - lamp_xz.unsqueeze(0), dim=1)  # (4,)\n            min_dist = distances.min()\n            total_distance += min_dist\n        \n        # Reward is negative sum of distances (capped at -5 for stability)\n        rewards[i] = -torch.clamp(total_distance, max=5.0)\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    # Scene 1: 4 lamps perfectly at bed corners (satisfies well)\n    # Bed at (0, 0.5, 0) with size (1.0, 0.4, 1.5) means corners at (±1.0, ±1.5)\n    scene_1 = create_scene(\n        room_type=room_type,\n        num_objects=5,\n        class_label_indices=[8, 3, 3, 3, 3],  # double_bed, 4 ceiling_lamps\n        translations=[(0, 0.5, 0), (1.0, 2.8, 1.5), (1.0, 2.8, -1.5), (-1.0, 2.8, 1.5), (-1.0, 2.8, -1.5)],\n        sizes=[(1.0, 0.4, 1.5), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2)],\n        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]\n    )\n    \n    # Scene 2: 4 lamps but offset from corners (satisfies poorly)\n    scene_2 = create_scene(\n        room_type=room_type,\n        num_objects=5,\n        class_label_indices=[8, 3, 3, 3, 3],\n        translations=[(0, 0.5, 0), (2.0, 2.8, 2.0), (2.0, 2.8, -2.0), (-2.0, 2.8, 2.0), (-2.0, 2.8, -2.0)],\n        sizes=[(1.0, 0.4, 1.5), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2)],\n        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]\n    )\n    \n    # Scene 3: No lamps (fails)\n    scene_3 = create_scene(\n        room_type=room_type,\n        num_objects=1,\n        class_label_indices=[8],\n        translations=[(0, 0.5, 0)],\n        sizes=[(1.0, 0.4, 1.5)],\n        orientations=[(1, 0)]\n    )\n    \n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print("Expected: Scene 1 close to 0 (perfect alignment), Scene 2 more negative (misaligned), Scene 3 = -10.0")\n    assert rewards.shape[0] == 3\n    assert rewards[0] > -0.5, f"Scene 1 should have reward close to 0 (got {rewards[0]})" \n    assert rewards[1] < rewards[0], f"Scene 2 should have worse reward than Scene 1 (got {rewards[1]} vs {rewards[0]})"\n    assert rewards[2] == -10.0, f"Scene 3 should have reward -10.0, got {rewards[2]}"', 'success_threshold': -0.5}, {'id': 'R4', 'constraint_id': 'C4', 'name': 'four_ceiling_lamps_required', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward for having exactly 4 ceiling lamps.\n    Returns 0 if exactly 4 ceiling lamps exist, negative penalty otherwise.\n    Penalty = -|count - 4| (linear penalty based on deviation from 4).\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]\n    \n    B = parsed_scenes[\'positions\'].shape[0]\n    rewards = torch.zeros(B, device=parsed_scenes[\'device\'])\n    \n    for i in range(B):\n        scene_one_hot = parsed_scenes[\'one_hot\'][i:i+1]\n        count = get_object_count(scene_one_hot, "ceiling_lamp", idx_to_labels)\n        \n        deviation = abs(count - 4)\n        rewards[i] = -float(deviation)\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    # Scene 1: Exactly 4 ceiling lamps (satisfies)\n    scene_1 = create_scene(\n        room_type=room_type,\n        num_objects=5,\n        class_label_indices=[3, 3, 3, 3, 8],  # 4 ceiling_lamps, double_bed\n        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 2.8, 1), (3, 2.8, 1), (2, 0.5, 2)],\n        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],\n        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]\n    )\n    \n    # Scene 2: 2 ceiling lamps (fails - too few)\n    scene_2 = create_scene(\n        room_type=room_type,\n        num_objects=3,\n        class_label_indices=[3, 3, 8],  # 2 ceiling_lamps, double_bed\n        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 0.5, 2)],\n        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],\n        orientations=[(1, 0), (1, 0), (1, 0)]\n    )\n    \n    # Scene 3: 6 ceiling lamps (fails - too many)\n    scene_3 = create_scene(\n        room_type=room_type,\n        num_objects=7,\n        class_label_indices=[3, 3, 3, 3, 3, 3, 8],  # 6 ceiling_lamps, double_bed\n        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 2.8, 1), (3, 2.8, 1), (4, 2.8, 2), (5, 2.8, 2), (2, 0.5, 2)],\n        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],\n        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]\n    )\n    \n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print("Expected: [0.0, -2.0, -2.0]")\n    assert rewards.shape[0] == 3\n    assert rewards[0] == 0.0, f"Scene 1 should have reward 0.0, got {rewards[0]}"\n    assert rewards[1] == -2.0, f"Scene 2 should have reward -2.0, got {rewards[1]}"\n    assert rewards[2] == -2.0, f"Scene 3 should have reward -2.0, got {rewards[2]}"', 'success_threshold': 0.0}], 'inpaint': {'ceiling_lamp': 4}}
        Reward Statistics = 

--- Stats from SR1_bed_presence_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: SR1_bed_presence ===

PERFORMANCE METRICS:
  • Success Rate: 32.0% (320/1000 scenes)
  • Mean Reward: -0.6800
  • Median Reward: -1.0000
  • Range: [-1.0000, 0.0000]
  • Std Dev: 0.4665
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.77
  • Kurtosis: -1.40
    • Min Rate: 68.0%
    • Near Min Rate: 68.0%
    • Max Rate: 32.0%
    • Near Max Rate: 32.0%

============================================================


--- Stats from SR2_ceiling_lamp_count_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: SR2_ceiling_lamp_count ===

PERFORMANCE METRICS:
  • Success Rate: 0.0% (0/4041 scenes)
  • Mean Reward: -3.6847
  • Median Reward: -4.0000
  • Range: [-4.0000, -1.0000]
  • Std Dev: 0.4746
  • Percentiles:
      - P1: -4.0000
      - P5: -4.0000
      - P25: -4.0000
      - P75: -3.0000
      - P95: -3.0000
      - P99: -3.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.94
  • Kurtosis: -0.66
    • Min Rate: 68.9%
    • Near Min Rate: 68.9%
    • Max Rate: 0.0%
    • Near Max Rate: 0.0%

============================================================


--- Stats from R3_ceiling_lamps_above_bed_corners_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R3_ceiling_lamps_above_bed_corners ===

PERFORMANCE METRICS:
  • Success Rate: 0.6% (26/4041 scenes)
  • Mean Reward: -7.2505
  • Median Reward: -10.0000
  • Range: [-10.0000, -0.0227]
  • Std Dev: 4.0985
  • Percentiles:
      - P1: -10.0000
      - P5: -10.0000
      - P25: -10.0000
      - P75: -1.3772
      - P95: -0.8734
      - P99: -0.5631

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.82
  • Kurtosis: -1.31
    • Min Rate: 68.9%
    • Near Min Rate: 68.9%
    • Max Rate: 0.0%
    • Near Max Rate: 0.0%

============================================================


--- Stats from SR4_lamps_near_bed_corners_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: SR4_lamps_near_bed_corners ===

PERFORMANCE METRICS:
  • Success Rate: 0.0% (0/1000 scenes)
  • Mean Reward: -0.9217
  • Median Reward: -0.9438
  • Range: [-1.0000, -0.3562]
  • Std Dev: 0.0851
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -0.8858
      - P95: -0.7532
      - P99: -0.6182

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 1.92
  • Kurtosis: 5.81
    • Min Rate: 25.9%
    • Near Min Rate: 44.5%
    • Max Rate: 0.1%
    • Near Max Rate: 0.2%

============================================================


--- Stats from R1_ceiling_lamp_presence_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R1_ceiling_lamp_presence ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (1000/1000 scenes)
  • Mean Reward: 0.0000
  • Median Reward: 0.0000
  • Range: [0.0000, 0.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: 0.0000
      - P5: 0.0000
      - P25: 0.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================


--- Stats from R4_four_ceiling_lamps_required_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R4_four_ceiling_lamps_required ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (1000/1000 scenes)
  • Mean Reward: 0.0000
  • Median Reward: 0.0000
  • Range: [-0.0000, -0.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: -0.0000
      - P5: -0.0000
      - P25: -0.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================


--- Stats from SR1_bed_presence_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: SR1_bed_presence ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (4041/4041 scenes)
  • Mean Reward: 0.0000
  • Median Reward: 0.0000
  • Range: [0.0000, 0.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: 0.0000
      - P5: 0.0000
      - P25: 0.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================


--- Stats from R4_four_ceiling_lamps_required_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R4_four_ceiling_lamps_required ===

PERFORMANCE METRICS:
  • Success Rate: 0.0% (0/4041 scenes)
  • Mean Reward: -3.6847
  • Median Reward: -4.0000
  • Range: [-4.0000, -1.0000]
  • Std Dev: 0.4746
  • Percentiles:
      - P1: -4.0000
      - P5: -4.0000
      - P25: -4.0000
      - P75: -3.0000
      - P95: -3.0000
      - P99: -3.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.94
  • Kurtosis: -0.66
    • Min Rate: 68.9%
    • Near Min Rate: 68.9%
    • Max Rate: 0.0%
    • Near Max Rate: 0.0%

============================================================


--- Stats from R2_bed_presence_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R2_bed_presence ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (4041/4041 scenes)
  • Mean Reward: 0.0000
  • Median Reward: 0.0000
  • Range: [0.0000, 0.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: 0.0000
      - P5: 0.0000
      - P25: 0.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================


--- Stats from SR3_ceiling_lamp_height_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: SR3_ceiling_lamp_height ===

PERFORMANCE METRICS:
  • Success Rate: 5.6% (56/1000 scenes)
  • Mean Reward: -0.4960
  • Median Reward: -0.4932
  • Range: [-0.9909, -0.0056]
  • Std Dev: 0.1826
  • Percentiles:
      - P1: -0.9388
      - P5: -0.8086
      - P25: -0.6072
      - P75: -0.3748
      - P95: -0.1841
      - P99: -0.0494

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -0.06
  • Kurtosis: 0.05
    • Min Rate: 0.1%
    • Near Min Rate: 1.0%
    • Max Rate: 0.1%
    • Near Max Rate: 1.2%

============================================================


--- Stats from SR4_lamps_near_bed_corners_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: SR4_lamps_near_bed_corners ===

PERFORMANCE METRICS:
  • Success Rate: 0.3% (12/4041 scenes)
  • Mean Reward: -0.9609
  • Median Reward: -1.0000
  • Range: [-1.0000, -0.0010]
  • Std Dev: 0.1023
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -0.9757
      - P95: -0.7774
      - P99: -0.4644

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 4.26
  • Kurtosis: 22.49
    • Min Rate: 68.9%
    • Near Min Rate: 80.7%
    • Max Rate: 0.0%
    • Near Max Rate: 0.0%

============================================================


--- Stats from R1_ceiling_lamp_presence_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R1_ceiling_lamp_presence ===

PERFORMANCE METRICS:
  • Success Rate: 31.1% (1256/4041 scenes)
  • Mean Reward: -0.6892
  • Median Reward: -1.0000
  • Range: [-1.0000, 0.0000]
  • Std Dev: 0.4628
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.82
  • Kurtosis: -1.33
    • Min Rate: 68.9%
    • Near Min Rate: 68.9%
    • Max Rate: 31.1%
    • Near Max Rate: 31.1%

============================================================


--- Stats from R3_ceiling_lamps_above_bed_corners_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R3_ceiling_lamps_above_bed_corners ===

PERFORMANCE METRICS:
  • Success Rate: 0.0% (0/1000 scenes)
  • Mean Reward: -5.8722
  • Median Reward: -4.8569
  • Range: [-10.0000, -1.8707]
  • Std Dev: 2.4899
  • Percentiles:
      - P1: -10.0000
      - P5: -10.0000
      - P25: -10.0000
      - P75: -4.2906
      - P95: -3.4039
      - P99: -2.8225

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -0.96
  • Kurtosis: -0.85
    • Min Rate: 25.9%
    • Near Min Rate: 25.9%
    • Max Rate: 0.1%
    • Near Max Rate: 0.1%

============================================================


--- Stats from SR3_ceiling_lamp_height_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: SR3_ceiling_lamp_height ===

PERFORMANCE METRICS:
  • Success Rate: 74.5% (3011/4041 scenes)
  • Mean Reward: -0.1353
  • Median Reward: 0.0000
  • Range: [-1.0000, 0.0000]
  • Std Dev: 0.2469
  • Percentiles:
      - P1: -0.9625
      - P5: -0.7045
      - P25: -0.2265
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -1.78
  • Kurtosis: 2.14
    • Min Rate: 0.0%
    • Near Min Rate: 1.2%
    • Max Rate: 68.9%
    • Near Max Rate: 71.2%

============================================================


--- Stats from R2_bed_presence_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R2_bed_presence ===

PERFORMANCE METRICS:
  • Success Rate: 32.0% (320/1000 scenes)
  • Mean Reward: -0.6800
  • Median Reward: -1.0000
  • Range: [-1.0000, 0.0000]
  • Std Dev: 0.4665
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 0.77
  • Kurtosis: -1.40
    • Min Rate: 68.0%
    • Near Min Rate: 68.0%
    • Max Rate: 32.0%
    • Near Max Rate: 32.0%

============================================================


--- Stats from SR2_ceiling_lamp_count_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: SR2_ceiling_lamp_count ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (1000/1000 scenes)
  • Mean Reward: 0.0000
  • Median Reward: 0.0000
  • Range: [-0.0000, -0.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: -0.0000
      - P5: -0.0000
      - P25: -0.0000
      - P75: 0.0000
      - P95: 0.0000
      - P99: 0.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================

    