
        User Prompt: I want to follow Vaastu for bedroom layout. The beds headboard should face east.
        Initial Constraints: {'constraints': [{'id': 'C1', 'name': 'bed_headboard_faces_east', 'description': "Ensures that the bed's headboard faces east (+X direction). The bed should be oriented such that when a person lies on it, their head points toward the +X axis. This checks that the bed's forward direction (where the headboard is located) aligns with the east direction within an acceptable angular tolerance."}, {'id': 'C2', 'name': 'bed_exists_in_scene', 'description': 'Verifies that at least one bed object (double_bed, single_bed, or kids_bed) exists in the bedroom scene and is not marked as empty. This is a prerequisite for applying the headboard orientation constraint.'}, {'id': 'C3', 'name': 'bed_placement_stability', 'description': 'Ensures that the bed is placed on the floor (y-position approximately equals bed_height/2) and not floating or embedded in the floor. This validates proper vertical positioning for the Vaastu-compliant bed.'}]}
        Initial Reward Functions: {'rewards': [{'id': 'R1', 'constraint_id': 'C1', 'name': 'bed_headboard_faces_east', 'code': 'import torch\nimport math\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to ensure bed headboard faces east (+X direction).\n    \n    The headboard facing east means when lying on the bed, the head points toward +X.\n    In 3D-FRONT, bed orientation is given as [cos(θ), sin(θ)] for z-axis rotation.\n    The bed\'s "front" (headboard) should align with +X axis (east).\n    \n    Reward: Cosine similarity between bed\'s front direction and east direction.\n    Range: [-1, 1] where 1 = perfect alignment, -1 = opposite direction\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    positions = parsed_scenes[\'positions\']  # (B, N, 3)\n    sizes = parsed_scenes[\'sizes\']  # (B, N, 3)\n    orientations = parsed_scenes[\'orientations\']  # (B, N, 2)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    \n    B, N = positions.shape[:2]\n    \n    # Get bed class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    bed_classes = [labels_to_idx[\'double_bed\'], labels_to_idx[\'single_bed\'], labels_to_idx[\'kids_bed\']]\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    for b in range(B):\n        # Find bed objects in the scene\n        bed_mask = torch.zeros(N, dtype=torch.bool, device=device)\n        for bed_class in bed_classes:\n            bed_mask |= (object_indices[b] == bed_class) & (~is_empty[b])\n        \n        if not bed_mask.any():\n            # No bed found, give penalty\n            rewards[b] = -1.0\n            continue\n        \n        # Process all beds in the scene (take the one with best alignment)\n        bed_indices = torch.where(bed_mask)[0]\n        max_alignment = -1.0\n        \n        for bed_idx in bed_indices:\n            # Get bed orientation\n            cos_theta = orientations[b, bed_idx, 0]\n            sin_theta = orientations[b, bed_idx, 1]\n            \n            # The orientation gives the forward direction of the bed in XZ plane\n            # Forward direction vector: [cos(θ), sin(θ)]\n            bed_forward = torch.tensor([cos_theta, sin_theta], device=device)\n            \n            # East direction is +X axis: [1, 0] in XZ plane\n            east_direction = torch.tensor([1.0, 0.0], device=device)\n            \n            # Compute cosine similarity (dot product of normalized vectors)\n            # Both vectors are already unit vectors\n            alignment = torch.dot(bed_forward, east_direction)\n            \n            max_alignment = max(max_alignment, alignment.item())\n        \n        rewards[b] = max_alignment\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for bed headboard faces east constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    double_bed_idx = labels_to_idx[\'double_bed\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    wardrobe_idx = labels_to_idx[\'wardrobe\']\n    \n    # Scene 1: Bed facing exactly east (orientation = [1, 0])\n    num_objects_1 = 3\n    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing east\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: Bed facing west (orientation = [-1, 0], 180 degrees)\n    num_objects_2 = 3\n    class_label_indices_2 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_2 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_2 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_2 = [(-1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing west\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: Bed facing north (orientation = [0, 1], 90 degrees)\n    num_objects_3 = 3\n    class_label_indices_3 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_3 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_3 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_3 = [(0.0, 1.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing north\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (bed facing east): {rewards[0].item():.4f}")\n    print(f"Scene 2 (bed facing west): {rewards[1].item():.4f}")\n    print(f"Scene 3 (bed facing north): {rewards[2].item():.4f}")\n    \n    assert rewards.shape[0] == 3, "Should have 3 scenes"\n    \n    # Test assertions\n    assert rewards[0].item() > 0.95, f"Scene 1: Bed facing east should have reward close to 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() < -0.95, f"Scene 2: Bed facing west should have reward close to -1.0, got {rewards[1].item()}"\n    assert abs(rewards[2].item()) < 0.1, f"Scene 3: Bed facing north should have reward close to 0.0, got {rewards[2].item()}"\n    assert rewards[0] > rewards[2] > rewards[1], f"Scene 1 > Scene 3 > Scene 2, got {rewards[0]}, {rewards[2]}, {rewards[1]}"\n    \n    print("All tests passed!")', 'success_threshold': 0.9}, {'id': 'R2', 'constraint_id': 'C2', 'name': 'bed_exists_in_scene', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to verify at least one bed exists in the scene.\n    \n    Returns:\n        1.0 if at least one bed (double_bed, single_bed, or kids_bed) exists\n        -1.0 if no bed exists\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    \n    B = object_indices.shape[0]\n    \n    # Get bed class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    bed_classes = [labels_to_idx[\'double_bed\'], labels_to_idx[\'single_bed\'], labels_to_idx[\'kids_bed\']]\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    for b in range(B):\n        # Check if any bed exists\n        has_bed = False\n        for bed_class in bed_classes:\n            bed_mask = (object_indices[b] == bed_class) & (~is_empty[b])\n            if bed_mask.any():\n                has_bed = True\n                break\n        \n        rewards[b] = 1.0 if has_bed else -1.0\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for bed existence constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    double_bed_idx = labels_to_idx[\'double_bed\']\n    single_bed_idx = labels_to_idx[\'single_bed\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    wardrobe_idx = labels_to_idx[\'wardrobe\']\n    chair_idx = labels_to_idx[\'chair\']\n    \n    # Scene 1: Has double bed\n    num_objects_1 = 3\n    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: Has single bed\n    num_objects_2 = 3\n    class_label_indices_2 = [single_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_2 = [(0, 0.3, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_2 = [(0.6, 0.3, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: No bed (only furniture)\n    num_objects_3 = 3\n    class_label_indices_3 = [nightstand_idx, wardrobe_idx, chair_idx]\n    translations_3 = [(0, 0.3, 0), (1.5, 0.5, 0), (-1.5, 0.4, 0)]\n    sizes_3 = [(0.3, 0.3, 0.3), (0.5, 0.5, 1.0), (0.4, 0.4, 0.4)]\n    orientations_3 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (has double bed): {rewards[0].item()}")\n    print(f"Scene 2 (has single bed): {rewards[1].item()}")\n    print(f"Scene 3 (no bed): {rewards[2].item()}")\n    \n    assert rewards.shape[0] == 3, "Should have 3 scenes"\n    \n    # Test assertions\n    assert rewards[0].item() == 1.0, f"Scene 1: Has double bed, should return 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() == 1.0, f"Scene 2: Has single bed, should return 1.0, got {rewards[1].item()}"\n    assert rewards[2].item() == -1.0, f"Scene 3: No bed, should return -1.0, got {rewards[2].item()}"\n    \n    print("All tests passed!")', 'success_threshold': 1.0}, {'id': 'R3', 'constraint_id': 'C3', 'name': 'bed_placement_stability', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to ensure bed is placed correctly on the floor.\n    \n    The bed should be at y-position ≈ bed_height/2 (since position is centroid and size is half-extent).\n    Reward is based on the deviation from expected floor placement.\n    \n    Returns:\n        Reward proportional to correct placement. Max reward = 1.0 for perfect placement.\n        Penalty increases with distance from correct y-position.\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    positions = parsed_scenes[\'positions\']  # (B, N, 3)\n    sizes = parsed_scenes[\'sizes\']  # (B, N, 3)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    \n    B, N = positions.shape[:2]\n    \n    # Get bed class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    bed_classes = [labels_to_idx[\'double_bed\'], labels_to_idx[\'single_bed\'], labels_to_idx[\'kids_bed\']]\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    # Tolerance for floor placement (in meters)\n    tolerance = 0.05  # 5cm tolerance\n    \n    for b in range(B):\n        # Find bed objects in the scene\n        bed_mask = torch.zeros(N, dtype=torch.bool, device=device)\n        for bed_class in bed_classes:\n            bed_mask |= (object_indices[b] == bed_class) & (~is_empty[b])\n        \n        if not bed_mask.any():\n            # No bed found, give penalty\n            rewards[b] = -1.0\n            continue\n        \n        # Process all beds in the scene (average reward across all beds)\n        bed_indices = torch.where(bed_mask)[0]\n        total_reward = 0.0\n        \n        for bed_idx in bed_indices:\n            # Get bed position and size\n            bed_y = positions[b, bed_idx, 1]  # y-coordinate of centroid\n            bed_height_half = sizes[b, bed_idx, 1]  # half-height (sy/2)\n            \n            # Expected y-position: bed should be at y = bed_height/2\n            # Since sizes are half-extents, full height = 2 * bed_height_half\n            # So centroid should be at y = bed_height_half\n            expected_y = bed_height_half\n            \n            # Calculate deviation from expected position\n            deviation = torch.abs(bed_y - expected_y)\n            \n            # Calculate reward: exponential decay based on deviation\n            # Perfect placement (deviation=0) -> reward=1.0\n            # Large deviation -> reward approaches -1.0\n            if deviation <= tolerance:\n                bed_reward = 1.0\n            else:\n                # Exponential decay: reward = 1 - 2 * (1 - exp(-deviation))\n                # This gives values from 1.0 (perfect) to -1.0 (very bad)\n                normalized_dev = (deviation - tolerance) / 0.5  # Normalize to reasonable scale\n                bed_reward = 1.0 - 2.0 * (1.0 - torch.exp(-normalized_dev))\n                bed_reward = torch.clamp(bed_reward, -1.0, 1.0)\n            \n            total_reward += bed_reward\n        \n        # Average reward across all beds\n        rewards[b] = total_reward / len(bed_indices)\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for bed floor placement constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    double_bed_idx = labels_to_idx[\'double_bed\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    wardrobe_idx = labels_to_idx[\'wardrobe\']\n    \n    # Scene 1: Bed correctly placed (y = bed_height/2 = 0.4)\n    num_objects_1 = 3\n    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Correct: y = size_y = 0.4\n    sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: Bed floating (y = 1.0, should be 0.4)\n    num_objects_2 = 3\n    class_label_indices_2 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_2 = [(0, 1.0, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Floating: y = 1.0 instead of 0.4\n    sizes_2 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: Bed slightly off (y = 0.5, should be 0.4)\n    num_objects_3 = 3\n    class_label_indices_3 = [double_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_3 = [(0, 0.5, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Slightly off: y = 0.5 instead of 0.4\n    sizes_3 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_3 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (correct placement): {rewards[0].item():.4f}")\n    print(f"Scene 2 (floating bed): {rewards[1].item():.4f}")\n    print(f"Scene 3 (slightly off): {rewards[2].item():.4f}")\n    \n    assert rewards.shape[0] == 3, "Should have 3 scenes"\n    \n    # Test assertions\n    assert rewards[0].item() > 0.95, f"Scene 1: Correct placement should have reward close to 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() < 0.0, f"Scene 2: Floating bed should have negative reward, got {rewards[1].item()}"\n    assert 0.0 < rewards[2].item() < 1.0, f"Scene 3: Slightly off should have moderate positive reward, got {rewards[2].item()}"\n    assert rewards[0] > rewards[2] > rewards[1], f"Scene 1 > Scene 3 > Scene 2, got {rewards[0]}, {rewards[2]}, {rewards[1]}"\n    \n    print("All tests passed!")', 'success_threshold': 0.95}], 'inpaint': {}}
        Reward Statistics = 

--- Stats from R1_bed_headboard_faces_east_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R1_bed_headboard_faces_east ===

PERFORMANCE METRICS:
  • Success Rate: 26.0% (1050/4041 scenes)
  • Mean Reward: 0.1372
  • Median Reward: 0.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.6045
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: 0.0000
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -0.07
  • Kurtosis: -0.37
    • Min Rate: 11.9%
    • Near Min Rate: 12.4%
    • Max Rate: 25.9%
    • Near Max Rate: 26.0%

============================================================


--- Stats from R2_bed_exists_in_scene_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R2_bed_exists_in_scene ===

PERFORMANCE METRICS:
  • Success Rate: 99.7% (997/1000 scenes)
  • Mean Reward: 0.9940
  • Median Reward: 1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.1094
  • Percentiles:
      - P1: 1.0000
      - P5: 1.0000
      - P25: 1.0000
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -18.18
  • Kurtosis: 328.34
    • Min Rate: 0.3%
    • Near Min Rate: 0.3%
    • Max Rate: 99.7%
    • Near Max Rate: 99.7%

============================================================


--- Stats from R3_bed_placement_stability_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R3_bed_placement_stability ===

PERFORMANCE METRICS:
  • Success Rate: 99.8% (4031/4041 scenes)
  • Mean Reward: 0.9996
  • Median Reward: 1.0000
  • Range: [0.8093, 1.0000]
  • Std Dev: 0.0078
  • Percentiles:
      - P1: 1.0000
      - P5: 1.0000
      - P25: 1.0000
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -22.66
  • Kurtosis: 526.33
    • Min Rate: 0.0%
    • Near Min Rate: 0.1%
    • Max Rate: 99.7%
    • Near Max Rate: 99.8%

============================================================


--- Stats from R3_bed_placement_stability_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R3_bed_placement_stability ===

PERFORMANCE METRICS:
  • Success Rate: 98.5% (985/1000 scenes)
  • Mean Reward: 0.9918
  • Median Reward: 1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.1111
  • Percentiles:
      - P1: 0.9304
      - P5: 0.9977
      - P25: 1.0000
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -17.40
  • Kurtosis: 307.12
    • Min Rate: 0.3%
    • Near Min Rate: 0.3%
    • Max Rate: 94.5%
    • Near Max Rate: 98.5%

============================================================


--- Stats from R1_bed_headboard_faces_east_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R1_bed_headboard_faces_east ===

PERFORMANCE METRICS:
  • Success Rate: 34.1% (341/1000 scenes)
  • Mean Reward: 0.2320
  • Median Reward: 0.0009
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.6325
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -0.0017
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: -0.24
  • Kurtosis: -0.65
    • Min Rate: 7.8%
    • Near Min Rate: 11.1%
    • Max Rate: 25.9%
    • Near Max Rate: 34.0%

============================================================


--- Stats from R2_bed_exists_in_scene_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R2_bed_exists_in_scene ===

PERFORMANCE METRICS:
  • Success Rate: 100.0% (4041/4041 scenes)
  • Mean Reward: 1.0000
  • Median Reward: 1.0000
  • Range: [1.0000, 1.0000]
  • Std Dev: 0.0000
  • Percentiles:
      - P1: 1.0000
      - P5: 1.0000
      - P25: 1.0000
      - P75: 1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: nan
  • Kurtosis: nan
    • Min Rate: 100.0%
    • Near Min Rate: 100.0%
    • Max Rate: 100.0%
    • Near Max Rate: 100.0%

============================================================

    