
        User Prompt: A kids bedroom where table top not reachable by a kid of 2 years old.
        Initial Constraints: {'constraints': [{'id': 'C1', 'name': 'kids_bed_present', 'description': 'Verifies that at least one kids_bed exists in the bedroom scene, confirming this is indeed a kids bedroom as specified by the user.'}, {'id': 'C2', 'name': 'table_height_unreachable_by_toddler', 'description': 'Ensures that any table objects (table, desk, dressing_table) in the scene have their top surface positioned at a height greater than 1.1 meters from the floor, making them unreachable by a 2-year-old child who can reach up to approximately 1.0-1.1 meters with arms extended. The table top height is calculated as the y-position (centroid) plus the half-height (size_y), since y-position represents the centroid and size represents half-extents.'}, {'id': 'C3', 'name': 'table_exists_in_scene', 'description': 'Verifies that at least one table-type object (table, desk, or dressing_table) exists in the scene. This is a prerequisite for applying the height constraint, as the safety requirement is meaningless without tables present.'}]}
        Initial Reward Functions: {'rewards': [{'id': 'R1', 'constraint_id': 'C1', 'name': 'kids_bed_present', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to verify that at least one kids_bed exists in the scene.\n    \n    Returns:\n        1.0 if kids_bed exists\n        -1.0 if no kids_bed exists\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    \n    B = object_indices.shape[0]\n    \n    # Get kids_bed class index\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    kids_bed_idx = labels_to_idx[\'kids_bed\']\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    for b in range(B):\n        # Check if kids_bed exists\n        kids_bed_mask = (object_indices[b] == kids_bed_idx) & (~is_empty[b])\n        has_kids_bed = kids_bed_mask.any()\n        \n        rewards[b] = 1.0 if has_kids_bed else -1.0\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for kids_bed presence constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    kids_bed_idx = labels_to_idx[\'kids_bed\']\n    table_idx = labels_to_idx[\'table\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    wardrobe_idx = labels_to_idx[\'wardrobe\']\n    \n    # Scene 1: Has kids_bed\n    num_objects_1 = 3\n    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]\n    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]\n    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: No kids_bed (has regular furniture)\n    num_objects_2 = 3\n    class_label_indices_2 = [table_idx, nightstand_idx, wardrobe_idx]\n    translations_2 = [(0, 0.75, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_2 = [(0.4, 0.75, 0.4), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: Has kids_bed with other furniture\n    num_objects_3 = 2\n    class_label_indices_3 = [kids_bed_idx, wardrobe_idx]\n    translations_3 = [(0, 0.3, 0), (1.5, 0.5, 0)]\n    sizes_3 = [(0.9, 0.3, 0.75), (0.5, 0.5, 1.0)]\n    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (has kids_bed): {rewards[0].item()}")\n    print(f"Scene 2 (no kids_bed): {rewards[1].item()}")\n    print(f"Scene 3 (has kids_bed): {rewards[2].item()}")\n    \n    assert rewards.shape[0] == 3, "Should have 3 scenes"\n    assert rewards[0].item() == 1.0, f"Scene 1: Has kids_bed, should return 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() == -1.0, f"Scene 2: No kids_bed, should return -1.0, got {rewards[1].item()}"\n    assert rewards[2].item() == 1.0, f"Scene 3: Has kids_bed, should return 1.0, got {rewards[2].item()}"\n    \n    print("All tests passed!")', 'success_threshold': 1.0}, {'id': 'R2', 'constraint_id': 'C2', 'name': 'table_height_unreachable_by_toddler', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to ensure table surfaces are unreachable by a 2-year-old child.\n    \n    A 2-year-old can reach approximately 1.0-1.1 meters high.\n    Table top height = position_y + size_y (centroid + half-height)\n    \n    Returns:\n        Reward based on how safely positioned tables are.\n        1.0 if all tables have top > 1.1m\n        Gradual penalty for tables below threshold\n        -1.0 if no tables exist (constraint not applicable)\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    positions = parsed_scenes[\'positions\']  # (B, N, 3)\n    sizes = parsed_scenes[\'sizes\']  # (B, N, 3)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    \n    B, N = positions.shape[:2]\n    \n    # Get table-type class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    table_classes = [labels_to_idx[\'table\'], labels_to_idx[\'desk\'], labels_to_idx[\'dressing_table\']]\n    \n    # Safety threshold: 2-year-old reach height\n    SAFE_HEIGHT = 1.1  # meters\n    TOLERANCE = 0.05  # 5cm tolerance for near-safe heights\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    for b in range(B):\n        # Find table-type objects in the scene\n        table_mask = torch.zeros(N, dtype=torch.bool, device=device)\n        for table_class in table_classes:\n            table_mask |= (object_indices[b] == table_class) & (~is_empty[b])\n        \n        if not table_mask.any():\n            # No tables found - constraint not applicable\n            rewards[b] = -1.0\n            continue\n        \n        # Process all tables in the scene\n        table_indices = torch.where(table_mask)[0]\n        total_reward = 0.0\n        \n        for table_idx in table_indices:\n            # Calculate table top height\n            table_centroid_y = positions[b, table_idx, 1]\n            table_half_height = sizes[b, table_idx, 1]\n            table_top_y = table_centroid_y + table_half_height\n            \n            # Calculate reward based on height safety\n            height_above_threshold = table_top_y - SAFE_HEIGHT\n            \n            if height_above_threshold >= TOLERANCE:\n                # Safely above threshold\n                table_reward = 1.0\n            elif height_above_threshold >= 0:\n                # Within tolerance - partial reward\n                table_reward = height_above_threshold / TOLERANCE\n            else:\n                # Below threshold - penalty proportional to how dangerous\n                # Use exponential decay to penalize dangerous heights more\n                normalized_violation = -height_above_threshold / 0.3  # Normalize to ~30cm range\n                table_reward = -torch.tanh(torch.tensor(normalized_violation, device=device))\n                table_reward = torch.clamp(table_reward, -1.0, 0.0)\n            \n            total_reward += table_reward\n        \n        # Average reward across all tables\n        rewards[b] = total_reward / len(table_indices)\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for table height safety constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    kids_bed_idx = labels_to_idx[\'kids_bed\']\n    table_idx = labels_to_idx[\'table\']\n    desk_idx = labels_to_idx[\'desk\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    \n    # Scene 1: Table with safe height (top at 1.5m)\n    # position_y = 0.75, size_y = 0.75 -> top = 1.5m (SAFE)\n    num_objects_1 = 3\n    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]\n    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]\n    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: Table with unsafe height (top at 0.9m)\n    # position_y = 0.45, size_y = 0.45 -> top = 0.9m (UNSAFE - below 1.1m)\n    num_objects_2 = 3\n    class_label_indices_2 = [kids_bed_idx, table_idx, nightstand_idx]\n    translations_2 = [(0, 0.3, 0), (1.5, 0.45, 0), (-1.5, 0.3, 0)]\n    sizes_2 = [(0.9, 0.3, 0.75), (0.4, 0.45, 0.4), (0.3, 0.3, 0.3)]\n    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: No table (only kids_bed)\n    num_objects_3 = 2\n    class_label_indices_3 = [kids_bed_idx, nightstand_idx]\n    translations_3 = [(0, 0.3, 0), (1.5, 0.3, 0)]\n    sizes_3 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3)]\n    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Scene 4: Table at exactly threshold (top at 1.1m)\n    # position_y = 0.55, size_y = 0.55 -> top = 1.1m (MARGINAL)\n    num_objects_4 = 2\n    class_label_indices_4 = [kids_bed_idx, desk_idx]\n    translations_4 = [(0, 0.3, 0), (1.5, 0.55, 0)]\n    sizes_4 = [(0.9, 0.3, 0.75), (0.5, 0.55, 0.5)]\n    orientations_4 = [(1.0, 0.0), (1.0, 0.0)]\n    scene_4 = create_scene(room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (safe table at 1.5m): {rewards[0].item():.4f}")\n    print(f"Scene 2 (unsafe table at 0.9m): {rewards[1].item():.4f}")\n    print(f"Scene 3 (no table): {rewards[2].item():.4f}")\n    print(f"Scene 4 (table at threshold 1.1m): {rewards[3].item():.4f}")\n    \n    assert rewards.shape[0] == 4, "Should have 4 scenes"\n    assert rewards[0].item() > 0.95, f"Scene 1: Safe table should have reward close to 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() < 0.0, f"Scene 2: Unsafe table should have negative reward, got {rewards[1].item()}"\n    assert rewards[2].item() == -1.0, f"Scene 3: No table should return -1.0, got {rewards[2].item()}"\n    assert 0.0 <= rewards[3].item() <= 0.1, f"Scene 4: Table at threshold should have reward near 0, got {rewards[3].item()}"\n    assert rewards[0] > rewards[3] > rewards[1], f"Scene 1 > Scene 4 > Scene 2, got {rewards[0]}, {rewards[3]}, {rewards[1]}"\n    \n    print("All tests passed!")', 'success_threshold': 0.95}, {'id': 'R3', 'constraint_id': 'C3', 'name': 'table_exists_in_scene', 'code': 'import torch\nfrom dynamic_constraint_rewards.utilities import get_all_utility_functions\n\ndef get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Reward function to verify that at least one table-type object exists in the scene.\n    \n    Returns:\n        1.0 if at least one table/desk/dressing_table exists\n        -1.0 if no table-type objects exist\n    \'\'\'\n    \n    device = parsed_scenes[\'device\']\n    object_indices = parsed_scenes[\'object_indices\']  # (B, N)\n    is_empty = parsed_scenes[\'is_empty\']  # (B, N)\n    \n    B = object_indices.shape[0]\n    \n    # Get table-type class indices\n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    table_classes = [labels_to_idx[\'table\'], labels_to_idx[\'desk\'], labels_to_idx[\'dressing_table\']]\n    \n    # Initialize rewards\n    rewards = torch.zeros(B, device=device)\n    \n    for b in range(B):\n        # Check if any table-type object exists\n        has_table = False\n        for table_class in table_classes:\n            table_mask = (object_indices[b] == table_class) & (~is_empty[b])\n            if table_mask.any():\n                has_table = True\n                break\n        \n        rewards[b] = 1.0 if has_table else -1.0\n    \n    return rewards\n\ndef test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):\n    \'\'\'\n    Test function for table existence constraint.\n    \'\'\'\n    utility_functions = get_all_utility_functions()\n    create_scene = utility_functions["create_scene_for_testing"]["function"]\n    \n    labels_to_idx = {v: k for k, v in idx_to_labels.items()}\n    kids_bed_idx = labels_to_idx[\'kids_bed\']\n    table_idx = labels_to_idx[\'table\']\n    desk_idx = labels_to_idx[\'desk\']\n    dressing_table_idx = labels_to_idx[\'dressing_table\']\n    nightstand_idx = labels_to_idx[\'nightstand\']\n    wardrobe_idx = labels_to_idx[\'wardrobe\']\n    \n    # Scene 1: Has table\n    num_objects_1 = 3\n    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]\n    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]\n    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]\n    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)\n    \n    # Scene 2: Has desk\n    num_objects_2 = 3\n    class_label_indices_2 = [kids_bed_idx, desk_idx, nightstand_idx]\n    translations_2 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]\n    sizes_2 = [(0.9, 0.3, 0.75), (0.5, 0.75, 0.5), (0.3, 0.3, 0.3)]\n    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)\n    \n    # Scene 3: Has dressing_table\n    num_objects_3 = 3\n    class_label_indices_3 = [kids_bed_idx, dressing_table_idx, nightstand_idx]\n    translations_3 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]\n    sizes_3 = [(0.9, 0.3, 0.75), (0.5, 0.75, 0.4), (0.3, 0.3, 0.3)]\n    orientations_3 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)\n    \n    # Scene 4: No table-type objects\n    num_objects_4 = 3\n    class_label_indices_4 = [kids_bed_idx, nightstand_idx, wardrobe_idx]\n    translations_4 = [(0, 0.3, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]\n    sizes_4 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]\n    orientations_4 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]\n    scene_4 = create_scene(room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4)\n    \n    # Stack scenes\n    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]\n    parsed_scenes = {\n        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)\n        for k in tensor_keys\n    }\n    parsed_scenes[\'room_type\'] = room_type\n    parsed_scenes[\'device\'] = scene_1[\'device\']\n    \n    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)\n    print("Rewards:", rewards)\n    print(f"Scene 1 (has table): {rewards[0].item()}")\n    print(f"Scene 2 (has desk): {rewards[1].item()}")\n    print(f"Scene 3 (has dressing_table): {rewards[2].item()}")\n    print(f"Scene 4 (no table-type): {rewards[3].item()}")\n    \n    assert rewards.shape[0] == 4, "Should have 4 scenes"\n    assert rewards[0].item() == 1.0, f"Scene 1: Has table, should return 1.0, got {rewards[0].item()}"\n    assert rewards[1].item() == 1.0, f"Scene 2: Has desk, should return 1.0, got {rewards[1].item()}"\n    assert rewards[2].item() == 1.0, f"Scene 3: Has dressing_table, should return 1.0, got {rewards[2].item()}"\n    assert rewards[3].item() == -1.0, f"Scene 4: No table-type, should return -1.0, got {rewards[3].item()}"\n    \n    print("All tests passed!")', 'success_threshold': 1.0}]}
        Reward Statistics = 

--- Stats from R3_table_exists_in_scene_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R3_table_exists_in_scene ===

PERFORMANCE METRICS:
  • Success Rate: 23.6% (953/4041 scenes)
  • Mean Reward: -0.5283
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.8490
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 1.24
  • Kurtosis: -0.45
    • Min Rate: 76.4%
    • Near Min Rate: 76.4%
    • Max Rate: 23.6%
    • Near Max Rate: 23.6%

============================================================


--- Stats from R1_kids_bed_present_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R1_kids_bed_present ===

PERFORMANCE METRICS:
  • Success Rate: 3.8% (154/4041 scenes)
  • Mean Reward: -0.9238
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.3829
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: -1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 4.82
  • Kurtosis: 21.28
    • Min Rate: 96.2%
    • Near Min Rate: 96.2%
    • Max Rate: 3.8%
    • Near Max Rate: 3.8%

============================================================


--- Stats from R2_table_height_unreachable_by_toddler_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R2_table_height_unreachable_by_toddler ===

PERFORMANCE METRICS:
  • Success Rate: 4.6% (46/1000 scenes)
  • Mean Reward: -0.8498
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.4566
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: 0.0768
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 3.34
  • Kurtosis: 10.06
    • Min Rate: 80.6%
    • Near Min Rate: 83.0%
    • Max Rate: 4.6%
    • Near Max Rate: 4.6%

============================================================


--- Stats from R2_table_height_unreachable_by_toddler_llm_summary_dataset.txt ---

=== REWARD ANALYSIS: R2_table_height_unreachable_by_toddler ===

PERFORMANCE METRICS:
  • Success Rate: 6.1% (248/4041 scenes)
  • Mean Reward: -0.8204
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.5059
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 3.03
  • Kurtosis: 7.77
    • Min Rate: 76.4%
    • Near Min Rate: 79.2%
    • Max Rate: 6.1%
    • Near Max Rate: 6.1%

============================================================


--- Stats from R1_kids_bed_present_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R1_kids_bed_present ===

PERFORMANCE METRICS:
  • Success Rate: 2.6% (26/1000 scenes)
  • Mean Reward: -0.9480
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.3183
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: -1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 5.96
  • Kurtosis: 33.49
    • Min Rate: 97.4%
    • Near Min Rate: 97.4%
    • Max Rate: 2.6%
    • Near Max Rate: 2.6%

============================================================


--- Stats from R3_table_exists_in_scene_llm_summary_baseline.txt ---

=== REWARD ANALYSIS: R3_table_exists_in_scene ===

PERFORMANCE METRICS:
  • Success Rate: 19.4% (194/1000 scenes)
  • Mean Reward: -0.6120
  • Median Reward: -1.0000
  • Range: [-1.0000, 1.0000]
  • Std Dev: 0.7909
  • Percentiles:
      - P1: -1.0000
      - P5: -1.0000
      - P25: -1.0000
      - P75: -1.0000
      - P95: 1.0000
      - P99: 1.0000

DISTRIBUTION CHARACTERISTICS:

  • Skewness: 1.55
  • Kurtosis: 0.40
    • Min Rate: 80.6%
    • Near Min Rate: 80.6%
    • Max Rate: 19.4%
    • Near Max Rate: 19.4%

============================================================

    