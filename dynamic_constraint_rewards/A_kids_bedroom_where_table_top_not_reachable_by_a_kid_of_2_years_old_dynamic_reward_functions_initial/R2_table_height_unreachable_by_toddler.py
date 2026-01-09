import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function to ensure table surfaces are unreachable by a 2-year-old child.
    
    A 2-year-old can reach approximately 1.0-1.1 meters high.
    Table top height = position_y + size_y (centroid + half-height)
    
    Returns:
        Reward based on how safely positioned tables are.
        1.0 if all tables have top > 1.1m
        Gradual penalty for tables below threshold
        -1.0 if no tables exist (constraint not applicable)
    '''
    
    device = parsed_scenes['device']
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    
    B, N = positions.shape[:2]
    
    # Get table-type class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    table_classes = [labels_to_idx['table'], labels_to_idx['desk'], labels_to_idx['dressing_table']]
    
    # Safety threshold: 2-year-old reach height
    SAFE_HEIGHT = 1.1  # meters
    TOLERANCE = 0.05  # 5cm tolerance for near-safe heights
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find table-type objects in the scene
        table_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for table_class in table_classes:
            table_mask |= (object_indices[b] == table_class) & (~is_empty[b])
        
        if not table_mask.any():
            # No tables found - constraint not applicable
            rewards[b] = -1.0
            continue
        
        # Process all tables in the scene
        table_indices = torch.where(table_mask)[0]
        total_reward = 0.0
        
        for table_idx in table_indices:
            # Calculate table top height
            table_centroid_y = positions[b, table_idx, 1]
            table_half_height = sizes[b, table_idx, 1]
            table_top_y = table_centroid_y + table_half_height
            
            # Calculate reward based on height safety
            height_above_threshold = table_top_y - SAFE_HEIGHT
            
            if height_above_threshold >= TOLERANCE:
                # Safely above threshold
                table_reward = 1.0
            elif height_above_threshold >= 0:
                # Within tolerance - partial reward
                table_reward = height_above_threshold / TOLERANCE
            else:
                # Below threshold - penalty proportional to how dangerous
                # Use exponential decay to penalize dangerous heights more
                normalized_violation = -height_above_threshold / 0.3  # Normalize to ~30cm range
                table_reward = -torch.tanh(torch.tensor(normalized_violation, device=device))
                table_reward = torch.clamp(table_reward, -1.0, 0.0)
            
            total_reward += table_reward
        
        # Average reward across all tables
        rewards[b] = total_reward / len(table_indices)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for table height safety constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    table_idx = labels_to_idx['table']
    desk_idx = labels_to_idx['desk']
    nightstand_idx = labels_to_idx['nightstand']
    
    # Scene 1: Table with safe height (top at 1.5m)
    # position_y = 0.75, size_y = 0.75 -> top = 1.5m (SAFE)
    num_objects_1 = 3
    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Table with unsafe height (top at 0.9m)
    # position_y = 0.45, size_y = 0.45 -> top = 0.9m (UNSAFE - below 1.1m)
    num_objects_2 = 3
    class_label_indices_2 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_2 = [(0, 0.3, 0), (1.5, 0.45, 0), (-1.5, 0.3, 0)]
    sizes_2 = [(0.9, 0.3, 0.75), (0.4, 0.45, 0.4), (0.3, 0.3, 0.3)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No table (only kids_bed)
    num_objects_3 = 2
    class_label_indices_3 = [kids_bed_idx, nightstand_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.3, 0)]
    sizes_3 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Scene 4: Table at exactly threshold (top at 1.1m)
    # position_y = 0.55, size_y = 0.55 -> top = 1.1m (MARGINAL)
    num_objects_4 = 2
    class_label_indices_4 = [kids_bed_idx, desk_idx]
    translations_4 = [(0, 0.3, 0), (1.5, 0.55, 0)]
    sizes_4 = [(0.9, 0.3, 0.75), (0.5, 0.55, 0.5)]
    orientations_4 = [(1.0, 0.0), (1.0, 0.0)]
    scene_4 = create_scene(room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4)
    
    # Stack scenes
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print(f"Scene 1 (safe table at 1.5m): {rewards[0].item():.4f}")
    print(f"Scene 2 (unsafe table at 0.9m): {rewards[1].item():.4f}")
    print(f"Scene 3 (no table): {rewards[2].item():.4f}")
    print(f"Scene 4 (table at threshold 1.1m): {rewards[3].item():.4f}")
    
    assert rewards.shape[0] == 4, "Should have 4 scenes"
    assert rewards[0].item() > 0.95, f"Scene 1: Safe table should have reward close to 1.0, got {rewards[0].item()}"
    assert rewards[1].item() < 0.0, f"Scene 2: Unsafe table should have negative reward, got {rewards[1].item()}"
    assert rewards[2].item() == -1.0, f"Scene 3: No table should return -1.0, got {rewards[2].item()}"
    assert 0.0 <= rewards[3].item() <= 0.1, f"Scene 4: Table at threshold should have reward near 0, got {rewards[3].item()}"
    assert rewards[0] > rewards[3] > rewards[1], f"Scene 1 > Scene 4 > Scene 2, got {rewards[0]}, {rewards[3]}, {rewards[1]}"
    
    print("All tests passed!")