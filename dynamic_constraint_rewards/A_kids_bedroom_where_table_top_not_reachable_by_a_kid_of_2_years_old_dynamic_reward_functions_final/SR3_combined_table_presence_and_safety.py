import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Combined reward that considers both table presence and safety.
    
    This provides a holistic view:
    - If tables exist and are safe (>1.1m): positive reward
    - If tables exist but unsafe (<1.1m): negative reward
    - If no tables exist: neutral (0.0)
    
    This helps the model understand the relationship between having tables
    and positioning them safely.
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
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find table-type objects
        table_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for table_class in table_classes:
            table_mask |= (object_indices[b] == table_class) & (~is_empty[b])
        
        if not table_mask.any():
            # No tables - neutral
            rewards[b] = 0.0
            continue
        
        # Tables exist - check safety
        table_indices = torch.where(table_mask)[0]
        safe_count = 0
        unsafe_count = 0
        
        for table_idx in table_indices:
            table_centroid_y = positions[b, table_idx, 1]
            table_half_height = sizes[b, table_idx, 1]
            table_top_y = table_centroid_y + table_half_height
            
            if table_top_y >= 1.1:
                safe_count += 1
            else:
                unsafe_count += 1
        
        total_tables = len(table_indices)
        safe_ratio = safe_count / total_tables
        
        # Reward based on safety ratio
        # All safe: +1.0, All unsafe: -1.0, Mixed: proportional
        rewards[b] = 2.0 * safe_ratio - 1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for combined table presence and safety.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    table_idx = labels_to_idx['table']
    desk_idx = labels_to_idx['desk']
    nightstand_idx = labels_to_idx['nightstand']
    
    # Scene 1: Safe table
    num_objects_1 = 2
    class_label_indices_1 = [kids_bed_idx, table_idx]
    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0)]
    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Unsafe table
    num_objects_2 = 2
    class_label_indices_2 = [kids_bed_idx, table_idx]
    translations_2 = [(0, 0.3, 0), (1.5, 0.375, 0)]
    sizes_2 = [(0.9, 0.3, 0.75), (0.4, 0.375, 0.4)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No table
    num_objects_3 = 2
    class_label_indices_3 = [kids_bed_idx, nightstand_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.3, 0)]
    sizes_3 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Scene 4: Mixed (one safe, one unsafe)
    num_objects_4 = 3
    class_label_indices_4 = [kids_bed_idx, table_idx, desk_idx]
    translations_4 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.375, 0)]
    sizes_4 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.5, 0.375, 0.5)]
    orientations_4 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
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
    print(f"Scene 1 (safe table): {rewards[0].item():.4f}")
    print(f"Scene 2 (unsafe table): {rewards[1].item():.4f}")
    print(f"Scene 3 (no table): {rewards[2].item():.4f}")
    print(f"Scene 4 (mixed): {rewards[3].item():.4f}")
    
    assert rewards.shape[0] == 4, "Should have 4 scenes"
    assert rewards[0].item() == 1.0, f"Scene 1: Safe table should return 1.0, got {rewards[0].item()}"
    assert rewards[1].item() == -1.0, f"Scene 2: Unsafe table should return -1.0, got {rewards[1].item()}"
    assert rewards[2].item() == 0.0, f"Scene 3: No table should return 0.0, got {rewards[2].item()}"
    assert abs(rewards[3].item()) < 0.1, f"Scene 4: Mixed should be close to 0.0, got {rewards[3].item()}"
    
    print("All tests passed!")