import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Gradient reward for table height safety with smoother learning curve.
    
    Uses a continuous reward function:
    - Tables > 1.15m: reward = 1.0 (very safe)
    - Tables 1.1-1.15m: reward = 0.5-1.0 (safe)
    - Tables 0.9-1.1m: reward = 0.0-0.5 (marginal)
    - Tables 0.75-0.9m: reward = -0.3-0.0 (standard but unsafe)
    - Tables < 0.75m: reward = -0.5 (very unsafe)
    - No tables: reward = 0.0 (neutral, not applicable)
    
    This gradient makes it easier for the model to learn incrementally.
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
        # Find table-type objects in the scene
        table_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for table_class in table_classes:
            table_mask |= (object_indices[b] == table_class) & (~is_empty[b])
        
        if not table_mask.any():
            # No tables - neutral (constraint not applicable)
            rewards[b] = 0.0
            continue
        
        # Process all tables in the scene
        table_indices = torch.where(table_mask)[0]
        total_reward = 0.0
        
        for table_idx in table_indices:
            # Calculate table top height
            table_centroid_y = positions[b, table_idx, 1]
            table_half_height = sizes[b, table_idx, 1]
            table_top_y = table_centroid_y + table_half_height
            
            # Gradient reward based on height
            if table_top_y >= 1.15:
                # Very safe height
                table_reward = 1.0
            elif table_top_y >= 1.1:
                # Safe height - linear interpolation
                table_reward = 0.5 + (table_top_y - 1.1) / 0.05 * 0.5
            elif table_top_y >= 0.9:
                # Marginal height - linear interpolation
                table_reward = (table_top_y - 0.9) / 0.2 * 0.5
            elif table_top_y >= 0.75:
                # Standard table height but unsafe for toddlers
                table_reward = -0.3 + (table_top_y - 0.75) / 0.15 * 0.3
            else:
                # Very low table - very unsafe
                table_reward = -0.5
            
            total_reward += table_reward
        
        # Average reward across all tables
        rewards[b] = total_reward / len(table_indices)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for table height safety gradient constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    table_idx = labels_to_idx['table']
    nightstand_idx = labels_to_idx['nightstand']
    
    # Scene 1: Very safe table (top at 1.5m)
    num_objects_1 = 3
    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Standard table height (top at 0.75m - typical)
    num_objects_2 = 3
    class_label_indices_2 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_2 = [(0, 0.3, 0), (1.5, 0.375, 0), (-1.5, 0.3, 0)]
    sizes_2 = [(0.9, 0.3, 0.75), (0.4, 0.375, 0.4), (0.3, 0.3, 0.3)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No table
    num_objects_3 = 2
    class_label_indices_3 = [kids_bed_idx, nightstand_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.3, 0)]
    sizes_3 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Scene 4: Marginal table height (top at 1.0m)
    num_objects_4 = 2
    class_label_indices_4 = [kids_bed_idx, table_idx]
    translations_4 = [(0, 0.3, 0), (1.5, 0.5, 0)]
    sizes_4 = [(0.9, 0.3, 0.75), (0.4, 0.5, 0.4)]
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
    print(f"Scene 1 (very safe table 1.5m): {rewards[0].item():.4f}")
    print(f"Scene 2 (standard table 0.75m): {rewards[1].item():.4f}")
    print(f"Scene 3 (no table): {rewards[2].item():.4f}")
    print(f"Scene 4 (marginal table 1.0m): {rewards[3].item():.4f}")
    
    assert rewards.shape[0] == 4, "Should have 4 scenes"
    assert rewards[0].item() > 0.95, f"Scene 1: Very safe table should have reward close to 1.0, got {rewards[0].item()}"
    assert -0.35 <= rewards[1].item() <= -0.25, f"Scene 2: Standard table should have reward around -0.3, got {rewards[1].item()}"
    assert rewards[2].item() == 0.0, f"Scene 3: No table should return 0.0, got {rewards[2].item()}"
    assert 0.2 <= rewards[3].item() <= 0.3, f"Scene 4: Marginal table should have reward around 0.25, got {rewards[3].item()}"
    assert rewards[0] > rewards[3] > rewards[2] > rewards[1], f"Expected order: Scene 1 > Scene 4 > Scene 3 > Scene 2, got {rewards[0]:.3f}, {rewards[3]:.3f}, {rewards[2]:.3f}, {rewards[1]:.3f}"
    
    print("All tests passed!")