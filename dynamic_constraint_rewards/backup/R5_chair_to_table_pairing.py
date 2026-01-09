import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for chair-table pairing - each chair should be within 1.0m of a table.
    
    Returns:
        reward: (B,) tensor - penalty for unpaired chairs
    '''
    utility_functions = get_all_utility_functions()
    distance_2d = utility_functions["distance_2d"]["function"]
    
    positions = parsed_scenes['positions']
    object_indices = parsed_scenes['object_indices']
    is_empty = parsed_scenes['is_empty']
    B, N = positions.shape[0], positions.shape[1]
    device = parsed_scenes['device']
    
    workspace_types = ['dining_table', 'desk', 'console_table']
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    
    workspace_indices = [labels_to_idx.get(wt, -1) for wt in workspace_types if labels_to_idx.get(wt, -1) != -1]
    seating_indices = [labels_to_idx.get(st, -1) for st in seating_types if labels_to_idx.get(st, -1) != -1]
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        workspace_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for idx in workspace_indices:
            workspace_mask |= (object_indices[b] == idx)
        workspace_mask &= ~is_empty[b]
        
        seat_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for idx in seating_indices:
            seat_mask |= (object_indices[b] == idx)
        seat_mask &= ~is_empty[b]
        
        workspace_positions = positions[b][workspace_mask]
        seat_positions = positions[b][seat_mask]
        num_seats = seat_mask.sum().item()
        
        if num_seats == 0 or workspace_positions.shape[0] == 0:
            rewards[b] = 0.0
            continue
        
        unpaired_chairs = 0
        for seat_pos in seat_positions:
            min_dist = float('inf')
            for workspace_pos in workspace_positions:
                dist = distance_2d(seat_pos, workspace_pos).item()
                min_dist = min(min_dist, dist)
            
            if min_dist > 1.0:
                unpaired_chairs += 1
        
        rewards[b] = -unpaired_chairs * 1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the chair_to_table_pairing reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx.get('dining_chair', 10)
    table_idx = labels_to_idx.get('dining_table', 11)
    
    # Scene 1: 4 chairs all within 1.0m of table (perfect pairing)
    num_objects_1 = 5
    class_label_indices_1 = [chair_idx, chair_idx, chair_idx, chair_idx, table_idx]
    translations_1 = [(0, 0.4, 0), (1.5, 0.4, 0), (0, 0.4, 0.8), (1.5, 0.4, 0.8), (0.75, 0.4, 0.4)]
    sizes_1 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.8, 0.4, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 4 chairs, 2 far from table (partial pairing)
    num_objects_2 = 5
    class_label_indices_2 = [chair_idx, chair_idx, chair_idx, chair_idx, table_idx]
    translations_2 = [(0, 0.4, 0), (1.5, 0.4, 0), (5, 0.4, 5), (6, 0.4, 5), (0.75, 0.4, 0.4)]
    sizes_2 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.8, 0.4, 0.5)]
    orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No tables (all chairs unpaired)
    num_objects_3 = 3
    class_label_indices_3 = [chair_idx, chair_idx, chair_idx]
    translations_3 = [(0, 0.4, 0), (1, 0.4, 0), (2, 0.4, 0)]
    sizes_3 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.3, 0.4, 0.3)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    assert rewards.shape[0] == 3
    
    assert rewards[0] >= -0.1, "Scene 1 should have no penalty (all chairs paired)"
    assert rewards[1] <= -1.5, "Scene 2 should have penalty (2 chairs unpaired)"
    assert rewards[2] == 0.0, "Scene 3 should have no penalty (no tables to pair with)"
    print("All tests passed for chair_to_table_pairing!")