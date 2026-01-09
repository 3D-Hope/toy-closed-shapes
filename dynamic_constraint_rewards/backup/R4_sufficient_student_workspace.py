import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for having adequate work surfaces near seating.
    Count tables/desks and verify they are within 1.5m of chairs.
    
    Returns:
        reward: (B,) tensor - penalty for insufficient workspace coverage
    '''
    utility_functions = get_all_utility_functions()
    distance_2d = utility_functions["distance_2d"]["function"]
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    positions = parsed_scenes['positions']  # (B, N, 3)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    B, N = positions.shape[0], positions.shape[1]
    device = parsed_scenes['device']
    
    # Workspace and seating furniture
    workspace_types = ['dining_table', 'desk', 'console_table']
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    
    workspace_indices = [labels_to_idx.get(wt, -1) for wt in workspace_types if labels_to_idx.get(wt, -1) != -1]
    seating_indices = [labels_to_idx.get(st, -1) for st in seating_types if labels_to_idx.get(st, -1) != -1]
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Count workspaces
        total_workspaces = 0
        for wt in workspace_types:
            total_workspaces += get_object_count(one_hot[b:b+1], wt, idx_to_labels)
        
        # Count seats
        total_seats = 0
        for st in seating_types:
            total_seats += get_object_count(one_hot[b:b+1], st, idx_to_labels)
        
        if total_seats == 0:
            rewards[b] = 0.0
            continue
        
        # Check workspace availability
        # Ideally: 1 workspace per 2-3 students (for group work)
        expected_workspaces = max(1, total_seats // 3)
        
        if total_workspaces < expected_workspaces:
            rewards[b] = -(expected_workspaces - total_workspaces) * 2.0
        else:
            # Now check proximity: are workspaces within 1.5m of seats?
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
            
            if workspace_positions.shape[0] > 0 and seat_positions.shape[0] > 0:
                # Check if each workspace is near at least one seat
                far_workspaces = 0
                for wp in workspace_positions:
                    min_dist = float('inf')
                    for sp in seat_positions:
                        dist = distance_2d(wp, sp).item()
                        min_dist = min(min_dist, dist)
                    if min_dist > 1.5:
                        far_workspaces += 1
                
                rewards[b] = -far_workspaces * 1.0
            else:
                rewards[b] = 0.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the sufficient_student_workspace reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx.get('dining_chair', 10)
    table_idx = labels_to_idx.get('dining_table', 11)
    desk_idx = labels_to_idx.get('desk', 7)
    
    # Scene 1: 6 chairs + 2 tables nearby (good)
    num_objects_1 = 8
    class_label_indices_1 = [chair_idx]*6 + [table_idx]*2
    translations_1 = [(0, 0.4, 0), (1, 0.4, 0), (2, 0.4, 0), (0, 0.4, 1), (1, 0.4, 1), (2, 0.4, 1), (0.5, 0.4, 0.5), (1.5, 0.4, 0.5)]
    sizes_1 = [(0.3, 0.4, 0.3)]*6 + [(0.8, 0.4, 0.5)]*2
    orientations_1 = [(1, 0)]*8
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 9 chairs but no tables (bad)
    num_objects_2 = 9
    class_label_indices_2 = [chair_idx]*9
    translations_2 = [(i, 0.4, 0) for i in range(9)]
    sizes_2 = [(0.3, 0.4, 0.3)]*9
    orientations_2 = [(1, 0)]*9
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: 6 chairs + 1 desk nearby (okay)
    num_objects_3 = 7
    class_label_indices_3 = [chair_idx]*6 + [desk_idx]
    translations_3 = [(0, 0.4, 0), (1, 0.4, 0), (2, 0.4, 0), (0, 0.4, 1), (1, 0.4, 1), (2, 0.4, 1), (1, 0.4, 0.5)]
    sizes_3 = [(0.3, 0.4, 0.3)]*6 + [(0.8, 0.4, 0.5)]
    orientations_3 = [(1, 0)]*7
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
    
    assert rewards[0] >= -1.0, "Scene 1 should have minimal penalty (good workspace)"
    assert rewards[1] <= -3.0, "Scene 2 should have penalty (no tables)"
    assert rewards[2] >= -2.0, "Scene 3 should have acceptable workspace"
    print("All tests passed for sufficient_student_workspace!")