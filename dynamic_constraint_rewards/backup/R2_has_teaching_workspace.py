import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for having at least one desk or console_table for teacher workspace.
    
    Returns:
        reward: (B,) tensor - 0 if has teaching workspace, -5.0 otherwise
    '''
    utility_functions = get_all_utility_functions()
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    B = one_hot.shape[0]
    device = parsed_scenes['device']
    
    # Teaching workspace furniture
    workspace_types = ['desk', 'console_table']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        has_workspace = False
        for workspace_type in workspace_types:
            count = get_object_count(one_hot[b:b+1], workspace_type, idx_to_labels)
            if count > 0:
                has_workspace = True
                break
        
        if has_workspace:
            rewards[b] = 0.0
        else:
            rewards[b] = -5.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the has_teaching_workspace reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    desk_idx = labels_to_idx.get('desk', 7)
    console_idx = labels_to_idx.get('console_table', 7)
    chair_idx = labels_to_idx.get('dining_chair', 10)
    
    # Scene 1: Has desk
    num_objects_1 = 2
    class_label_indices_1 = [desk_idx, chair_idx]
    translations_1 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_1 = [(0.6, 0.4, 0.4), (0.3, 0.4, 0.3)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has console_table
    num_objects_2 = 2
    class_label_indices_2 = [console_idx, chair_idx]
    translations_2 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_2 = [(0.6, 0.4, 0.3), (0.3, 0.4, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No desk or console_table
    num_objects_3 = 2
    class_label_indices_3 = [chair_idx, chair_idx]
    translations_3 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_3 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3)]
    orientations_3 = [(1, 0), (1, 0)]
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
    
    assert rewards[0] >= -0.1, "Scene 1 should have 0 penalty (has desk)"
    assert rewards[1] >= -0.1, "Scene 2 should have 0 penalty (has console_table)"
    assert rewards[2] <= -4.9, "Scene 3 should have penalty (no workspace)"
    print("All tests passed for has_teaching_workspace!")