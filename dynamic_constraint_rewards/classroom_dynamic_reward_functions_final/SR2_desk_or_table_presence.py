import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Verifies that at least one desk or dining table exists.
    Reward: 0 if present, -5 if absent
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    device = parsed_scenes['device']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        desk_count = utility_functions['get_object_count_in_a_scene']['function'](
            parsed_scenes['one_hot'][b:b+1],
            'desk',
            idx_to_labels
        )
        
        dining_table_count = utility_functions['get_object_count_in_a_scene']['function'](
            parsed_scenes['one_hot'][b:b+1],
            'dining_table',
            idx_to_labels
        )
        
        total_count = desk_count + dining_table_count
        
        if total_count >= 1:
            rewards[b] = 0.0
        else:
            rewards[b] = -5.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions['create_scene_for_testing']['function']
    
    # Scene 1: Has dining table
    num_objects_1 = 2
    class_label_indices_1 = [11, 10]  # dining_table, dining_chair
    translations_1 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_1 = [(0.8, 0.4, 0.6), (0.25, 0.4, 0.25)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has desk
    num_objects_2 = 2
    class_label_indices_2 = [9, 10]  # desk, dining_chair
    translations_2 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_2 = [(0.6, 0.4, 0.4), (0.25, 0.4, 0.25)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No desk or table
    num_objects_3 = 2
    class_label_indices_3 = [10, 10]  # only chairs
    translations_3 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_3 = [(0.25, 0.4, 0.25), (0.25, 0.4, 0.25)]
    orientations_3 = [(1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print("Expected: [0.0 (has table), 0.0 (has desk), -5.0 (no table/desk)]")
    
    assert rewards.shape[0] == 3
    assert rewards[0] >= -0.1, f"Scene 1 should satisfy (has table): got {rewards[0]}"
    assert rewards[1] >= -0.1, f"Scene 2 should satisfy (has desk): got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(-5.0, device=rewards.device), atol=0.1), f"Scene 3 should be -5.0: got {rewards[2]}"
    print("All tests passed!")