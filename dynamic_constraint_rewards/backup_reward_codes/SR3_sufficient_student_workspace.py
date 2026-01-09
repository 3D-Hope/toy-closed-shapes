import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions, create_scene_for_testing


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Ensure there are at least 2 work surfaces (dining_table, desk, console_table) for students.
    Reward: 0 if count >= 2, otherwise -(2 - count)
    '''
    utility_functions = get_all_utility_functions()
    one_hot = parsed_scenes['one_hot']
    B = one_hot.shape[0]
    
    workspace_types = ['dining_table', 'desk', 'console_table']
    rewards = []
    
    for i in range(B):
        scene_one_hot = one_hot[i].unsqueeze(0)
        total_tables = 0
        
        for table_type in workspace_types:
            count = utility_functions['get_object_count_in_a_scene']['function'](
                scene_one_hot, table_type, idx_to_labels
            )
            total_tables += count
        
        if total_tables >= 2:
            reward_val = 0.0
        else:
            reward_val = -(2 - total_tables)
        
        rewards.append(reward_val)
    
    reward = torch.tensor(rewards, device=parsed_scenes['device'], dtype=torch.float)
    return reward


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    import torch
    
    # Scene 1: Has 2 dining tables (satisfies)
    num_objects_1 = 2
    class_label_indices_1 = [11, 11]  # dining_table
    translations_1 = [(1, 0.4, 1), (3, 0.4, 1)]
    sizes_1 = [(0.8, 0.4, 0.6), (0.8, 0.4, 0.6)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has only 1 table
    num_objects_2 = 1
    class_label_indices_2 = [11]
    translations_2 = [(2, 0.4, 2)]
    sizes_2 = [(0.8, 0.4, 0.6)]
    orientations_2 = [(1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has 3 tables (exceeds)
    num_objects_3 = 3
    class_label_indices_3 = [11, 11, 9]  # 2 dining_table + 1 desk
    translations_3 = [(1, 0.4, 1), (3, 0.4, 1), (5, 0.4, 1)]
    sizes_3 = [(0.8, 0.4, 0.6), (0.8, 0.4, 0.6), (0.6, 0.4, 0.4)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0) for k in tensor_keys
    }
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Sufficient Student Workspace Rewards:", rewards)
    
    assert torch.isclose(rewards[0], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[1], torch.tensor(-1.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[2], torch.tensor(0.0, device=parsed_scenes['device']))
    
    return rewards
