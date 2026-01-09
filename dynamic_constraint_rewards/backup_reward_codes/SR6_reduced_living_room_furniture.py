import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions, create_scene_for_testing


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Penalize presence of living room furniture: sofas, coffee_table, tv_stand.
    Reward: 0 if none present, otherwise -1 per item (max penalty -5)
    '''
    utility_functions = get_all_utility_functions()
    one_hot = parsed_scenes['one_hot']
    B = one_hot.shape[0]
    
    unwanted_types = ['multi_seat_sofa', 'loveseat_sofa', 'l_shaped_sofa', 'coffee_table', 'tv_stand']
    rewards = []
    
    for i in range(B):
        scene_one_hot = one_hot[i].unsqueeze(0)
        total_unwanted = 0
        
        for unwanted_type in unwanted_types:
            count = utility_functions['get_object_count_in_a_scene']['function'](
                scene_one_hot, unwanted_type, idx_to_labels
            )
            total_unwanted += count
        
        if total_unwanted == 0:
            reward_val = 0.0
        else:
            reward_val = -min(total_unwanted, 5)
        
        rewards.append(reward_val)
    
    reward = torch.tensor(rewards, device=parsed_scenes['device'], dtype=torch.float)
    return reward


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    import torch
    
    # Scene 1: No living room furniture
    num_objects_1 = 2
    class_label_indices_1 = [10, 11]  # chair + table
    translations_1 = [(1, 0.5, 1), (2, 0.4, 1)]
    sizes_1 = [(0.3, 0.5, 0.3), (0.8, 0.4, 0.6)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has 1 sofa and 1 coffee table
    num_objects_2 = 2
    class_label_indices_2 = [16, 6]  # multi_seat_sofa + coffee_table
    translations_2 = [(2, 0.4, 2), (3, 0.3, 3)]
    sizes_2 = [(1.0, 0.4, 0.6), (0.5, 0.3, 0.5)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has 3 unwanted items
    num_objects_3 = 3
    class_label_indices_3 = [16, 6, 21]  # sofa + coffee_table + tv_stand
    translations_3 = [(2, 0.4, 2), (3, 0.3, 3), (4, 0.5, 4)]
    sizes_3 = [(1.0, 0.4, 0.6), (0.5, 0.3, 0.5), (0.8, 0.5, 0.4)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0) for k in tensor_keys
    }
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Reduced Living Room Furniture Rewards:", rewards)
    
    assert torch.isclose(rewards[0], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[1], torch.tensor(-2.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[2], torch.tensor(-3.0, device=parsed_scenes['device']))
    
    return rewards
