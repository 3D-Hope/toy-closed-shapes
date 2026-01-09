import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions, create_scene_for_testing


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Ensure the scene contains seating for at least 10 students.
    Count all seating furniture: dining_chair, stool, armchair, lounge_chair, chinese_chair
    Reward: 0 if count >= 10, otherwise penalty of -(10 - count)
    '''
    utility_functions = get_all_utility_functions()
    one_hot = parsed_scenes['one_hot']  # shape (B, N, num_classes)
    B = one_hot.shape[0]
    
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    rewards = []
    
    for i in range(B):
        scene_one_hot = one_hot[i].unsqueeze(0)  # (1, N, num_classes)
        total_seats = 0
        
        for seat_type in seating_types:
            count = utility_functions['get_object_count_in_a_scene']['function'](
                scene_one_hot, seat_type, idx_to_labels
            )
            total_seats += count
        
        if total_seats >= 10:
            reward_val = 0.0
        else:
            reward_val = -(10 - total_seats)
        
        rewards.append(reward_val)
    
    reward = torch.tensor(rewards, device=parsed_scenes['device'], dtype=torch.float)
    return reward


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    import torch
    
    # Scene 1: Exactly 10 chairs (satisfies constraint)
    num_objects_1 = 10
    class_label_indices_1 = [10] * 10  # dining_chair
    translations_1 = [(i*0.8, 0.5, 0) for i in range(10)]
    sizes_1 = [(0.3, 0.5, 0.3)] * 10
    orientations_1 = [(1, 0)] * 10
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Only 7 chairs (does not satisfy)
    num_objects_2 = 7
    class_label_indices_2 = [10] * 7
    translations_2 = [(i*0.8, 0.5, 0) for i in range(7)]
    sizes_2 = [(0.3, 0.5, 0.3)] * 7
    orientations_2 = [(1, 0)] * 7
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: 12 chairs (exceeds requirement)
    num_objects_3 = 12
    class_label_indices_3 = [10] * 12
    translations_3 = [(i*0.8, 0.5, 0) for i in range(12)]
    sizes_3 = [(0.3, 0.5, 0.3)] * 12
    orientations_3 = [(1, 0)] * 12
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0) for k in tensor_keys
    }
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Minimum Seating Capacity Rewards:", rewards)
    
    # Test cases
    assert torch.isclose(rewards[0], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[1], torch.tensor(-3.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[2], torch.tensor(0.0, device=parsed_scenes['device']))
    
    return rewards
