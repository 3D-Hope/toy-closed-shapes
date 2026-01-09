import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for having at least 10 seating positions in the scene.
    Seating furniture includes: dining_chair, stool, armchair, lounge_chair, chinese_chair
    
    Returns:
        reward: (B,) tensor - 0 if >= 10 seats, negative penalty otherwise
    '''
    utility_functions = get_all_utility_functions()
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    B = one_hot.shape[0]
    device = parsed_scenes['device']
    
    # Define seating furniture types
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Count total seating capacity
        total_seats = 0
        for seat_type in seating_types:
            count = get_object_count(one_hot[b:b+1], seat_type, idx_to_labels)
            total_seats += count
        
        # Reward: 0 if >= 10 seats, else penalty proportional to shortfall
        if total_seats >= 10:
            rewards[b] = 0.0
        else:
            rewards[b] = -(10 - total_seats) * 1.0  # -1.0 per missing seat
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the minimum_seating_capacity reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Get indices for furniture
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx.get('dining_chair', 10)
    stool_idx = labels_to_idx.get('stool', 20)
    table_idx = labels_to_idx.get('dining_table', 11)
    
    # Scene 1: 10 chairs (exactly meets requirement)
    num_objects_1 = 10
    class_label_indices_1 = [chair_idx] * 10
    translations_1 = [(i, 0.4, 0) for i in range(10)]
    sizes_1 = [(0.3, 0.4, 0.3)] * 10
    orientations_1 = [(1, 0)] * 10
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 5 chairs + 6 stools = 11 (exceeds requirement)
    num_objects_2 = 11
    class_label_indices_2 = [chair_idx] * 5 + [stool_idx] * 6
    translations_2 = [(i, 0.4, 0) for i in range(11)]
    sizes_2 = [(0.3, 0.4, 0.3)] * 11
    orientations_2 = [(1, 0)] * 11
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Only 6 chairs (below requirement)
    num_objects_3 = 6
    class_label_indices_3 = [chair_idx] * 6
    translations_3 = [(i, 0.4, 0) for i in range(6)]
    sizes_3 = [(0.3, 0.4, 0.3)] * 6
    orientations_3 = [(1, 0)] * 6
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Stack scenes
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
    
    # Test assertions
    assert rewards[0] >= -0.1, "Scene 1 should have 0 penalty (10 chairs)"
    assert rewards[1] >= -0.1, "Scene 2 should have 0 penalty (11 seats)"
    assert rewards[2] < -3.5, "Scene 3 should have penalty (only 6 chairs)"
    print("All tests passed for minimum_seating_capacity!")