import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Ensures at least 10 seating positions are available.
    Counts: dining_chair (1 seat), armchair (1), lounge_chair (1), stool (1),
    chinese_chair (1), multi_seat_sofa (3), loveseat_sofa (2), l_shaped_sofa (4),
    chaise_longue_sofa (2), lazy_sofa (1)
    
    Reward: 0 if >= 10 seats, negative penalty proportional to shortage otherwise
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    device = parsed_scenes['device']
    
    # Seating capacity map
    seating_capacity = {
        'dining_chair': 1,
        'armchair': 1,
        'lounge_chair': 1,
        'stool': 1,
        'chinese_chair': 1,
        'multi_seat_sofa': 3,
        'loveseat_sofa': 2,
        'l_shaped_sofa': 4,
        'chaise_longue_sofa': 2,
        'lazy_sofa': 1
    }
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        total_seats = 0
        
        for furniture_type, capacity in seating_capacity.items():
            count = utility_functions['get_object_count_in_a_scene']['function'](
                parsed_scenes['one_hot'][b:b+1],
                furniture_type,
                idx_to_labels
            )
            total_seats += count * capacity
        
        # Target is 10 seats
        if total_seats >= 10:
            rewards[b] = 0.0
        else:
            # Penalty proportional to shortage, capped at -10
            shortage = 10 - total_seats
            rewards[b] = -min(shortage * 1.0, 10.0)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions['create_scene_for_testing']['function']
    
    # Scene 1: Exactly 10 dining chairs (satisfies constraint)
    num_objects_1 = 10
    class_label_indices_1 = [10] * 10  # dining_chair
    translations_1 = [(i*1.5, 0.4, 0) for i in range(10)]
    sizes_1 = [(0.25, 0.4, 0.25)] * 10
    orientations_1 = [(1, 0)] * 10
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 2 multi_seat_sofas (2*3=6 seats) + 4 dining chairs = 10 seats
    num_objects_2 = 6
    class_label_indices_2 = [16, 16, 10, 10, 10, 10]  # multi_seat_sofa x2, dining_chair x4
    translations_2 = [(0, 0.5, 0), (2, 0.5, 0), (4, 0.4, 0), (5, 0.4, 0), (6, 0.4, 0), (7, 0.4, 0)]
    sizes_2 = [(1.0, 0.5, 0.5), (1.0, 0.5, 0.5), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25)]
    orientations_2 = [(1, 0)] * 6
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Only 5 dining chairs (insufficient)
    num_objects_3 = 5
    class_label_indices_3 = [10] * 5
    translations_3 = [(i*1.5, 0.4, 0) for i in range(5)]
    sizes_3 = [(0.25, 0.4, 0.25)] * 5
    orientations_3 = [(1, 0)] * 5
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
    print("Expected: [0.0 (10 seats), 0.0 (10 seats), -5.0 (5 seats shortage)]")
    
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert rewards[0] >= -0.1, f"Scene 1 should satisfy (10 chairs): got {rewards[0]}"
    assert rewards[1] >= -0.1, f"Scene 2 should satisfy (10 seats): got {rewards[1]}"
    assert rewards[2] < -4.0 and rewards[2] > -6.0, f"Scene 3 should have ~-5 penalty (5 seats): got {rewards[2]}"
    print("All tests passed!")