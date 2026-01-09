import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Penalize presence of living room furniture that doesn't belong in classroom.
    
    Returns:
        reward: (B,) tensor - penalty for each living room furniture item
    '''
    utility_functions = get_all_utility_functions()
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    one_hot = parsed_scenes['one_hot']
    B = one_hot.shape[0]
    device = parsed_scenes['device']
    
    living_room_types = ['multi_seat_sofa', 'loveseat_sofa', 'l_shaped_sofa', 'chaise_longue_sofa', 
                         'lazy_sofa', 'coffee_table', 'tv_stand', 'lounge_chair']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        total_living_room_items = 0
        for lr_type in living_room_types:
            count = get_object_count(one_hot[b:b+1], lr_type, idx_to_labels)
            total_living_room_items += count
        
        rewards[b] = -total_living_room_items * 2.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the minimal_living_room_furniture reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx.get('dining_chair', 10)
    table_idx = labels_to_idx.get('dining_table', 11)
    sofa_idx = labels_to_idx.get('multi_seat_sofa', 16)
    coffee_idx = labels_to_idx.get('coffee_table', 6)
    tv_idx = labels_to_idx.get('tv_stand', 21)
    
    # Scene 1: No living room furniture (good)
    num_objects_1 = 3
    class_label_indices_1 = [chair_idx, chair_idx, table_idx]
    translations_1 = [(0, 0.4, 0), (1, 0.4, 0), (0.5, 0.4, 0.5)]
    sizes_1 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (0.8, 0.4, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has 1 sofa (bad)
    num_objects_2 = 3
    class_label_indices_2 = [chair_idx, sofa_idx, table_idx]
    translations_2 = [(0, 0.4, 0), (2, 0.5, 0), (0.5, 0.4, 0.5)]
    sizes_2 = [(0.3, 0.4, 0.3), (1.0, 0.5, 0.8), (0.8, 0.4, 0.5)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has multiple living room items (worse)
    num_objects_3 = 4
    class_label_indices_3 = [sofa_idx, coffee_idx, tv_idx, chair_idx]
    translations_3 = [(0, 0.5, 0), (1, 0.3, 0), (2, 0.4, 0), (3, 0.4, 0)]
    sizes_3 = [(1.0, 0.5, 0.8), (0.5, 0.3, 0.5), (0.8, 0.5, 0.4), (0.3, 0.4, 0.3)]
    orientations_3 = [(1, 0), (1, 0), (1, 0), (1, 0)]
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
    
    assert rewards[0] >= -0.1, "Scene 1 should have no penalty (no living room furniture)"
    assert rewards[1] <= -1.9, "Scene 2 should have penalty (1 sofa)"
    assert rewards[2] <= -5.9, "Scene 3 should have higher penalty (3 living room items)"
    print("All tests passed for minimal_living_room_furniture!")