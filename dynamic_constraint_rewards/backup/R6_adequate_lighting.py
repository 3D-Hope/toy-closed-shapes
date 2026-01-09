import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for having adequate lighting (1-2 light sources minimum).
    
    Returns:
        reward: (B,) tensor - 0 if >= 1 light, negative penalty otherwise
    '''
    utility_functions = get_all_utility_functions()
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    one_hot = parsed_scenes['one_hot']
    B = one_hot.shape[0]
    device = parsed_scenes['device']
    
    lighting_types = ['pendant_lamp', 'ceiling_lamp']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        total_lights = 0
        for light_type in lighting_types:
            count = get_object_count(one_hot[b:b+1], light_type, idx_to_labels)
            total_lights += count
        
        if total_lights >= 1:
            rewards[b] = 0.0
        else:
            rewards[b] = -3.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the adequate_lighting reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    pendant_idx = labels_to_idx.get('pendant_lamp', 13)
    ceiling_idx = labels_to_idx.get('ceiling_lamp', 3)
    chair_idx = labels_to_idx.get('dining_chair', 10)
    
    # Scene 1: Has pendant lamp
    num_objects_1 = 2
    class_label_indices_1 = [pendant_idx, chair_idx]
    translations_1 = [(0, 2.5, 0), (1, 0.4, 0)]
    sizes_1 = [(0.2, 0.3, 0.2), (0.3, 0.4, 0.3)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has ceiling lamp
    num_objects_2 = 2
    class_label_indices_2 = [ceiling_idx, chair_idx]
    translations_2 = [(0, 2.7, 0), (1, 0.4, 0)]
    sizes_2 = [(0.3, 0.1, 0.3), (0.3, 0.4, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No lighting
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
    
    assert rewards[0] >= -0.1, "Scene 1 should have no penalty (has pendant lamp)"
    assert rewards[1] >= -0.1, "Scene 2 should have no penalty (has ceiling lamp)"
    assert rewards[2] <= -2.9, "Scene 3 should have penalty (no lighting)"
    print("All tests passed for adequate_lighting!")