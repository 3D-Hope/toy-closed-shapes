import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions, create_scene_for_testing


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Check for adequate lighting: at least 1 lighting fixture (pendant_lamp or ceiling_lamp).
    Reward: 0 if at least 1 light, otherwise -1
    '''
    utility_functions = get_all_utility_functions()
    one_hot = parsed_scenes['one_hot']
    B = one_hot.shape[0]
    
    lighting_types = ['pendant_lamp', 'ceiling_lamp']
    rewards = []
    
    for i in range(B):
        scene_one_hot = one_hot[i].unsqueeze(0)
        total_lights = 0
        
        for light_type in lighting_types:
            count = utility_functions['get_object_count_in_a_scene']['function'](
                scene_one_hot, light_type, idx_to_labels
            )
            total_lights += count
        
        if total_lights >= 1:
            reward_val = 0.0
        else:
            reward_val = -1.0
        
        rewards.append(reward_val)
    
    reward = torch.tensor(rewards, device=parsed_scenes['device'], dtype=torch.float)
    return reward


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    import torch
    
    # Scene 1: Has ceiling lamp
    num_objects_1 = 1
    class_label_indices_1 = [3]  # ceiling_lamp
    translations_1 = [(2, 2.8, 2)]
    sizes_1 = [(0.3, 0.3, 0.3)]
    orientations_1 = [(1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has pendant lamp
    num_objects_2 = 1
    class_label_indices_2 = [17]  # pendant_lamp
    translations_2 = [(2, 2.5, 2)]
    sizes_2 = [(0.2, 0.4, 0.2)]
    orientations_2 = [(1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No lighting
    num_objects_3 = 1
    class_label_indices_3 = [10]  # just a chair
    translations_3 = [(2, 0.5, 2)]
    sizes_3 = [(0.3, 0.5, 0.3)]
    orientations_3 = [(1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0) for k in tensor_keys
    }
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Adequate Lighting Rewards:", rewards)
    
    assert torch.isclose(rewards[0], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[1], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[2], torch.tensor(-1.0, device=parsed_scenes['device']))
    
    return rewards
