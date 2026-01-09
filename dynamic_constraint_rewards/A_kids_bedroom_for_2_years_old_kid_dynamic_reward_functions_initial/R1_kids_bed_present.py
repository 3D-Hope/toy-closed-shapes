import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that checks if at least one kids_bed is present in the scene.
    Returns 1.0 if kids_bed is present, 0.0 otherwise.
    '''
    device = parsed_scenes['device']
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, num_classes = one_hot.shape
    
    # Find kids_bed index
    # kids_bed_idx = None
    # for idx, label in idx_to_labels.items():
    #     if label == 'kids_bed':
    #         kids_bed_idx = idx
    #         break
    
    # if kids_bed_idx is None:
    #     return torch.zeros(B, device=device)
    
    # Check if kids_bed is present (not empty and class matches)
    # kids_bed_mask = one_hot[:, :, kids_bed_idx] * (~is_empty)  # (B, N)
    # has_kids_bed = (kids_bed_mask.sum(dim=1) > 0).float()  # (B,)
    
    rewards = torch.zeros(B, device=device)
    for b in range(B):
        utility_functions = get_all_utility_functions()
        get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
        kids_bed_count = get_object_count(one_hot[b:b+1], 'kids_bed', idx_to_labels)
        if kids_bed_count > 0:
            rewards[b] = 1.0
        else:
            rewards[b] = 0.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Has kids_bed (should get reward 1.0)
    num_objects_1 = 3
    class_label_indices_1 = [11, 12, 13]  # kids_bed, nightstand, pendant_lamp
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (3.0, 2.5, 2.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.2, 0.1, 0.2)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: No kids_bed (should get reward 0.0)
    num_objects_2 = 3
    class_label_indices_2 = [12, 20, 13]  # nightstand, wardrobe, pendant_lamp
    translations_2 = [(1.0, 0.25, 1.0), (2.5, 1.0, 1.0), (3.0, 2.5, 2.0)]
    sizes_2 = [(0.3, 0.25, 0.3), (0.8, 1.0, 0.6), (0.2, 0.1, 0.2)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has kids_bed among other furniture (should get reward 1.0)
    num_objects_3 = 4
    class_label_indices_3 = [11, 12, 5, 3]  # kids_bed, nightstand, children_cabinet, ceiling_lamp
    translations_3 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.5, 3.0), (3.0, 2.7, 2.0)]
    sizes_3 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4), (0.3, 0.1, 0.3)]
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
    print("Expected: [1.0, 0.0, 1.0]")
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert torch.isclose(rewards[0], torch.tensor(1.0)), f"Scene 1 should have reward 1.0, got {rewards[0]}"
    assert torch.isclose(rewards[1], torch.tensor(0.0)), f"Scene 2 should have reward 0.0, got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(1.0)), f"Scene 3 should have reward 1.0, got {rewards[2]}"
    print("All tests passed for kids_bed_present!")