import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that ensures non-ceiling furniture has height <= 1.2m.
    Returns proportion of non-ceiling furniture that satisfies height constraint.
    Normalized to [0, 1] where 1.0 means all furniture meets the constraint.
    '''
    device = parsed_scenes['device']
    sizes = parsed_scenes['sizes']  # (B, N, 3) - half-extents
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, _ = sizes.shape
    
    # Find ceiling lamp and pendant lamp indices
    ceiling_lamp_idx = None
    pendant_lamp_idx = None
    for idx, label in idx_to_labels.items():
        if label == 'ceiling_lamp':
            ceiling_lamp_idx = idx
        elif label == 'pendant_lamp':
            pendant_lamp_idx = idx
    
    # Full height = 2 * y_size (since sizes are half-extents)
    heights = 2 * sizes[:, :, 1]  # (B, N)
    
    # Create mask for non-empty, non-ceiling furniture
    non_ceiling_mask = ~is_empty  # Start with non-empty
    
    if ceiling_lamp_idx is not None:
        is_ceiling_lamp = one_hot[:, :, ceiling_lamp_idx].bool()
        non_ceiling_mask = non_ceiling_mask & ~is_ceiling_lamp
    
    if pendant_lamp_idx is not None:
        is_pendant_lamp = one_hot[:, :, pendant_lamp_idx].bool()
        non_ceiling_mask = non_ceiling_mask & ~is_pendant_lamp
    
    # Check height constraint (height <= 1.2m)
    max_height = 1.2
    satisfies_constraint = (heights <= max_height).float()  # (B, N)
    
    # Calculate reward for each scene
    rewards = torch.zeros(B, device=device)
    for b in range(B):
        non_ceiling_objects = non_ceiling_mask[b]
        if non_ceiling_objects.sum() > 0:
            # Proportion of non-ceiling furniture meeting constraint
            proportion = (satisfies_constraint[b] * non_ceiling_objects.float()).sum() / non_ceiling_objects.sum()
            rewards[b] = proportion
        else:
            # No non-ceiling furniture, constraint is satisfied
            rewards[b] = 1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: All furniture meets height constraint (should get reward 1.0)
    num_objects_1 = 4
    class_label_indices_1 = [11, 12, 5, 13]  # kids_bed, nightstand, children_cabinet, pendant_lamp
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.5, 3.0), (3.0, 2.5, 2.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4), (0.2, 0.1, 0.2)]  # Heights: 0.6m, 0.5m, 1.0m, (ceiling)
    orientations_1 = [(1, 0), (1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: One furniture violates constraint (should get reward < 1.0)
    num_objects_2 = 3
    class_label_indices_2 = [11, 20, 12]  # kids_bed, wardrobe (tall), nightstand
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 1.2, 1.0), (0.5, 0.25, 3.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.8, 1.2, 0.6), (0.3, 0.25, 0.3)]  # Heights: 0.6m, 2.4m (violates), 0.5m
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: All furniture violates constraint (should get reward 0.0)
    num_objects_3 = 2
    class_label_indices_3 = [20, 2]  # wardrobe, cabinet (both tall)
    translations_3 = [(1.0, 1.3, 1.0), (2.5, 1.1, 3.0)]
    sizes_3 = [(0.8, 1.3, 0.6), (0.6, 1.1, 0.5)]  # Heights: 2.6m, 2.2m (both violate)
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
    print("Expected: [1.0, ~0.67, 0.0]")
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert torch.isclose(rewards[0], torch.tensor(1.0), atol=0.01), f"Scene 1 should have reward 1.0, got {rewards[0]}"
    assert 0.6 < rewards[1] < 0.7, f"Scene 2 should have reward ~0.67 (2/3), got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(0.0), atol=0.01), f"Scene 3 should have reward 0.0, got {rewards[2]}"
    print("All tests passed for low_furniture_height!")