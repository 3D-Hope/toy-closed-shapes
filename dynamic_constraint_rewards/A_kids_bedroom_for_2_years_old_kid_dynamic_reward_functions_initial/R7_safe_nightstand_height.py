import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that ensures nightstands have height <= 0.6m if present.
    Returns 1.0 if all nightstands satisfy constraint or no nightstands present,
    proportion of compliant nightstands otherwise.
    '''
    device = parsed_scenes['device']
    sizes = parsed_scenes['sizes']  # (B, N, 3) - half-extents
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, _ = sizes.shape
    
    # Find nightstand index
    nightstand_idx = None
    for idx, label in idx_to_labels.items():
        if label == 'nightstand':
            nightstand_idx = idx
            break
    
    if nightstand_idx is None:
        return torch.ones(B, device=device)
    
    # Full height = 2 * y_size
    heights = 2 * sizes[:, :, 1]  # (B, N)
    max_height = 0.6
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find nightstands in this scene
        is_nightstand = one_hot[b, :, nightstand_idx].bool() & ~is_empty[b]
        
        if is_nightstand.sum() == 0:
            # No nightstands, constraint is satisfied
            rewards[b] = 1.0
        else:
            # Check height constraint for nightstands
            nightstand_heights = heights[b, is_nightstand]
            satisfies_constraint = (nightstand_heights <= max_height).float()
            # Proportion of nightstands meeting constraint
            rewards[b] = satisfies_constraint.mean()
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Nightstand with safe height (should get reward 1.0)
    num_objects_1 = 2
    class_label_indices_1 = [11, 12]  # kids_bed, nightstand
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3)]  # Nightstand height = 0.5m
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Nightstand too tall (should get reward 0.0)
    num_objects_2 = 2
    class_label_indices_2 = [11, 12]  # kids_bed, nightstand
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 0.4, 1.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.3, 0.4, 0.3)]  # Nightstand height = 0.8m (too tall)
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No nightstand (should get reward 1.0)
    num_objects_3 = 2
    class_label_indices_3 = [11, 5]  # kids_bed, children_cabinet
    translations_3 = [(1.0, 0.3, 1.0), (2.5, 0.5, 3.0)]
    sizes_3 = [(0.8, 0.3, 1.0), (0.4, 0.5, 0.4)]
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
    print("Expected: [1.0, 0.0, 1.0]")
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert torch.isclose(rewards[0], torch.tensor(1.0), atol=0.01), f"Scene 1 should have reward 1.0, got {rewards[0]}"
    assert torch.isclose(rewards[1], torch.tensor(0.0), atol=0.01), f"Scene 2 should have reward 0.0, got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(1.0), atol=0.01), f"Scene 3 should have reward 1.0, got {rewards[2]}"
    print("All tests passed for safe_nightstand_height!")