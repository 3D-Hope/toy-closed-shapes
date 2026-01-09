import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that encourages children_cabinet when storage furniture is present.
    Returns 1.0 if children_cabinet is present when any cabinet/wardrobe exists,
    1.0 if no storage furniture exists, 0.0 otherwise.
    '''
    device = parsed_scenes['device']
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, num_classes = one_hot.shape
    
    # Find relevant furniture indices
    children_cabinet_idx = None
    cabinet_idx = None
    wardrobe_idx = None
    
    for idx, label in idx_to_labels.items():
        if label == 'children_cabinet':
            children_cabinet_idx = idx
        elif label == 'cabinet':
            cabinet_idx = idx
        elif label == 'wardrobe':
            wardrobe_idx = idx
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Check if children_cabinet is present
        has_children_cabinet = False
        if children_cabinet_idx is not None:
            has_children_cabinet = (one_hot[b, :, children_cabinet_idx] * ~is_empty[b]).sum() > 0
        
        # Check if any adult storage furniture is present
        has_adult_storage = False
        if cabinet_idx is not None:
            has_adult_storage = has_adult_storage or ((one_hot[b, :, cabinet_idx] * ~is_empty[b]).sum() > 0)
        if wardrobe_idx is not None:
            has_adult_storage = has_adult_storage or ((one_hot[b, :, wardrobe_idx] * ~is_empty[b]).sum() > 0)
        
        # Reward logic
        if not has_adult_storage:
            # No adult storage, constraint is satisfied
            rewards[b] = 1.0
        elif has_children_cabinet:
            # Has adult storage but also has children_cabinet
            rewards[b] = 1.0
        else:
            # Has adult storage but no children_cabinet
            rewards[b] = 0.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Has children_cabinet with wardrobe (should get reward 1.0)
    num_objects_1 = 3
    class_label_indices_1 = [11, 5, 20]  # kids_bed, children_cabinet, wardrobe
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.5, 1.0), (0.5, 1.0, 3.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.4, 0.5, 0.4), (0.8, 1.0, 0.6)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has wardrobe but no children_cabinet (should get reward 0.0)
    num_objects_2 = 3
    class_label_indices_2 = [11, 20, 12]  # kids_bed, wardrobe, nightstand
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 1.0, 1.0), (0.5, 0.25, 3.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.8, 1.0, 0.6), (0.3, 0.25, 0.3)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No storage furniture at all (should get reward 1.0)
    num_objects_3 = 3
    class_label_indices_3 = [11, 12, 13]  # kids_bed, nightstand, pendant_lamp
    translations_3 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (3.0, 2.5, 2.0)]
    sizes_3 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.2, 0.1, 0.2)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
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
    print("All tests passed for children_storage_preference!")