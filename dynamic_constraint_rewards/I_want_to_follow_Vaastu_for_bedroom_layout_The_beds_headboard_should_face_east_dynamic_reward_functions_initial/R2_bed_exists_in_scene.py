import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function to verify at least one bed exists in the scene.
    
    Returns:
        1.0 if at least one bed (double_bed, single_bed, or kids_bed) exists
        -1.0 if no bed exists
    '''
    
    device = parsed_scenes['device']
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    
    B = object_indices.shape[0]
    
    # Get bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = [labels_to_idx['double_bed'], labels_to_idx['single_bed'], labels_to_idx['kids_bed']]
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Check if any bed exists
        has_bed = False
        for bed_class in bed_classes:
            bed_mask = (object_indices[b] == bed_class) & (~is_empty[b])
            if bed_mask.any():
                has_bed = True
                break
        
        rewards[b] = 1.0 if has_bed else -1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for bed existence constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    single_bed_idx = labels_to_idx['single_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    chair_idx = labels_to_idx['chair']
    
    # Scene 1: Has double bed
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has single bed
    num_objects_2 = 3
    class_label_indices_2 = [single_bed_idx, nightstand_idx, wardrobe_idx]
    translations_2 = [(0, 0.3, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    sizes_2 = [(0.6, 0.3, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No bed (only furniture)
    num_objects_3 = 3
    class_label_indices_3 = [nightstand_idx, wardrobe_idx, chair_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.5, 0), (-1.5, 0.4, 0)]
    sizes_3 = [(0.3, 0.3, 0.3), (0.5, 0.5, 1.0), (0.4, 0.4, 0.4)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
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
    print(f"Scene 1 (has double bed): {rewards[0].item()}")
    print(f"Scene 2 (has single bed): {rewards[1].item()}")
    print(f"Scene 3 (no bed): {rewards[2].item()}")
    
    assert rewards.shape[0] == 3, "Should have 3 scenes"
    
    # Test assertions
    assert rewards[0].item() == 1.0, f"Scene 1: Has double bed, should return 1.0, got {rewards[0].item()}"
    assert rewards[1].item() == 1.0, f"Scene 2: Has single bed, should return 1.0, got {rewards[1].item()}"
    assert rewards[2].item() == -1.0, f"Scene 3: No bed, should return -1.0, got {rewards[2].item()}"
    
    print("All tests passed!")