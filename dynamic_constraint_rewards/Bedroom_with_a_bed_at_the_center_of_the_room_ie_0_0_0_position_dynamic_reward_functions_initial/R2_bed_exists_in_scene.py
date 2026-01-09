import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function for verifying at least one bed exists in the scene.
    
    Input:
        - parsed_scenes: dict with batched scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string, Example: "bedroom" or "livingroom"
        - floor_polygons: list of ordered floor polygon vertices
        - **kwargs: additional keyword arguments

    Output:
        reward: torch.Tensor of shape (B,) - 1.0 if bed exists, -1.0 if not
    '''
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    device = parsed_scenes['device']
    
    B, N = object_indices.shape
    
    # Find bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = []
    for bed_type in ['double_bed', 'single_bed', 'kids_bed']:
        if bed_type in labels_to_idx:
            bed_classes.append(labels_to_idx[bed_type])
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Check if any bed exists in this scene
        has_bed = False
        for bed_idx in bed_classes:
            bed_mask = (object_indices[b] == bed_idx) & ~is_empty[b]
            if bed_mask.any():
                has_bed = True
                break
        
        if has_bed:
            rewards[b] = 1.0
        else:
            rewards[b] = -1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the bed_exists_in_scene reward function.
    '''
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    single_bed_idx = labels_to_idx['single_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Has double_bed
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0.0, 0.4, 0.0), (2.0, 0.3, 2.0), (-2.0, 0.5, -2.0)]
    sizes_1 = [(1.0, 0.4, 1.0), (0.3, 0.3, 0.3), (0.5, 0.5, 0.8)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has single_bed
    num_objects_2 = 2
    class_label_indices_2 = [single_bed_idx, nightstand_idx]
    translations_2 = [(1.5, 0.3, 1.5), (2.5, 0.3, 2.5)]
    sizes_2 = [(0.8, 0.3, 0.9), (0.3, 0.3, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No bed, only other furniture
    num_objects_3 = 2
    class_label_indices_3 = [nightstand_idx, wardrobe_idx]
    translations_3 = [(1.0, 0.3, 1.0), (-1.0, 0.5, -1.0)]
    sizes_3 = [(0.3, 0.3, 0.3), (0.5, 0.5, 0.8)]
    orientations_3 = [(1, 0), (1, 0)]
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
    print(f"Scene 1 (has double_bed): {rewards[0].item()}")
    print(f"Scene 2 (has single_bed): {rewards[1].item()}")
    print(f"Scene 3 (no bed): {rewards[2].item()}")
    
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    
    # Scene 1: Has double_bed, should get reward 1.0
    assert rewards[0] == 1.0, f"Scene 1 (has double_bed) should have reward 1.0, got {rewards[0].item()}"
    
    # Scene 2: Has single_bed, should get reward 1.0
    assert rewards[1] == 1.0, f"Scene 2 (has single_bed) should have reward 1.0, got {rewards[1].item()}"
    
    # Scene 3: No bed, should get penalty -1.0
    assert rewards[2] == -1.0, f"Scene 3 (no bed) should have penalty -1.0, got {rewards[2].item()}"
    
    print("All tests passed!")