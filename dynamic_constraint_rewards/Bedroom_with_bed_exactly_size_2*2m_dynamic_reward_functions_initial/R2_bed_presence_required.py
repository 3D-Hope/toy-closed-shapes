import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function for bed presence.
    Returns +1.0 if at least one bed is present, -1.0 otherwise.
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: floor polygon vertices
        - **kwargs: additional keyword arguments
    
    Output:
        reward: torch.Tensor of shape (B,)
    '''
    device = parsed_scenes['device']
    B = parsed_scenes['positions'].shape[0]
    N = parsed_scenes['positions'].shape[1]
    
    # Get bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = ['double_bed', 'single_bed', 'kids_bed']
    bed_indices = [labels_to_idx[bed_class] for bed_class in bed_classes if bed_class in labels_to_idx]
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        has_bed = False
        
        for n in range(N):
            # Skip empty slots
            if parsed_scenes['is_empty'][b, n]:
                continue
            
            obj_idx = parsed_scenes['object_indices'][b, n].item()
            
            # Check if this is a bed
            if obj_idx in bed_indices:
                has_bed = True
                break
        
        # Assign reward
        rewards[b] = 1.0 if has_bed else -1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for bed_presence_required reward.
    '''
    pass
    return
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    single_bed_idx = labels_to_idx['single_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    chair_idx = labels_to_idx['chair']
    
    # Scene 1: Has double bed
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0, 0.5, 0), (2, 0.3, 0), (-2, 1, 0)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.8, 1.0, 0.4)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, 
                                       translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has single bed
    num_objects_2 = 2
    class_label_indices_2 = [single_bed_idx, nightstand_idx]
    translations_2 = [(0, 0.4, 0), (1.5, 0.3, 0)]
    sizes_2 = [(0.9, 0.4, 0.5), (0.3, 0.3, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, 
                                       translations_2, sizes_2, orientations_2)
    
    # Scene 3: No bed (only furniture)
    num_objects_3 = 3
    class_label_indices_3 = [wardrobe_idx, nightstand_idx, chair_idx]
    translations_3 = [(-2, 1, 0), (2, 0.3, 0), (0, 0.4, 0)]
    sizes_3 = [(0.8, 1.0, 0.4), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, 
                                       translations_3, sizes_3, orientations_3)
    
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
    print(f"Scene 1 (has double bed): {rewards[0].item():.4f}")
    print(f"Scene 2 (has single bed): {rewards[1].item():.4f}")
    print(f"Scene 3 (no bed): {rewards[2].item():.4f}")
    
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    
    # Scene 1: Should have reward of 1.0 (has double bed)
    assert rewards[0] == 1.0, f"Scene 1 should have reward 1.0, got {rewards[0].item()}"
    
    # Scene 2: Should have reward of 1.0 (has single bed)
    assert rewards[1] == 1.0, f"Scene 2 should have reward 1.0, got {rewards[1].item()}"
    
    # Scene 3: Should have reward of -1.0 (no bed)
    assert rewards[2] == -1.0, f"Scene 3 should have reward -1.0, got {rewards[2].item()}"
    
    print("All tests passed!")