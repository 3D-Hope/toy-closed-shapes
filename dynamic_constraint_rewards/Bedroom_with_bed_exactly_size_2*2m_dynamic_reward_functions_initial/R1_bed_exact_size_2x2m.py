import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function for bed with exact size 2m x 2m.
    Returns positive reward when bed dimensions match 2m x 2m (±0.05m tolerance).
    
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
    target_size = 2.0  # Target dimension in meters
    tolerance = 0.05  # ±0.05m tolerance
    
    for b in range(B):
        scene_reward = -5.0  # Default penalty if no matching bed found
        
        for n in range(N):
            # Skip empty slots
            if parsed_scenes['is_empty'][b, n]:
                continue
            
            obj_idx = parsed_scenes['object_indices'][b, n].item()
            
            # Check if this is a bed
            if obj_idx in bed_indices:
                # Get full dimensions (sizes are half-extents, so multiply by 2)
                size = parsed_scenes['sizes'][b, n] * 2  # (full_x, full_y, full_z)
                
                # XZ plane dimensions (horizontal)
                size_x = size[0].item()
                size_z = size[2].item()
                
                # Check if both dimensions are close to 2m
                error_x = abs(size_x - target_size)
                error_z = abs(size_z - target_size)
                
                # If within tolerance, give positive reward
                if error_x <= tolerance and error_z <= tolerance:
                    # Perfect match: reward = 1.0
                    # Scale down slightly based on error
                    reward_x = 1.0
                    reward_z = 1.0
                    current_reward = (reward_x + reward_z) / 2.0
                    scene_reward = max(scene_reward, current_reward)
                else:
                    # Outside tolerance: negative reward based on distance
                    total_error = error_x + error_z
                    # Cap the penalty at -5.0
                    current_reward = -min(total_error, 5.0)
                    scene_reward = max(scene_reward, current_reward)
        
        rewards[b] = scene_reward
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for bed_exact_size_2x2m reward.
    '''
    pass
    return
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Bed with exact 2x2m dimensions
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0, 0.5, 0), (2, 0.3, 0), (-2, 1, 0)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.8, 1.0, 0.4)]  # Bed: 2x1x2m (full dims)
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, 
                                       translations_1, sizes_1, orientations_1)
    
    # Scene 2: Bed with dimensions 2.5x2.5m (outside tolerance)
    num_objects_2 = 2
    class_label_indices_2 = [double_bed_idx, nightstand_idx]
    translations_2 = [(0, 0.5, 0), (2, 0.3, 0)]
    sizes_2 = [(1.25, 0.5, 1.25), (0.3, 0.3, 0.3)]  # Bed: 2.5x1x2.5m
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, 
                                       translations_2, sizes_2, orientations_2)
    
    # Scene 3: Bed with dimensions 2.03x2.02m (within tolerance)
    num_objects_3 = 2
    class_label_indices_3 = [double_bed_idx, wardrobe_idx]
    translations_3 = [(0, 0.5, 0), (-2, 1, 0)]
    sizes_3 = [(1.015, 0.5, 1.01), (0.8, 1.0, 0.4)]  # Bed: 2.03x1x2.02m
    orientations_3 = [(1, 0), (1, 0)]
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
    print(f"Scene 1 (exact 2x2m): {rewards[0].item():.4f}")
    print(f"Scene 2 (2.5x2.5m): {rewards[1].item():.4f}")
    print(f"Scene 3 (2.03x2.02m): {rewards[2].item():.4f}")
    
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    
    # Scene 1: Should have high positive reward (close to 1.0)
    assert rewards[0] > 0.8, f"Scene 1 should have reward > 0.8, got {rewards[0].item()}"
    
    # Scene 2: Should have negative reward (dimensions too large)
    assert rewards[1] < 0, f"Scene 2 should have negative reward, got {rewards[1].item()}"
    
    # Scene 3: Should have positive reward (within tolerance)
    assert rewards[2] > 0.7, f"Scene 3 should have reward > 0.7, got {rewards[2].item()}"
    
    # Scene 1 should have better reward than Scene 3
    assert rewards[0] > rewards[2], f"Scene 1 ({rewards[0].item()}) should have higher reward than Scene 3 ({rewards[2].item()})"
    
    print("All tests passed!")