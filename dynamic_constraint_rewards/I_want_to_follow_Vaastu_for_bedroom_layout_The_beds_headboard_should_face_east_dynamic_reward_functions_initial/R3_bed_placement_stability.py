import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function to ensure bed is placed correctly on the floor.
    
    The bed should be at y-position ≈ bed_height/2 (since position is centroid and size is half-extent).
    Reward is based on the deviation from expected floor placement.
    
    Returns:
        Reward proportional to correct placement. Max reward = 1.0 for perfect placement.
        Penalty increases with distance from correct y-position.
    '''
    
    device = parsed_scenes['device']
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    
    B, N = positions.shape[:2]
    
    # Get bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = [labels_to_idx['double_bed'], labels_to_idx['single_bed'], labels_to_idx['kids_bed']]
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    # Tolerance for floor placement (in meters)
    tolerance = 0.05  # 5cm tolerance
    
    for b in range(B):
        # Find bed objects in the scene
        bed_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for bed_class in bed_classes:
            bed_mask |= (object_indices[b] == bed_class) & (~is_empty[b])
        
        if not bed_mask.any():
            # No bed found, give penalty
            rewards[b] = -1.0
            continue
        
        # Process all beds in the scene (average reward across all beds)
        bed_indices = torch.where(bed_mask)[0]
        total_reward = 0.0
        
        for bed_idx in bed_indices:
            # Get bed position and size
            bed_y = positions[b, bed_idx, 1]  # y-coordinate of centroid
            bed_height_half = sizes[b, bed_idx, 1]  # half-height (sy/2)
            
            # Expected y-position: bed should be at y = bed_height/2
            # Since sizes are half-extents, full height = 2 * bed_height_half
            # So centroid should be at y = bed_height_half
            expected_y = bed_height_half
            
            # Calculate deviation from expected position
            deviation = torch.abs(bed_y - expected_y)
            
            # Calculate reward: exponential decay based on deviation
            # Perfect placement (deviation=0) -> reward=1.0
            # Large deviation -> reward approaches -1.0
            if deviation <= tolerance:
                bed_reward = 1.0
            else:
                # Exponential decay: reward = 1 - 2 * (1 - exp(-deviation))
                # This gives values from 1.0 (perfect) to -1.0 (very bad)
                normalized_dev = (deviation - tolerance) / 0.5  # Normalize to reasonable scale
                bed_reward = 1.0 - 2.0 * (1.0 - torch.exp(-normalized_dev))
                bed_reward = torch.clamp(bed_reward, -1.0, 1.0)
            
            total_reward += bed_reward
        
        # Average reward across all beds
        rewards[b] = total_reward / len(bed_indices)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for bed floor placement constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Bed correctly placed (y = bed_height/2 = 0.4)
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Correct: y = size_y = 0.4
    sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Bed floating (y = 1.0, should be 0.4)
    num_objects_2 = 3
    class_label_indices_2 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_2 = [(0, 1.0, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Floating: y = 1.0 instead of 0.4
    sizes_2 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Bed slightly off (y = 0.5, should be 0.4)
    num_objects_3 = 3
    class_label_indices_3 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_3 = [(0, 0.5, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]  # Slightly off: y = 0.5 instead of 0.4
    sizes_3 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
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
    print(f"Scene 1 (correct placement): {rewards[0].item():.4f}")
    print(f"Scene 2 (floating bed): {rewards[1].item():.4f}")
    print(f"Scene 3 (slightly off): {rewards[2].item():.4f}")
    
    assert rewards.shape[0] == 3, "Should have 3 scenes"
    
    # Test assertions
    assert rewards[0].item() > 0.95, f"Scene 1: Correct placement should have reward close to 1.0, got {rewards[0].item()}"
    assert rewards[1].item() < 0.0, f"Scene 2: Floating bed should have negative reward, got {rewards[1].item()}"
    assert 0.0 < rewards[2].item() < 1.0, f"Scene 3: Slightly off should have moderate positive reward, got {rewards[2].item()}"
    assert rewards[0] > rewards[2] > rewards[1], f"Scene 1 > Scene 3 > Scene 2, got {rewards[0]}, {rewards[2]}, {rewards[1]}"
    
    print("All tests passed!")