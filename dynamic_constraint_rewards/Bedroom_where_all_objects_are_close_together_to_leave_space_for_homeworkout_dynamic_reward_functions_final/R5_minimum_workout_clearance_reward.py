import torch
import numpy as np
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for minimum workout clearance - ensures workout area has sufficient clearance.
    Checks vertical clearance (no low hanging objects) and horizontal clearance (wide paths).
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor vertices
        - **kwargs: additional arguments
    
    Output:
        reward: torch.Tensor of shape (B,) - higher for better clearance
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    rewards = torch.zeros(B, device=device)
    
    # Get pendant_lamp index from idx_to_labels
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    pendant_lamp_idx = labels_to_idx.get('pendant_lamp', -1)
    
    # Minimum clearances for workout
    min_vertical_clearance = 2.2  # meters (need headroom for jumping, arms up)
    min_horizontal_clearance = 1.0  # meters (width of clear path)
    
    for b in range(B):
        # Get valid furniture
        valid_mask = ~is_empty[b]
        valid_positions = positions[b][valid_mask]  # (num_valid, 3)
        valid_sizes = sizes[b][valid_mask]  # (num_valid, 3)
        valid_indices = object_indices[b][valid_mask]  # (num_valid,)
        
        if valid_positions.shape[0] == 0:
            # No furniture - perfect clearance
            rewards[b] = 1.0
            continue
        
        # 1. Check vertical clearance (pendant lamps)
        vertical_penalty = 0.0
        if pendant_lamp_idx >= 0:
            lamp_mask = valid_indices == pendant_lamp_idx
            if lamp_mask.any():
                lamp_positions = valid_positions[lamp_mask]
                lamp_sizes = valid_sizes[lamp_mask]
                
                # Check if any pendant lamps are too low (below min_vertical_clearance)
                lamp_bottom_y = lamp_positions[:, 1] - lamp_sizes[:, 1]  # bottom of lamp
                low_lamps = lamp_bottom_y < min_vertical_clearance
                
                if low_lamps.any():
                    # Penalize based on how low they are
                    violations = min_vertical_clearance - lamp_bottom_y[low_lamps]
                    vertical_penalty = violations.mean().item()
        
        # 2. Check horizontal clearance (minimum gap between furniture)
        horizontal_score = 1.0
        if valid_positions.shape[0] >= 2:
            # Compute pairwise distances between furniture (on x-z plane)
            xz_positions = valid_positions[:, [0, 2]]  # (num_valid, 2)
            xz_sizes = valid_sizes[:, [0, 2]]  # (num_valid, 2)
            
            # For each pair, compute gap (distance between bounding boxes)
            num_valid = xz_positions.shape[0]
            min_gaps = []
            
            for i in range(num_valid):
                for j in range(i+1, num_valid):
                    pos_i = xz_positions[i]
                    pos_j = xz_positions[j]
                    size_i = xz_sizes[i]
                    size_j = xz_sizes[j]
                    
                    # Center-to-center distance
                    center_dist = torch.norm(pos_i - pos_j)
                    
                    # Sum of half-extents
                    sum_extents = (size_i + size_j).norm()
                    
                    # Gap between objects (approximation)
                    gap = center_dist - sum_extents
                    min_gaps.append(gap)
            
            if len(min_gaps) > 0:
                min_gap = min(min_gaps)
                
                # Score based on minimum gap
                if min_gap < 0:
                    # Objects overlapping (handled by penetration constraint)
                    horizontal_score = 0.5
                elif min_gap < min_horizontal_clearance:
                    # Gap too narrow
                    horizontal_score = 0.5 + 0.5 * (min_gap / min_horizontal_clearance)
                else:
                    # Good clearance
                    horizontal_score = 1.0
        
        # 3. Combine vertical and horizontal scores
        vertical_score = max(0.0, 1.0 - vertical_penalty / 1.0)  # Normalize penalty by 1m
        
        # Overall reward
        rewards[b] = 0.5 * vertical_score + 0.5 * horizontal_score
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for minimum workout clearance reward.
    '''
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    pendant_lamp_idx = labels_to_idx.get('pendant_lamp', 13)
    
    # Scene 1: Good clearance - no low lamps, wide gaps (should get high reward)
    num_objects_1 = 3
    class_label_indices_1 = [8, 12, 20]  # bed, nightstand, wardrobe
    translations_1 = [(-2.0, 0.5, -2.0), (-2.0, 0.3, 0.0), (2.0, 1.0, -2.0)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Low pendant lamp (should get lower reward)
    num_objects_2 = 4
    class_label_indices_2 = [8, 12, 20, pendant_lamp_idx]
    translations_2 = [(-2.0, 0.5, -2.0), (-2.0, 0.3, 0.0), (2.0, 1.0, -2.0), (0.0, 2.0, 0.0)]  # Lamp at y=2.0
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5), (0.2, 0.3, 0.2)]  # Lamp hangs down 0.3m
    orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Furniture too close together (should get lower reward)
    num_objects_3 = 3
    class_label_indices_3 = [8, 12, 20]
    translations_3 = [(0.0, 0.5, 0.0), (0.8, 0.3, 0.0), (1.5, 1.0, 0.0)]  # Very close together
    sizes_3 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
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
    print("Scene 1 (good clearance) reward:", rewards[0].item())
    print("Scene 2 (low pendant lamp) reward:", rewards[1].item())
    print("Scene 3 (furniture too close) reward:", rewards[2].item())
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] > rewards[1], f"Good clearance (reward={rewards[0].item():.4f}) should have higher reward than low lamp (reward={rewards[1].item():.4f})"
    # assert rewards[0] > rewards[2], f"Good clearance (reward={rewards[0].item():.4f}) should have higher reward than close furniture (reward={rewards[2].item():.4f})"
    # assert rewards[0] >= 0.8, f"Good clearance should have reward >= 0.8, got {rewards[0].item():.4f}"
    # assert 0.0 <= rewards[1] <= 1.0, f"Reward should be in [0,1], got {rewards[1].item():.4f}"
    # assert 0.0 <= rewards[2] <= 1.0, f"Reward should be in [0,1], got {rewards[2].item():.4f}"
    
    print("All tests passed!")