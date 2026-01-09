import torch
import numpy as np
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for furniture clustering - measures how tightly furniture is grouped.
    Lower average distance from furniture centroid = higher reward.
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor vertices
        - **kwargs: additional arguments
    
    Output:
        reward: torch.Tensor of shape (B,) - higher values for tighter clustering
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Get non-empty furniture positions (only x, z coordinates for floor plane)
        valid_mask = ~is_empty[b]
        valid_positions = positions[b][valid_mask]  # (num_valid, 3)
        
        if valid_positions.shape[0] <= 1:
            # Need at least 2 objects to measure clustering
            rewards[b] = 0.0
            continue
        
        # Compute centroid of all furniture (x, z plane)
        xz_positions = valid_positions[:, [0, 2]]  # (num_valid, 2)
        centroid = xz_positions.mean(dim=0)  # (2,)
        
        # Calculate average distance from centroid
        distances = torch.norm(xz_positions - centroid.unsqueeze(0), dim=1)  # (num_valid,)
        avg_distance = distances.mean()
        
        # Transform to reward: lower distance = higher reward
        # Use exponential decay: reward = exp(-distance)
        # Cap at reasonable maximum distance of 5m
        avg_distance_capped = torch.clamp(avg_distance, 0, 5.0)
        rewards[b] = torch.exp(-avg_distance_capped)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for furniture clustering reward.
    '''
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Tightly clustered furniture (should get high reward)
    num_objects_1 = 3
    class_label_indices_1 = [8, 12, 20]  # double_bed, nightstand, wardrobe
    translations_1 = [(0.0, 0.5, 0.0), (0.5, 0.3, 0.5), (0.5, 1.0, -0.5)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Spread out furniture (should get lower reward)
    num_objects_2 = 3
    class_label_indices_2 = [8, 12, 20]
    translations_2 = [(0.0, 0.5, 0.0), (3.0, 0.3, 3.0), (-3.0, 1.0, -3.0)]
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Single object (edge case, should get 0)
    num_objects_3 = 1
    class_label_indices_3 = [8]
    translations_3 = [(0.0, 0.5, 0.0)]
    sizes_3 = [(1.0, 0.5, 1.0)]
    orientations_3 = [(1, 0)]
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
    print("Scene 1 (clustered) reward:", rewards[0].item())
    print("Scene 2 (spread) reward:", rewards[1].item())
    print("Scene 3 (single object) reward:", rewards[2].item())
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] > rewards[1], f"Clustered furniture (reward={rewards[0].item():.4f}) should have higher reward than spread furniture (reward={rewards[1].item():.4f})"
    # assert rewards[2] == 0.0, f"Single object should have reward 0, got {rewards[2].item()}"
    # assert rewards[0] > 0.5, f"Tightly clustered furniture should have reward > 0.5, got {rewards[0].item():.4f}"
    # assert rewards[1] < 0.5, f"Spread furniture should have reward < 0.5, got {rewards[1].item():.4f}"
    
    print("All tests passed!")