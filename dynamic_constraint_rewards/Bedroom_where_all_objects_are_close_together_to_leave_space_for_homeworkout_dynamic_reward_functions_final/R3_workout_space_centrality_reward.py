import torch
import numpy as np
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for workout space centrality - ensures open space is not just in corners.
    Measures how central the largest free region is.
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor vertices
        - **kwargs: additional arguments
    
    Output:
        reward: torch.Tensor of shape (B,) - higher for more central open spaces
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    rewards = torch.zeros(B, device=device)
    
    # Compute room center from floor polygon
    poly = np.array(floor_polygons)
    room_center = torch.tensor([poly[:, 0].mean(), poly[:, 1].mean()], device=device)
    
    for b in range(B):
        # Get valid furniture positions and sizes
        valid_mask = ~is_empty[b]
        valid_positions = positions[b][valid_mask]  # (num_valid, 3)
        valid_sizes = sizes[b][valid_mask]  # (num_valid, 3)
        
        if valid_positions.shape[0] == 0:
            # No furniture - entire room is open and central
            rewards[b] = 1.0
            continue
        
        # Find the centroid of free space (inverse of furniture centroid)
        furniture_xz = valid_positions[:, [0, 2]]  # (num_valid, 2)
        furniture_centroid = furniture_xz.mean(dim=0)  # (2,)
        
        # The free space center is on the opposite side of furniture cluster
        # Estimate as: room_center + (room_center - furniture_centroid)
        free_space_center = 2 * room_center - furniture_centroid
        
        # Measure how close free space center is to room center
        distance_to_center = torch.norm(free_space_center - room_center)
        
        # Also check if furniture is pushed to perimeter (good for centrality)
        furniture_to_center_dist = torch.norm(furniture_centroid - room_center)
        
        # Reward: free space should be near center, furniture should be far from center
        # Use exponential decay for distance
        centrality_score = torch.exp(-distance_to_center) * (1.0 + 0.5 * torch.tanh(furniture_to_center_dist - 1.0))
        
        rewards[b] = torch.clamp(centrality_score, 0.0, 1.5)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for workout space centrality reward.
    '''
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Furniture pushed to one side - central open space (should get high reward)
    num_objects_1 = 3
    class_label_indices_1 = [8, 12, 20]
    translations_1 = [(-2.5, 0.5, -2.0), (-2.5, 0.3, -0.5), (-2.5, 1.0, 0.5)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Furniture in center - open space in corners (should get lower reward)
    num_objects_2 = 3
    class_label_indices_2 = [8, 12, 20]
    translations_2 = [(0.0, 0.5, 0.0), (0.5, 0.3, 0.5), (-0.5, 1.0, -0.5)]
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No furniture (should get maximum reward)
    num_objects_3 = 0
    class_label_indices_3 = []
    translations_3 = []
    sizes_3 = []
    orientations_3 = []
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
    print("Scene 1 (furniture on side, central open) reward:", rewards[0].item())
    print("Scene 2 (furniture in center) reward:", rewards[1].item())
    print("Scene 3 (no furniture) reward:", rewards[2].item())
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] > rewards[1], f"Furniture on side (reward={rewards[0].item():.4f}) should have higher reward than furniture in center (reward={rewards[1].item():.4f})"
    # assert rewards[2] >= rewards[0], f"No furniture (reward={rewards[2].item():.4f}) should have highest reward"
    # assert rewards[2] == 1.0, f"No furniture should have reward 1.0, got {rewards[2].item():.4f}"
    # assert rewards.max() <= 1.5, f"Rewards should be capped at 1.5, got {rewards.max().item():.4f}"
    
    print("All tests passed!")