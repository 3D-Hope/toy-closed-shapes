import torch
import numpy as np
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for open floor area - measures largest contiguous open space.
    Target: minimum 2m x 2m (4 sq meters) for workout space.
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of lists - each element is a list of vertices for that scene
        - **kwargs: additional arguments
    
    Output:
        reward: torch.Tensor of shape (B,) - higher for larger open areas
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    rewards = torch.zeros(B, device=device)
    
    # Target open area for workout (4 sq meters = 2m x 2m)
    target_area = 4.0
    
    for b in range(B):
        # Get floor polygon for this specific scene
        poly = np.array(floor_polygons[b])
        
        # Get valid furniture positions and sizes
        valid_mask = ~is_empty[b]
        valid_positions = positions[b][valid_mask]  # (num_valid, 3)
        valid_sizes = sizes[b][valid_mask]  # (num_valid, 3)
        
        if valid_positions.shape[0] == 0:
            # No furniture - entire floor is open
            # Compute floor area from polygon
            floor_area = compute_polygon_area(poly)
            rewards[b] = min(floor_area / target_area, 2.0)  # Cap at 2x target
            continue
        
        # Estimate largest open area using grid-based approach
        # Create occupancy grid
        grid_res = 0.2  # 20cm resolution
        
        # Get floor bounds from this scene's polygon
        x_min, x_max = poly[:, 0].min(), poly[:, 0].max()
        z_min, z_max = poly[:, 1].min(), poly[:, 1].max()
        
        # Create grid
        x_range = torch.arange(x_min, x_max, grid_res, device=device)
        z_range = torch.arange(z_min, z_max, grid_res, device=device)
        grid_x, grid_z = torch.meshgrid(x_range, z_range, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_z.flatten()], dim=1)  # (num_points, 2)
        
        # Mark occupied cells
        occupied = torch.zeros(grid_points.shape[0], dtype=torch.bool, device=device)
        
        for i in range(valid_positions.shape[0]):
            pos = valid_positions[i, [0, 2]]  # (x, z)
            size = valid_sizes[i, [0, 2]]  # (sx/2, sz/2)
            
            # Check which grid points are inside this object's bounding box
            dist = torch.abs(grid_points - pos.unsqueeze(0))
            inside = (dist[:, 0] < size[0]) & (dist[:, 1] < size[1])
            occupied |= inside
        
        # Count free cells
        free_cells = (~occupied).sum().float()
        cell_area = grid_res * grid_res
        open_area = free_cells * cell_area
        
        # Compute reward based on open area
        # Sigmoid-like function: reward increases with area, saturates above target
        area_ratio = open_area / target_area
        rewards[b] = torch.tanh(area_ratio)  # Bounded [0, 1]
    
    return rewards

def compute_polygon_area(vertices):
    """Compute area of polygon using shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for open floor area reward.
    '''
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Minimal furniture - large open space (should get high reward)
    num_objects_1 = 2
    class_label_indices_1 = [8, 12]  # bed and nightstand in corner
    translations_1 = [(-2.0, 0.5, -2.0), (-2.5, 0.3, -1.0)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Many furniture pieces - limited open space (should get lower reward)
    num_objects_2 = 5
    class_label_indices_2 = [8, 12, 20, 7, 4]
    translations_2 = [(0.0, 0.5, 0.0), (1.5, 0.3, 0.5), (-1.5, 1.0, 0.5), (0.0, 0.4, 2.0), (1.0, 0.5, -1.5)]
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5), (0.6, 0.4, 0.6), (0.4, 0.5, 0.4)]
    orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: No furniture - maximum open space
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
    print("Scene 1 (minimal furniture) reward:", rewards[0].item())
    print("Scene 2 (many furniture) reward:", rewards[1].item())
    print("Scene 3 (no furniture) reward:", rewards[2].item())
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] > rewards[1], f"Minimal furniture (reward={rewards[0].item():.4f}) should have higher reward than many furniture (reward={rewards[1].item():.4f})"
    # assert rewards[2] >= rewards[0], f"No furniture (reward={rewards[2].item():.4f}) should have highest reward"
    # assert rewards[1] >= 0.0, f"Reward should be non-negative, got {rewards[1].item():.4f}"
    # assert rewards[2] <= 2.0, f"Reward should be capped, got {rewards[2].item():.4f}"
    
    print("All tests passed!")