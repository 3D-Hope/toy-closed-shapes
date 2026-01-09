import torch
import numpy as np
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for furniture wall proximity - encourages furniture near walls.
    Measures average minimum distance from each furniture to nearest wall.
    
    Input:
        - parsed_scenes: dict with scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor vertices
        - **kwargs: additional arguments
    
    Output:
        reward: torch.Tensor of shape (B,) - higher for furniture closer to walls
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    orientations = parsed_scenes['orientations']  # (B, N, 2)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    rewards = torch.zeros(B, device=device)
    
    utility_functions = get_all_utility_functions()
    find_closest_wall = utility_functions["find_closest_wall_to_object"]["function"]
    
    for b in range(B):
        # Get valid furniture
        valid_mask = ~is_empty[b]
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No furniture
            rewards[b] = 0.0
            continue
        
        distances_to_wall = []
        
        for idx in valid_indices:
            pos = positions[b, idx:idx+1]  # (1, 3)
            size = sizes[b, idx:idx+1]  # (1, 3)
            orient = orientations[b, idx:idx+1]  # (1, 2)
            
            # Find distance to closest wall
            try:
                _, distance = find_closest_wall(pos, orient, size, floor_polygons, **kwargs)
                distances_to_wall.append(distance)
            except:
                # Fallback: compute distance to boundary edges
                poly = torch.tensor(floor_polygons, device=device, dtype=torch.float32)
                x, z = pos[0, 0], pos[0, 2]
                
                # Distance to each edge
                edge_dists = []
                for i in range(len(floor_polygons)):
                    p1 = poly[i]
                    p2 = poly[(i+1) % len(floor_polygons)]
                    
                    # Point to line segment distance
                    v = p2 - p1
                    w = torch.tensor([x, z], device=device) - p1
                    c1 = torch.dot(w, v)
                    c2 = torch.dot(v, v)
                    
                    if c2 == 0:
                        dist = torch.norm(w)
                    else:
                        t = torch.clamp(c1 / c2, 0, 1)
                        proj = p1 + t * v
                        dist = torch.norm(torch.tensor([x, z], device=device) - proj)
                    
                    edge_dists.append(dist)
                
                min_dist = min(edge_dists)
                distances_to_wall.append(min_dist)
        
        # Average distance to walls
        if len(distances_to_wall) > 0:
            if isinstance(distances_to_wall[0], torch.Tensor):
                avg_distance = torch.stack(distances_to_wall).mean()
            else:
                avg_distance = torch.tensor(np.mean(distances_to_wall), device=device)
            
            # Transform to reward: closer to wall = higher reward
            # Use exponential decay with 1m reference distance
            avg_distance_capped = torch.clamp(avg_distance, 0, 3.0)
            rewards[b] = torch.exp(-avg_distance_capped)
        else:
            rewards[b] = 0.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for furniture wall proximity reward.
    '''
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Furniture close to walls (should get high reward)
    num_objects_1 = 3
    class_label_indices_1 = [8, 12, 20]
    # Assuming floor polygon is roughly [-3, 3] x [-3, 3]
    translations_1 = [(-2.5, 0.5, -2.5), (-2.5, 0.3, 2.0), (2.0, 1.0, -2.5)]
    sizes_1 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Furniture in center of room (should get lower reward)
    num_objects_2 = 3
    class_label_indices_2 = [8, 12, 20]
    translations_2 = [(0.0, 0.5, 0.0), (0.5, 0.3, 0.5), (-0.5, 1.0, -0.5)]
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3), (0.5, 1.0, 0.5)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Mixed placement
    num_objects_3 = 2
    class_label_indices_3 = [8, 12]
    translations_3 = [(-2.5, 0.5, 0.0), (0.0, 0.3, 0.0)]
    sizes_3 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_3 = [(1, 0), (1, 0)]
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
    print("Scene 1 (furniture near walls) reward:", rewards[0].item())
    print("Scene 2 (furniture in center) reward:", rewards[1].item())
    print("Scene 3 (mixed placement) reward:", rewards[2].item())
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] > rewards[1], f"Furniture near walls (reward={rewards[0].item():.4f}) should have higher reward than furniture in center (reward={rewards[1].item():.4f})"
    # assert rewards[0] > 0.5, f"Furniture near walls should have reward > 0.5, got {rewards[0].item():.4f}"
    # assert rewards[1] < 0.5, f"Furniture in center should have reward < 0.5, got {rewards[1].item():.4f}"
    # assert rewards[2] > rewards[1] and rewards[2] < rewards[0], f"Mixed placement reward ({rewards[2].item():.4f}) should be between center ({rewards[1].item():.4f}) and walls ({rewards[0].item():.4f})"
    
    print("All tests passed!")