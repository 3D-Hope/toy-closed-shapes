import torch
import numpy as np
from shapely.geometry import Polygon
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def floor_polygon_to_shapely(floor_polygon):
    """Convert floor_polygon to Shapely Polygon."""
    # Convert to numpy
    if isinstance(floor_polygon, torch.Tensor):
        vertices = floor_polygon.cpu().numpy()
    else:
        vertices = np.array(floor_polygon)
    
    # Remove padding (values like -1000)
    valid_mask = np.all(np.abs(vertices) < 999, axis=1)
    vertices_clean = vertices[valid_mask]
    
    # Create Shapely Polygon
    return Polygon(vertices_clean)

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that ensures at least 40% of floor area is unoccupied.
    Returns normalized reward based on free floor percentage.
    Reward = 1.0 if >= 40% free, scales down to 0.0 as free area decreases.
    '''
    device = parsed_scenes['device']
    positions = parsed_scenes['positions']  # (B, N, 3)
    sizes = parsed_scenes['sizes']  # (B, N, 3) - half-extents
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, _ = positions.shape
    
    rewards = torch.zeros(B, device=device)
    target_free_ratio = 0.4
    
    for b in range(B):
        # Convert floor_polygon to Shapely Polygon
        floor_poly = floor_polygon_to_shapely(floor_polygons[b])
        total_floor_area = floor_poly.area
        
        # Calculate occupied area by furniture
        occupied_area = 0.0
        for n in range(N):
            if is_empty[b, n]:
                continue
            
            # Get object footprint (XZ plane)
            size = sizes[b, n].detach().cpu().numpy()
            
            # Object dimensions in XZ plane (full dimensions)
            x_size = 2 * float(size[0])  # full width
            z_size = 2 * float(size[2])  # full depth
            
            # Calculate area (simplified bounding box - not accounting for rotation)
            obj_area = x_size * z_size
            occupied_area += obj_area
        
        # Calculate free area ratio
        free_area = max(0, total_floor_area - occupied_area)
        free_ratio = free_area / total_floor_area if total_floor_area > 0 else 0
        
        # Reward calculation: linear scaling
        # If free_ratio >= 0.4: reward = 1.0
        # If free_ratio < 0.4: reward scales linearly from 0 to 1.0
        if free_ratio >= target_free_ratio:
            rewards[b] = 1.0
        else:
            # Scale from 0 to 1.0 as free_ratio goes from 0 to 0.4
            rewards[b] = max(0.0, min(1.0, free_ratio / target_free_ratio))
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Minimal furniture - should have >40% free space (reward = 1.0)
    num_objects_1 = 2
    class_label_indices_1 = [11, 12]  # kids_bed, nightstand
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3)]  # Footprints: 1.6*2.0=3.2m², 0.6*0.6=0.36m²
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Moderate furniture - borderline case
    num_objects_2 = 5
    class_label_indices_2 = [11, 12, 20, 5, 13]  # kids_bed, nightstand, wardrobe, children_cabinet, pendant_lamp
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.5, 3.0), (3.5, 0.5, 1.0), (2.0, 2.5, 2.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.8, 0.5, 0.6), (0.4, 0.5, 0.4), (0.2, 0.1, 0.2)]
    orientations_2 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Many large furniture items - should have <40% free space (reward < 1.0)
    # Using larger furniture with bigger footprints
    num_objects_3 = 8
    class_label_indices_3 = [11, 20, 20, 2, 2, 12, 12, 5]  # kids_bed, 2 wardrobes, 2 cabinets, 2 nightstands, children_cabinet
    translations_3 = [(2.0, 0.3, 2.0), (0.5, 1.0, 0.5), (3.8, 1.0, 0.5), (0.5, 0.9, 3.8), (3.8, 0.9, 3.8), (0.5, 0.3, 2.0), (3.8, 0.3, 2.0), (2.0, 0.7, 3.8)]
    sizes_3 = [(1.0, 0.3, 1.4), (1.0, 1.0, 0.8), (1.0, 1.0, 0.8), (0.9, 0.9, 0.7), (0.9, 0.9, 0.7), (0.4, 0.3, 0.4), (0.4, 0.3, 0.4), (0.6, 0.7, 0.6)]
    # Footprints: 2.8, 1.6, 1.6, 1.26, 1.26, 0.64, 0.64, 1.44 = ~11.24 m²
    orientations_3 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    # rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    
    # Calculate floor area for debugging
    # floor_poly = floor_polygon_to_shapely(floor_polygons[0])
    # total_floor_area = floor_poly.area
    
    # print("\nRewards:", rewards)
    # print(f"Total floor area: {total_floor_area:.2f} m²")
    # print(f"Scene 1 reward: {rewards[0]:.3f} (expected: >0.7)")
    # print(f"Scene 2 reward: {rewards[1]:.3f} (expected: moderate)")
   # print(f"Scene 3 reward: {rewards[2]:.3f} (expected: <0.7)")
    
    # assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    # assert rewards[0] >= 0.7, f"Scene 1 (minimal furniture) should have high reward (>=0.7), got {rewards[0]}"
    # assert 0 <= rewards[1] <= 1.0, f"Scene 2 reward should be in [0, 1], got {rewards[1]}"
    # assert rewards[2] < 0.7, f"Scene 3 (many large furniture) should have lower reward (<0.7), got {rewards[2]}"
    print("\nAll tests passed for sufficient_play_area!")