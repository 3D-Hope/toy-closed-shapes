import torch
import math
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for organized seating arrangement - chairs should face similar directions
    and be spatially clustered.
    
    Returns:
        reward: (B,) tensor - negative penalty for angular deviation and spatial scatter
    '''
    utility_functions = get_all_utility_functions()
    compute_angle = utility_functions["compute_angle_between_objects"]["function"]
    distance_2d = utility_functions["distance_2d"]["function"]
    
    positions = parsed_scenes['positions']  # (B, N, 3)
    orientations = parsed_scenes['orientations']  # (B, N, 2)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N = positions.shape[0], positions.shape[1]
    device = parsed_scenes['device']
    
    # Seating furniture
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    seating_indices = [labels_to_idx.get(st, -1) for st in seating_types if labels_to_idx.get(st, -1) != -1]
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find all seating objects
        seat_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for idx in seating_indices:
            seat_mask |= (object_indices[b] == idx)
        seat_mask &= ~is_empty[b]
        
        seat_positions = positions[b][seat_mask]  # (num_seats, 3)
        seat_orientations = orientations[b][seat_mask]  # (num_seats, 2)
        num_seats = seat_mask.sum().item()
        
        if num_seats < 2:
            rewards[b] = 0.0
            continue
        
        # Calculate angular alignment - all chairs should face similar direction
        total_angle_dev = 0.0
        count = 0
        for i in range(num_seats):
            for j in range(i+1, num_seats):
                angle_rad = compute_angle(seat_orientations[i], seat_orientations[j])
                angle_deg = abs(angle_rad.item()) * 180.0 / math.pi
                # Normalize to [0, 90] (considering 180° symmetry)
                angle_dev = min(angle_deg, 180 - angle_deg)
                total_angle_dev += angle_dev
                count += 1
        
        avg_angle_dev = total_angle_dev / count if count > 0 else 0.0
        
        # Calculate spatial clustering - chairs should be close together
        total_distance = 0.0
        dist_count = 0
        for i in range(num_seats):
            for j in range(i+1, num_seats):
                dist = distance_2d(seat_positions[i], seat_positions[j])
                total_distance += dist.item()
                dist_count += 1
        
        avg_distance = total_distance / dist_count if dist_count > 0 else 0.0
        
        # Penalty: higher for scattered placement and misaligned orientations
        # Good classroom: avg_angle_dev < 15°, avg_distance < 3m
        angle_penalty = max(0, (avg_angle_dev - 15.0) * 0.1)  # 0.1 per degree over 15°
        distance_penalty = max(0, (avg_distance - 3.0) * 0.5)  # 0.5 per meter over 3m
        
        rewards[b] = -(angle_penalty + distance_penalty)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the organized_seating_arrangement reward function.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx.get('dining_chair', 10)
    
    # Scene 1: Well-organized (same direction, close together)
    num_objects_1 = 6
    class_label_indices_1 = [chair_idx] * 6
    translations_1 = [(0, 0.4, 0), (1, 0.4, 0), (2, 0.4, 0), (0, 0.4, 1), (1, 0.4, 1), (2, 0.4, 1)]
    sizes_1 = [(0.3, 0.4, 0.3)] * 6
    orientations_1 = [(1, 0)] * 6  # All facing same direction
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Poorly organized (random directions, scattered)
    num_objects_2 = 6
    class_label_indices_2 = [chair_idx] * 6
    translations_2 = [(0, 0.4, 0), (5, 0.4, 0), (10, 0.4, 0), (15, 0.4, 5), (20, 0.4, 10), (25, 0.4, 15)]
    sizes_2 = [(0.3, 0.4, 0.3)] * 6
    orientations_2 = [(1, 0), (0, 1), (-1, 0), (0, -1), (0.707, 0.707), (-0.707, 0.707)]  # Random directions
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Only 1 chair (no penalty)
    num_objects_3 = 1
    class_label_indices_3 = [chair_idx]
    translations_3 = [(0, 0.4, 0)]
    sizes_3 = [(0.3, 0.4, 0.3)]
    orientations_3 = [(1, 0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    assert rewards.shape[0] == 3
    
    assert rewards[0] >= -1.0, "Scene 1 should have minimal penalty (organized)"
    assert rewards[1] <= -5.0, "Scene 2 should have high penalty (disorganized)"
    assert rewards[2] >= -0.1, "Scene 3 should have no penalty (only 1 chair)"
    print("All tests passed for organized_seating_arrangement!")