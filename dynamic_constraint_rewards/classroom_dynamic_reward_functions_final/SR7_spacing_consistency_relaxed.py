import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Checks that seating positions maintain consistent spacing with relaxed tolerance.
    Computes pairwise distances between adjacent seats and checks standard deviation.
    Reward: 0 if spacing variance <= 1.0m, penalty proportional to variance otherwise
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    N = parsed_scenes['positions'].shape[1]
    device = parsed_scenes['device']
    
    seating_types = {'dining_chair', 'armchair', 'lounge_chair', 'stool', 'chinese_chair',
                     'multi_seat_sofa', 'loveseat_sofa', 'l_shaped_sofa', 'chaise_longue_sofa', 'lazy_sofa'}
    
    rewards = torch.zeros(B, device=device)
    spacing_threshold = 1.0  # meters variance allowed - relaxed from 0.5m
    
    for b in range(B):
        seat_positions = []
        
        for n in range(N):
            if parsed_scenes['is_empty'][b, n]:
                continue
            
            obj_idx = parsed_scenes['object_indices'][b, n].item()
            obj_label = idx_to_labels.get(obj_idx, 'unknown')
            
            if obj_label in seating_types:
                pos = parsed_scenes['positions'][b, n]
                seat_positions.append(pos)
        
        if len(seat_positions) < 3:
            # Need at least 3 seats to check spacing consistency
            rewards[b] = 0.0
            continue
        
        # Sort seats by position (use x-coordinate primarily, then z)
        seat_positions_sorted = sorted(seat_positions, key=lambda p: (p[0].item(), p[2].item()))
        
        # Calculate distances between adjacent seats
        distances = []
        for i in range(len(seat_positions_sorted) - 1):
            pos1 = seat_positions_sorted[i]
            pos2 = seat_positions_sorted[i + 1]
            # XZ plane distance
            dist = torch.sqrt((pos2[0] - pos1[0])**2 + (pos2[2] - pos1[2])**2)
            distances.append(dist.item())
        
        if len(distances) == 0:
            rewards[b] = 0.0
            continue
        
        # Calculate standard deviation of distances
        distances_tensor = torch.tensor(distances, device=device)
        std_dev = torch.std(distances_tensor)
        
        if std_dev <= spacing_threshold:
            rewards[b] = 0.0
        else:
            # Penalty increases with spacing variance, capped at -5
            rewards[b] = -min((std_dev - spacing_threshold).item() * 3.0, 5.0)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions['create_scene_for_testing']['function']
    
    # Scene 1: Uniformly spaced chairs (1.5m apart)
    num_objects_1 = 5
    class_label_indices_1 = [10] * 5
    translations_1 = [(i*1.5, 0.4, 0) for i in range(5)]  # Uniform 1.5m spacing
    sizes_1 = [(0.25, 0.4, 0.25)] * 5
    orientations_1 = [(1, 0)] * 5
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Moderate variance in spacing (within threshold ~0.8m std)
    num_objects_2 = 5
    class_label_indices_2 = [10] * 5
    translations_2 = [(0, 0.4, 0), (1.2, 0.4, 0), (2.6, 0.4, 0), (4.0, 0.4, 0), (5.5, 0.4, 0)]  # Moderate variance
    sizes_2 = [(0.25, 0.4, 0.25)] * 5
    orientations_2 = [(1, 0)] * 5
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Highly irregular spacing (large variance)
    num_objects_3 = 5
    class_label_indices_3 = [10] * 5
    translations_3 = [(0, 0.4, 0), (0.5, 0.4, 0), (2.5, 0.4, 0), (3.0, 0.4, 0), (7.0, 0.4, 0)]  # Very irregular
    sizes_3 = [(0.25, 0.4, 0.25)] * 5
    orientations_3 = [(1, 0)] * 5
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print("Expected: [~0.0 (uniform), ~0.0 (moderate variance), <-2.0 (irregular)]")
    
    assert rewards.shape[0] == 3
    assert rewards[0] >= -0.2, f"Scene 1 should have uniform spacing: got {rewards[0]}"
    assert rewards[1] >= -1.0, f"Scene 2 should have acceptable variance: got {rewards[1]}"
    assert rewards[2] < -1.5, f"Scene 3 should be penalized (irregular spacing): got {rewards[2]}"
    print("All tests passed!")