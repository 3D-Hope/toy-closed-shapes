import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Ensures at least 50% of seating furniture is within 2.5m of desks or tables.
    Relaxed version: distance threshold increased to 2.5m, only requires 50% compliance
    Reward: 0 if >= 50% seats are near tables, penalty for lower percentage
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    N = parsed_scenes['positions'].shape[1]
    device = parsed_scenes['device']
    
    seating_types = {'dining_chair', 'armchair', 'lounge_chair', 'stool', 'chinese_chair',
                     'multi_seat_sofa', 'loveseat_sofa', 'l_shaped_sofa', 'chaise_longue_sofa', 'lazy_sofa'}
    table_types = {'desk', 'dining_table'}
    
    rewards = torch.zeros(B, device=device)
    distance_threshold = 2.5  # meters - relaxed from 1.5m
    
    for b in range(B):
        # Find all seating and table positions
        seat_positions = []
        table_positions = []
        
        for n in range(N):
            if parsed_scenes['is_empty'][b, n]:
                continue
            
            obj_idx = parsed_scenes['object_indices'][b, n].item()
            obj_label = idx_to_labels.get(obj_idx, 'unknown')
            
            if obj_label in seating_types:
                seat_positions.append(parsed_scenes['positions'][b, n])
            elif obj_label in table_types:
                table_positions.append(parsed_scenes['positions'][b, n])
        
        if len(seat_positions) == 0:
            # No seats to check
            rewards[b] = 0.0
            continue
        
        if len(table_positions) == 0:
            # No tables but seats exist - moderate penalty
            rewards[b] = -3.0
            continue
        
        # Count seats within distance threshold
        seats_near_table = 0
        for seat_pos in seat_positions:
            min_dist = float('inf')
            for table_pos in table_positions:
                # XZ plane distance
                dist = torch.sqrt((seat_pos[0] - table_pos[0])**2 + (seat_pos[2] - table_pos[2])**2)
                min_dist = min(min_dist, dist.item())
            
            if min_dist <= distance_threshold:
                seats_near_table += 1
        
        # Calculate percentage of seats near tables
        percentage_near = seats_near_table / len(seat_positions)
        
        if percentage_near >= 0.5:  # 50% threshold
            rewards[b] = 0.0
        else:
            # Penalty proportional to shortfall, capped at -5
            shortfall = 0.5 - percentage_near
            rewards[b] = -min(shortfall * 10.0, 5.0)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions['create_scene_for_testing']['function']
    
    # Scene 1: All chairs near table (within 2.5m) - 100% compliance
    num_objects_1 = 5
    class_label_indices_1 = [11, 10, 10, 10, 10]  # dining_table + 4 chairs
    translations_1 = [(2, 0.4, 2), (1.0, 0.4, 2), (3.0, 0.4, 2), (2, 0.4, 0.3), (2, 0.4, 3.7)]
    sizes_1 = [(0.8, 0.4, 0.6), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25)]
    orientations_1 = [(1, 0)] * 5
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 50% chairs near table - exactly at threshold
    num_objects_2 = 5
    class_label_indices_2 = [11, 10, 10, 10, 10]  # dining_table + 4 chairs (2 near, 2 far)
    translations_2 = [(0, 0.4, 0), (1, 0.4, 0), (2, 0.4, 0), (5, 0.4, 5), (6, 0.4, 6)]
    sizes_2 = [(0.8, 0.4, 0.6), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25), (0.25, 0.4, 0.25)]
    orientations_2 = [(1, 0)] * 5
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Chairs without any table
    num_objects_3 = 2
    class_label_indices_3 = [10, 10]
    translations_3 = [(0, 0.4, 0), (1, 0.4, 0)]
    sizes_3 = [(0.25, 0.4, 0.25), (0.25, 0.4, 0.25)]
    orientations_3 = [(1, 0), (1, 0)]
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
    print("Expected: [~0.0 (all chairs near), ~0.0 (50% near), -3.0 (no table)]")
    
    assert rewards.shape[0] == 3
    assert rewards[0] >= -0.5, f"Scene 1 should satisfy (all chairs near): got {rewards[0]}"
    assert rewards[1] >= -0.5, f"Scene 2 should satisfy (50% near): got {rewards[1]}"
    assert rewards[2] <= -2.5 and rewards[2] >= -3.5, f"Scene 3 should be ~-3.0 (no table): got {rewards[2]}"
    print("All tests passed!")