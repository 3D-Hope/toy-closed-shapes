import torch
import math
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Encourages chairs to have similar orientations with relaxed threshold.
    Uses circular standard deviation - lower is better.
    Reward: 0 if std < 0.8 radians (~46 degrees), penalty proportional to std otherwise
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    N = parsed_scenes['positions'].shape[1]
    device = parsed_scenes['device']
    
    seating_types = {'dining_chair', 'armchair', 'lounge_chair', 'stool', 'chinese_chair',
                     'multi_seat_sofa', 'loveseat_sofa', 'l_shaped_sofa', 'chaise_longue_sofa', 'lazy_sofa'}
    
    rewards = torch.zeros(B, device=device)
    angle_threshold = 0.8  # radians (~46 degrees) - relaxed from 0.3
    
    for b in range(B):
        seat_angles = []
        
        for n in range(N):
            if parsed_scenes['is_empty'][b, n]:
                continue
            
            obj_idx = parsed_scenes['object_indices'][b, n].item()
            obj_label = idx_to_labels.get(obj_idx, 'unknown')
            
            if obj_label in seating_types:
                # Extract angle from orientation (cos, sin)
                cos_theta = parsed_scenes['orientations'][b, n, 0].item()
                sin_theta = parsed_scenes['orientations'][b, n, 1].item()
                angle = math.atan2(sin_theta, cos_theta)
                seat_angles.append(angle)
        
        if len(seat_angles) < 2:
            # Need at least 2 seats to check alignment
            rewards[b] = 0.0
            continue
        
        # Calculate circular standard deviation
        angles_tensor = torch.tensor(seat_angles, device=device)
        
        # Convert to unit vectors and compute mean direction
        mean_cos = torch.cos(angles_tensor).mean()
        mean_sin = torch.sin(angles_tensor).mean()
        R = torch.sqrt(mean_cos**2 + mean_sin**2)
        
        # Circular standard deviation
        circ_std = torch.sqrt(-2 * torch.log(R + 1e-8))
        
        if circ_std < angle_threshold:
            rewards[b] = 0.0
        else:
            # Penalty increases with deviation, capped at -5
            rewards[b] = -min((circ_std - angle_threshold).item() * 3.0, 5.0)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene_for_testing = utility_functions['create_scene_for_testing']['function']
    
    # Scene 1: All chairs facing same direction (0 degrees)
    num_objects_1 = 5
    class_label_indices_1 = [10] * 5
    translations_1 = [(i*1.5, 0.4, 0) for i in range(5)]
    sizes_1 = [(0.25, 0.4, 0.25)] * 5
    orientations_1 = [(1, 0)] * 5  # All facing 0 degrees
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Chairs with moderate variations (within threshold ~40 degrees)
    import math
    num_objects_2 = 4
    class_label_indices_2 = [10] * 4
    translations_2 = [(i*1.5, 0.4, 0) for i in range(4)]
    sizes_2 = [(0.25, 0.4, 0.25)] * 4
    # Variations: 0, 20, -20, 30 degrees
    orientations_2 = [
        (math.cos(0), math.sin(0)),
        (math.cos(math.radians(20)), math.sin(math.radians(20))),
        (math.cos(math.radians(-20)), math.sin(math.radians(-20))),
        (math.cos(math.radians(30)), math.sin(math.radians(30)))
    ]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Chairs facing random directions
    num_objects_3 = 4
    class_label_indices_3 = [10] * 4
    translations_3 = [(i*1.5, 0.4, 0) for i in range(4)]
    sizes_3 = [(0.25, 0.4, 0.25)] * 4
    orientations_3 = [
        (1, 0),  # 0 degrees
        (0, 1),  # 90 degrees
        (-1, 0),  # 180 degrees
        (0, -1)  # 270 degrees
    ]
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
    print("Expected: [~0.0 (aligned), ~0.0 (moderate variance), <-2.0 (random)]")
    
    assert rewards.shape[0] == 3
    assert rewards[0] >= -0.5, f"Scene 1 should be aligned: got {rewards[0]}"
    assert rewards[1] >= -1.0, f"Scene 2 should be mostly aligned: got {rewards[1]}"
    assert rewards[2] < -1.5, f"Scene 3 should be penalized (random directions): got {rewards[2]}"
    print("All tests passed!")