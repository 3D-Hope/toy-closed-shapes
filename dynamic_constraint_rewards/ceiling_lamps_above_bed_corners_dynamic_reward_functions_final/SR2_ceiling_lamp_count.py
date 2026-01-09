import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for having exactly 4 ceiling lamps.
    Returns 0 if exactly 4 ceiling lamps exist, negative penalty otherwise.
    Penalty = -|count - 4| (linear penalty based on deviation from 4).
    '''
    utility_functions = get_all_utility_functions()
    get_object_count = utility_functions["get_object_count_in_a_scene"]["function"]
    
    B = parsed_scenes['positions'].shape[0]
    rewards = torch.zeros(B, device=parsed_scenes['device'])
    
    for i in range(B):
        scene_one_hot = parsed_scenes['one_hot'][i:i+1]
        count = get_object_count(scene_one_hot, "ceiling_lamp", idx_to_labels)
        
        deviation = abs(count - 4)
        rewards[i] = -float(deviation)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: Exactly 4 ceiling lamps (satisfies)
    scene_1 = create_scene(
        room_type=room_type,
        num_objects=5,
        class_label_indices=[3, 3, 3, 3, 8],  # 4 ceiling_lamps, double_bed
        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 2.8, 1), (3, 2.8, 1), (2, 0.5, 2)],
        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],
        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    )
    
    # Scene 2: 2 ceiling lamps (fails - too few)
    scene_2 = create_scene(
        room_type=room_type,
        num_objects=3,
        class_label_indices=[3, 3, 8],  # 2 ceiling_lamps, double_bed
        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 0.5, 2)],
        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],
        orientations=[(1, 0), (1, 0), (1, 0)]
    )
    
    # Scene 3: 6 ceiling lamps (fails - too many)
    scene_3 = create_scene(
        room_type=room_type,
        num_objects=7,
        class_label_indices=[3, 3, 3, 3, 3, 3, 8],  # 6 ceiling_lamps, double_bed
        translations=[(0, 2.8, 0), (1, 2.8, 0), (2, 2.8, 1), (3, 2.8, 1), (4, 2.8, 2), (5, 2.8, 2), (2, 0.5, 2)],
        sizes=[(0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (1.0, 0.4, 1.5)],
        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    )
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print("Expected: [0.0, -2.0, -2.0]")
    assert rewards.shape[0] == 3
    assert rewards[0] == 0.0, f"Scene 1 should have reward 0.0, got {rewards[0]}"
    assert rewards[1] == -2.0, f"Scene 2 should have reward -2.0, got {rewards[1]}"
    assert rewards[2] == -2.0, f"Scene 3 should have reward -2.0, got {rewards[2]}"