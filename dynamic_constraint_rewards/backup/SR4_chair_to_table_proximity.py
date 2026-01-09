import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions, create_scene_for_testing


def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Verify that seating has reasonable proximity to tables (within 1.5m).
    For each chair/stool, check if there's a table within 1.5m.
    Reward: 0 if all chairs have nearby table, otherwise -1 per unpaired chair
    '''
    utility_functions = get_all_utility_functions()
    positions = parsed_scenes['positions']  # (B, N, 3)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B = positions.shape[0]
    N = positions.shape[1]
    
    seating_types = ['dining_chair', 'stool', 'armchair', 'lounge_chair', 'chinese_chair']
    table_types = ['dining_table', 'desk', 'console_table']
    max_distance = 1.5
    
    rewards = []
    
    for i in range(B):
        # Find all chairs and tables in this scene
        chair_positions = []
        table_positions = []
        
        for j in range(N):
            if is_empty[i, j]:
                continue
            
            obj_idx = int(object_indices[i, j].item())
            obj_label = idx_to_labels[obj_idx]
            pos = positions[i, j]
            
            if obj_label in seating_types:
                chair_positions.append(pos)
            elif obj_label in table_types:
                table_positions.append(pos)
        
        if len(chair_positions) == 0:
            rewards.append(0.0)
            continue
        
        if len(table_positions) == 0:
            # No tables but have chairs - penalize all chairs
            rewards.append(-len(chair_positions))
            continue
        
        # Count chairs without nearby table
        unpaired_chairs = 0
        for chair_pos in chair_positions:
            has_nearby_table = False
            for table_pos in table_positions:
                dist = utility_functions['distance_2d']['function'](chair_pos, table_pos)
                if dist.item() <= max_distance:
                    has_nearby_table = True
                    break
            
            if not has_nearby_table:
                unpaired_chairs += 1
        
        if unpaired_chairs == 0:
            reward_val = 0.0
        else:
            reward_val = -unpaired_chairs
        
        rewards.append(reward_val)
    
    reward = torch.tensor(rewards, device=parsed_scenes['device'], dtype=torch.float)
    return reward


def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    import torch
    
    # Scene 1: 2 chairs near a table (within 1.5m)
    num_objects_1 = 3
    class_label_indices_1 = [10, 10, 11]  # 2 chairs + 1 table
    translations_1 = [(1, 0.5, 1), (2, 0.5, 1), (1.5, 0.4, 1.8)]
    sizes_1 = [(0.3, 0.5, 0.3), (0.3, 0.5, 0.3), (0.8, 0.4, 0.6)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene_for_testing(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 2 chairs but one is far from table (>1.5m)
    num_objects_2 = 3
    class_label_indices_2 = [10, 10, 11]
    translations_2 = [(1, 0.5, 1), (5, 0.5, 1), (1.5, 0.4, 1.8)]  # second chair far
    sizes_2 = [(0.3, 0.5, 0.3), (0.3, 0.5, 0.3), (0.8, 0.4, 0.6)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene_for_testing(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Chairs but no table
    num_objects_3 = 2
    class_label_indices_3 = [10, 10]
    translations_3 = [(1, 0.5, 1), (2, 0.5, 1)]
    sizes_3 = [(0.3, 0.5, 0.3), (0.3, 0.5, 0.3)]
    orientations_3 = [(1, 0), (1, 0)]
    scene_3 = create_scene_for_testing(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0) for k in tensor_keys
    }
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Chair to Table Proximity Rewards:", rewards)
    
    assert torch.isclose(rewards[0], torch.tensor(0.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[1], torch.tensor(-1.0, device=parsed_scenes['device']))
    assert torch.isclose(rewards[2], torch.tensor(-2.0, device=parsed_scenes['device']))
    
    return rewards
