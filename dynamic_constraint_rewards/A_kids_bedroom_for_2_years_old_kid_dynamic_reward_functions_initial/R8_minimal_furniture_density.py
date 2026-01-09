import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that limits furniture count to <= 8 objects.
    Returns 1.0 if count <= 8, scales down linearly as count increases above 8.
    Capped at 0.0 for very high counts (>16).
    '''
    device = parsed_scenes['device']
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N = is_empty.shape
    
    # Count non-empty objects per scene
    object_counts = (~is_empty).sum(dim=1).float()  # (B,)
    
    max_desired = 8.0
    max_tolerable = 16.0
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        count = object_counts[b].item()
        
        if count <= max_desired:
            # Perfect: <= 8 objects
            rewards[b] = 1.0
        elif count <= max_tolerable:
            # Linear decay from 1.0 to 0.0 as count goes from 8 to 16
            rewards[b] = 1.0 - (count - max_desired) / (max_tolerable - max_desired)
        else:
            # Too many objects
            rewards[b] = 0.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: 5 objects (should get reward 1.0)
    num_objects_1 = 5
    class_label_indices_1 = [11, 12, 5, 13, 3]  # kids_bed, nightstand, children_cabinet, pendant_lamp, ceiling_lamp
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.5, 3.0), (3.0, 2.5, 2.0), (2.0, 2.7, 2.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4), (0.2, 0.1, 0.2), (0.3, 0.1, 0.3)]
    orientations_1 = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: 10 objects (should get reward ~0.75)
    num_objects_2 = 10
    class_label_indices_2 = [11, 12, 12, 5, 20, 17, 17, 14, 13, 3]
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.25, 1.0), (0.5, 0.5, 3.0), 
                      (3.5, 0.8, 0.5), (1.5, 0.2, 2.0), (2.5, 0.2, 2.5), (0.3, 0.3, 0.5),
                      (3.0, 2.5, 2.0), (2.0, 2.7, 2.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4),
               (0.8, 0.8, 0.6), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.3, 0.3, 0.2),
               (0.2, 0.1, 0.2), (0.3, 0.1, 0.3)]
    orientations_2 = [(1, 0)] * 10
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: 12 objects (should get reward 0.5)
    num_objects_3 = 12
    class_label_indices_3 = [11, 12, 12, 5, 20, 17, 17, 14, 4, 4, 13, 3]
    translations_3 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.25, 1.0), (0.5, 0.5, 3.0),
                      (3.5, 0.8, 0.5), (1.5, 0.2, 2.0), (2.5, 0.2, 2.5), (0.3, 0.3, 0.5),
                      (1.0, 0.3, 0.5), (3.0, 0.3, 0.5), (3.0, 2.5, 2.0), (2.0, 2.7, 2.0)]
    sizes_3 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4),
               (0.8, 0.8, 0.6), (0.2, 0.2, 0.2), (0.2, 0.2, 0.2), (0.3, 0.3, 0.2),
               (0.3, 0.3, 0.3), (0.3, 0.3, 0.3), (0.2, 0.1, 0.2), (0.3, 0.1, 0.3)]
    orientations_3 = [(1, 0)] * 12
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
    print("Expected: [1.0, ~0.75, 0.5]")
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert torch.isclose(rewards[0], torch.tensor(1.0), atol=0.01), f"Scene 1 should have reward 1.0, got {rewards[0]}"
    assert torch.isclose(rewards[1], torch.tensor(0.75), atol=0.05), f"Scene 2 should have reward ~0.75, got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(0.5), atol=0.05), f"Scene 3 should have reward 0.5, got {rewards[2]}"
    print("All tests passed for minimal_furniture_density!")