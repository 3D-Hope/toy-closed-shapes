import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function that penalizes presence of hazardous furniture.
    Returns 1.0 if no hazardous furniture present, 0.0 otherwise.
    Hazardous furniture: dressing_table, desk, tv_stand, coffee_table
    '''
    device = parsed_scenes['device']
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N, num_classes = one_hot.shape
    
    # Find hazardous furniture indices
    hazardous_indices = []
    for idx, label in idx_to_labels.items():
        if label in ['dressing_table', 'desk', 'tv_stand', 'coffee_table']:
            hazardous_indices.append(idx)
    
    rewards = torch.ones(B, device=device)
    
    # Check for presence of any hazardous furniture
    for hazard_idx in hazardous_indices:
        hazard_mask = one_hot[:, :, hazard_idx] * (~is_empty)  # (B, N)
        has_hazard = (hazard_mask.sum(dim=1) > 0).float()  # (B,)
        rewards = rewards * (1.0 - has_hazard)  # Set to 0 if any hazard found
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: No hazardous furniture (should get reward 1.0)
    num_objects_1 = 3
    class_label_indices_1 = [11, 12, 5]  # kids_bed, nightstand, children_cabinet
    translations_1 = [(1.0, 0.3, 1.0), (2.5, 0.25, 1.0), (0.5, 0.5, 3.0)]
    sizes_1 = [(0.8, 0.3, 1.0), (0.3, 0.25, 0.3), (0.4, 0.5, 0.4)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has desk (should get reward 0.0)
    num_objects_2 = 3
    class_label_indices_2 = [11, 7, 12]  # kids_bed, desk, nightstand
    translations_2 = [(1.0, 0.3, 1.0), (2.5, 0.4, 1.0), (0.5, 0.25, 3.0)]
    sizes_2 = [(0.8, 0.3, 1.0), (0.6, 0.4, 0.8), (0.3, 0.25, 0.3)]
    orientations_2 = [(1, 0), (1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has tv_stand and dressing_table (should get reward 0.0)
    num_objects_3 = 4
    class_label_indices_3 = [11, 19, 10, 12]  # kids_bed, tv_stand, dressing_table, nightstand
    translations_3 = [(1.0, 0.3, 1.0), (2.5, 0.3, 1.0), (0.5, 0.4, 3.0), (3.5, 0.25, 1.0)]
    sizes_3 = [(0.8, 0.3, 1.0), (0.5, 0.3, 0.4), (0.6, 0.4, 0.5), (0.3, 0.25, 0.3)]
    orientations_3 = [(1, 0), (1, 0), (1, 0), (1, 0)]
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
    print("Expected: [1.0, 0.0, 0.0]")
    assert rewards.shape[0] == 3, f"Expected 3 rewards, got {rewards.shape[0]}"
    assert torch.isclose(rewards[0], torch.tensor(1.0)), f"Scene 1 should have reward 1.0, got {rewards[0]}"
    assert torch.isclose(rewards[1], torch.tensor(0.0)), f"Scene 2 should have reward 0.0, got {rewards[1]}"
    assert torch.isclose(rewards[2], torch.tensor(0.0)), f"Scene 3 should have reward 0.0, got {rewards[2]}"
    print("All tests passed for no_hazardous_furniture!")