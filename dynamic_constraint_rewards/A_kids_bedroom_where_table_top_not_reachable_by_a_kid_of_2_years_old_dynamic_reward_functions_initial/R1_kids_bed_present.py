import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function to verify that at least one kids_bed exists in the scene.
    
    Returns:
        1.0 if kids_bed exists
        -1.0 if no kids_bed exists
    '''
    
    device = parsed_scenes['device']
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    
    B = object_indices.shape[0]
    
    # Get kids_bed class index
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Check if kids_bed exists
        kids_bed_mask = (object_indices[b] == kids_bed_idx) & (~is_empty[b])
        has_kids_bed = kids_bed_mask.any()
        
        rewards[b] = 1.0 if has_kids_bed else -1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for kids_bed presence constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    table_idx = labels_to_idx['table']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Has kids_bed
    num_objects_1 = 3
    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: No kids_bed (has regular furniture)
    num_objects_2 = 3
    class_label_indices_2 = [table_idx, nightstand_idx, wardrobe_idx]
    translations_2 = [(0, 0.75, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    sizes_2 = [(0.4, 0.75, 0.4), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has kids_bed with other furniture
    num_objects_3 = 2
    class_label_indices_3 = [kids_bed_idx, wardrobe_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.5, 0)]
    sizes_3 = [(0.9, 0.3, 0.75), (0.5, 0.5, 1.0)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
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
    print(f"Scene 1 (has kids_bed): {rewards[0].item()}")
    print(f"Scene 2 (no kids_bed): {rewards[1].item()}")
    print(f"Scene 3 (has kids_bed): {rewards[2].item()}")
    
    assert rewards.shape[0] == 3, "Should have 3 scenes"
    assert rewards[0].item() == 1.0, f"Scene 1: Has kids_bed, should return 1.0, got {rewards[0].item()}"
    assert rewards[1].item() == -1.0, f"Scene 2: No kids_bed, should return -1.0, got {rewards[1].item()}"
    assert rewards[2].item() == 1.0, f"Scene 3: Has kids_bed, should return 1.0, got {rewards[2].item()}"
    
    print("All tests passed!")