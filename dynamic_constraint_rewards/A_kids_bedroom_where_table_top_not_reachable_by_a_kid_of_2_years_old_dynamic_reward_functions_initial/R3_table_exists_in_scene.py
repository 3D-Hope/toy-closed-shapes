import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function to verify that at least one table-type object exists in the scene.
    
    Returns:
        1.0 if at least one table/desk/dressing_table exists
        -1.0 if no table-type objects exist
    '''
    
    device = parsed_scenes['device']
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    
    B = object_indices.shape[0]
    
    # Get table-type class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    table_classes = [labels_to_idx['table'], labels_to_idx['desk'], labels_to_idx['dressing_table']]
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Check if any table-type object exists
        has_table = False
        for table_class in table_classes:
            table_mask = (object_indices[b] == table_class) & (~is_empty[b])
            if table_mask.any():
                has_table = True
                break
        
        rewards[b] = 1.0 if has_table else -1.0
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for table existence constraint.
    '''
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    kids_bed_idx = labels_to_idx['kids_bed']
    table_idx = labels_to_idx['table']
    desk_idx = labels_to_idx['desk']
    dressing_table_idx = labels_to_idx['dressing_table']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Has table
    num_objects_1 = 3
    class_label_indices_1 = [kids_bed_idx, table_idx, nightstand_idx]
    translations_1 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_1 = [(0.9, 0.3, 0.75), (0.4, 0.75, 0.4), (0.3, 0.3, 0.3)]
    orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Has desk
    num_objects_2 = 3
    class_label_indices_2 = [kids_bed_idx, desk_idx, nightstand_idx]
    translations_2 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_2 = [(0.9, 0.3, 0.75), (0.5, 0.75, 0.5), (0.3, 0.3, 0.3)]
    orientations_2 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Has dressing_table
    num_objects_3 = 3
    class_label_indices_3 = [kids_bed_idx, dressing_table_idx, nightstand_idx]
    translations_3 = [(0, 0.3, 0), (1.5, 0.75, 0), (-1.5, 0.3, 0)]
    sizes_3 = [(0.9, 0.3, 0.75), (0.5, 0.75, 0.4), (0.3, 0.3, 0.3)]
    orientations_3 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Scene 4: No table-type objects
    num_objects_4 = 3
    class_label_indices_4 = [kids_bed_idx, nightstand_idx, wardrobe_idx]
    translations_4 = [(0, 0.3, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    sizes_4 = [(0.9, 0.3, 0.75), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    orientations_4 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    scene_4 = create_scene(room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4)
    
    # Stack scenes
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print(f"Scene 1 (has table): {rewards[0].item()}")
    print(f"Scene 2 (has desk): {rewards[1].item()}")
    print(f"Scene 3 (has dressing_table): {rewards[2].item()}")
    print(f"Scene 4 (no table-type): {rewards[3].item()}")
    
    assert rewards.shape[0] == 4, "Should have 4 scenes"
    assert rewards[0].item() == 1.0, f"Scene 1: Has table, should return 1.0, got {rewards[0].item()}"
    assert rewards[1].item() == 1.0, f"Scene 2: Has desk, should return 1.0, got {rewards[1].item()}"
    assert rewards[2].item() == 1.0, f"Scene 3: Has dressing_table, should return 1.0, got {rewards[2].item()}"
    assert rewards[3].item() == -1.0, f"Scene 4: No table-type, should return -1.0, got {rewards[3].item()}"
    
    print("All tests passed!")