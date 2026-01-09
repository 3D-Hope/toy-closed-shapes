import torch
import math
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward function for bed positioned at room center (0, 0, 0).
    
    Input:
        - parsed_scenes: dict with batched scene data
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string, Example: "bedroom" or "livingroom"
        - floor_polygons: list of ordered floor polygon vertices
        - **kwargs: additional keyword arguments

    Output:
        reward: torch.Tensor of shape (B,) - higher reward when bed is closer to center
    '''
    positions = parsed_scenes['positions']  # (B, N, 3)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    device = parsed_scenes['device']
    
    B, N, _ = positions.shape
    
    # Find bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = []
    for bed_type in ['double_bed', 'single_bed', 'kids_bed']:
        if bed_type in labels_to_idx:
            bed_classes.append(labels_to_idx[bed_type])
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find bed objects in this scene
        bed_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for bed_idx in bed_classes:
            bed_mask |= (object_indices[b] == bed_idx)
        bed_mask &= ~is_empty[b]  # Exclude empty slots
        
        if bed_mask.sum() == 0:
            # No bed found - give large penalty
            rewards[b] = -10.0
            continue
        
        # Get all bed positions
        bed_positions = positions[b][bed_mask]  # (num_beds, 3)
        
        # Calculate distance from center (0, 0, 0) for each bed
        # We care about XZ distance primarily, but include Y for completeness
        distances = torch.norm(bed_positions, dim=1)  # (num_beds,)
        
        # Use the minimum distance (closest bed to center)
        min_distance = distances.min()
        
        # Reward function: exponential decay based on distance
        # Perfect center (distance=0) gives reward=0
        # Distance increases -> reward becomes more negative
        # Tolerance threshold of 0.3m gives reward ~ -0.9
        # Distance of 1m gives reward ~ -10
        rewards[b] = -10.0 * (1.0 - torch.exp(-min_distance / 0.3))
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the bed_at_room_center reward function.
    '''
    from dynamic_constraint_rewards.utilities import get_all_utility_functions
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    double_bed_idx = labels_to_idx['double_bed']
    nightstand_idx = labels_to_idx['nightstand']
    wardrobe_idx = labels_to_idx['wardrobe']
    
    # Scene 1: Bed exactly at center (0, 0, 0) - should get high reward (close to 0)
    num_objects_1 = 3
    class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    translations_1 = [(0.0, 0.4, 0.0), (2.0, 0.3, 2.0), (-2.0, 0.5, -2.0)]
    sizes_1 = [(1.0, 0.4, 1.0), (0.3, 0.3, 0.3), (0.5, 0.5, 0.8)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # Scene 2: Bed at (0.2, 0.4, 0.15) - slightly off center, should get moderate reward
    num_objects_2 = 2
    class_label_indices_2 = [double_bed_idx, nightstand_idx]
    translations_2 = [(0.2, 0.4, 0.15), (2.5, 0.3, 2.5)]
    sizes_2 = [(1.0, 0.4, 1.0), (0.3, 0.3, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # Scene 3: Bed far from center at (3.0, 0.4, 3.0) - should get low reward (very negative)
    num_objects_3 = 2
    class_label_indices_3 = [double_bed_idx, wardrobe_idx]
    translations_3 = [(3.0, 0.4, 3.0), (-2.0, 0.5, -1.0)]
    sizes_3 = [(1.0, 0.4, 1.0), (0.5, 0.5, 0.8)]
    orientations_3 = [(1, 0), (1, 0)]
    scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # Scene 4: No bed in scene - should get large penalty
    num_objects_4 = 2
    class_label_indices_4 = [nightstand_idx, wardrobe_idx]
    translations_4 = [(1.0, 0.3, 1.0), (-1.0, 0.5, -1.0)]
    sizes_4 = [(0.3, 0.3, 0.3), (0.5, 0.5, 0.8)]
    orientations_4 = [(1, 0), (1, 0)]
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
    print(f"Scene 1 (bed at center): {rewards[0].item():.4f}")
    print(f"Scene 2 (bed slightly off): {rewards[1].item():.4f}")
    print(f"Scene 3 (bed far away): {rewards[2].item():.4f}")
    print(f"Scene 4 (no bed): {rewards[3].item():.4f}")
    
    assert rewards.shape[0] == 4, f"Expected 4 rewards, got {rewards.shape[0]}"
    
    # Scene 1: Bed at center should have reward close to 0 (>-1.0)
    assert rewards[0] > -1.0, f"Scene 1 (bed at center) should have high reward, got {rewards[0].item()}"
    
    # Scene 2: Slightly off center should have moderate negative reward
    assert -3.0 < rewards[1] < -0.5, f"Scene 2 (slightly off) should have moderate reward between -3.0 and -0.5, got {rewards[1].item()}"
    
    # Scene 3: Far from center should have very negative reward
    assert rewards[2] < -8.0, f"Scene 3 (far away) should have very negative reward (<-8.0), got {rewards[2].item()}"
    
    # Scene 4: No bed should have the worst penalty
    assert rewards[3] == -10.0, f"Scene 4 (no bed) should have penalty of -10.0, got {rewards[3].item()}"
    
    # Scene 1 should be better than Scene 2
    assert rewards[0] > rewards[1], f"Scene 1 should be better than Scene 2"
    
    # Scene 2 should be better than Scene 3
    assert rewards[1] > rewards[2], f"Scene 2 should be better than Scene 3"
    
    print("All tests passed!")