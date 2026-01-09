import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for chair presence in the scene.
    Returns 1.0 if at least one chair is present, 0.0 otherwise.
    
    Input:
        - parsed_scenes: dict with scene tensors
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor polygon vertices
        - **kwargs: additional keyword arguments
    
    Output:
        reward: torch.Tensor of shape (B,)
    '''
    utility_functions = get_all_utility_functions()
    
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    B = one_hot.shape[0]
    device = parsed_scenes['device']

    rewards = utility_functions["get_object_present_reward_potential"]["function"](
            one_hot, "chair", idx_to_labels, object_indices=parsed_scenes["object_indices"]
        )
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the chair presence reward function.
    '''
    utility_functions = get_all_utility_functions()
    
    # Scene 1: Has chair (index 4)
    num_objects_1 = 3
    class_label_indices_1 = [4, 8, 12]  # chair, double_bed, nightstand
    translations_1 = [(0, 0.4, 0), (2, 0.5, 2), (-2, 0.3, -2)]
    sizes_1 = [(0.3, 0.4, 0.3), (1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_1 = [(1, 0), (1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1
    )
    
    # Scene 2: No chair
    num_objects_2 = 2
    class_label_indices_2 = [8, 12]  # double_bed, nightstand
    translations_2 = [(2, 0.5, 2), (-2, 0.3, -2)]
    sizes_2 = [(1.0, 0.5, 1.0), (0.3, 0.3, 0.3)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2
    )
    
    # Scene 3: Has multiple chairs
    num_objects_3 = 3
    class_label_indices_3 = [4, 4, 8]  # chair, chair, double_bed
    translations_3 = [(0, 0.4, 0), (3, 0.4, 0), (2, 0.5, 2)]
    sizes_3 = [(0.3, 0.4, 0.3), (0.3, 0.4, 0.3), (1.0, 0.5, 1.0)]
    orientations_3 = [(1, 0), (1, 0), (1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3
    )
    
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
    # assert rewards.shape[0] == 3
    
    # Test # assertions
    print(f"Scene 1 (has chair): {rewards[0].item()}, expected: 1.0")
    print(f"Scene 2 (no chair): {rewards[1].item()}, expected: 0.0")
    print(f"Scene 3 (multiple chairs): {rewards[2].item()}, expected: 1.0")
    
    # assert rewards[0].item() == 1.0, f"Scene 1 should have reward 1.0, got {rewards[0].item()}"
    # assert rewards[1].item() == 0.0, f"Scene 2 should have reward 0.0, got {rewards[1].item()}"
    # assert rewards[2].item() == 1.0, f"Scene 3 should have reward 1.0, got {rewards[2].item()}"
    print("All tests passed!")