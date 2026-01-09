import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Relaxed reward for ceiling lamps being near bed corners.
    Uses soft assignment: each lamp contributes based on proximity to nearest corner.
    Reward = average over lamps of: exp(-min_distance^2 / (2*sigma^2)) - 1
    This gives smoother gradients and is more learnable.
    '''
    utility_functions = get_all_utility_functions()
    
    B = parsed_scenes['positions'].shape[0]
    N = parsed_scenes['positions'].shape[1]
    rewards = torch.zeros(B, device=parsed_scenes['device'])
    
    # Find class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    ceiling_lamp_idx = labels_to_idx["ceiling_lamp"]
    bed_indices = [labels_to_idx["double_bed"], labels_to_idx["single_bed"], labels_to_idx["kids_bed"]]
    
    sigma = 0.5  # Distance scale in meters for exponential decay
    
    for i in range(B):
        positions = parsed_scenes['positions'][i]  # (N, 3)
        sizes = parsed_scenes['sizes'][i]  # (N, 3)
        orientations = parsed_scenes['orientations'][i]  # (N, 2)
        object_indices = parsed_scenes['object_indices'][i]  # (N,)
        is_empty = parsed_scenes['is_empty'][i]  # (N,)
        
        # Find ceiling lamps
        lamp_mask = (object_indices == ceiling_lamp_idx) & (~is_empty)
        lamp_positions = positions[lamp_mask]  # (num_lamps, 3)
        
        # Find beds
        bed_mask = torch.zeros_like(is_empty, dtype=torch.bool)
        for bed_idx in bed_indices:
            bed_mask |= (object_indices == bed_idx)
        bed_mask &= (~is_empty)
        
        if lamp_mask.sum() == 0 or bed_mask.sum() == 0:
            rewards[i] = -1.0  # No lamps or beds
            continue
        
        # Get bed info (use first bed if multiple)
        bed_idx_tensor = torch.where(bed_mask)[0][0]
        bed_pos = positions[bed_idx_tensor]  # (3,)
        bed_size = sizes[bed_idx_tensor]  # (3,)
        bed_orient = orientations[bed_idx_tensor]  # (2,)
        
        # Calculate bed corners in XZ plane
        cos_theta = bed_orient[0]
        sin_theta = bed_orient[1]
        
        # Half-sizes
        half_x = bed_size[0]
        half_z = bed_size[2]
        
        # Four corners relative to bed center (in local frame)
        local_corners = torch.tensor([
            [half_x, half_z],
            [half_x, -half_z],
            [-half_x, half_z],
            [-half_x, -half_z]
        ], device=parsed_scenes['device'])
        
        # Rotate corners to world frame
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], device=parsed_scenes['device'])
        
        world_corners = torch.matmul(local_corners, rotation_matrix.T)  # (4, 2)
        world_corners[:, 0] += bed_pos[0]  # Add bed x
        world_corners[:, 1] += bed_pos[2]  # Add bed z
        
        # For each lamp, compute proximity to nearest corner
        lamp_rewards = []
        for lamp_pos in lamp_positions:
            lamp_xz = torch.tensor([lamp_pos[0], lamp_pos[2]], device=parsed_scenes['device'])
            distances = torch.norm(world_corners - lamp_xz.unsqueeze(0), dim=1)  # (4,)
            min_dist = distances.min()
            
            # Exponential decay: reward ranges from -1 (far) to 0 (at corner)
            lamp_reward = torch.exp(-min_dist**2 / (2 * sigma**2)) - 1.0
            lamp_rewards.append(lamp_reward)
        
        # Average reward over all lamps
        rewards[i] = torch.stack(lamp_rewards).mean()
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    utility_functions = get_all_utility_functions()
    create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # Scene 1: 4 lamps perfectly at bed corners (satisfies well)
    # Bed at (0, 0.5, 0) with size (1.0, 0.4, 1.5) means corners at (±1.0, ±1.5)
    scene_1 = create_scene(
        room_type=room_type,
        num_objects=5,
        class_label_indices=[8, 3, 3, 3, 3],  # double_bed, 4 ceiling_lamps
        translations=[(0, 0.5, 0), (1.0, 2.8, 1.5), (1.0, 2.8, -1.5), (-1.0, 2.8, 1.5), (-1.0, 2.8, -1.5)],
        sizes=[(1.0, 0.4, 1.5), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2)],
        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    )
    
    # Scene 2: 4 lamps moderately offset from corners
    scene_2 = create_scene(
        room_type=room_type,
        num_objects=5,
        class_label_indices=[8, 3, 3, 3, 3],
        translations=[(0, 0.5, 0), (1.5, 2.8, 2.0), (1.5, 2.8, -2.0), (-1.5, 2.8, 2.0), (-1.5, 2.8, -2.0)],
        sizes=[(1.0, 0.4, 1.5), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2), (0.2, 0.05, 0.2)],
        orientations=[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    )
    
    # Scene 3: No lamps (fails)
    scene_3 = create_scene(
        room_type=room_type,
        num_objects=1,
        class_label_indices=[8],
        translations=[(0, 0.5, 0)],
        sizes=[(1.0, 0.4, 1.5)],
        orientations=[(1, 0)]
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
    print("Expected: Scene 1 close to 0 (perfect), Scene 2 more negative (offset), Scene 3 = -1.0")
    assert rewards.shape[0] == 3
    assert rewards[0] >= -0.1, f"Scene 1 should have reward close to 0 (got {rewards[0]})"
    assert rewards[1] < rewards[0] and rewards[1] > -0.9, f"Scene 2 should have moderate penalty (got {rewards[1]})"
    assert rewards[2] == -1.0, f"Scene 3 should have reward -1.0, got {rewards[2]}"