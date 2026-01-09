import math

import numpy as np
import torch

from universal_constraint_rewards.commons import idx_to_labels

# Objects that should be axis-aligned (typically large furniture placed against walls)
AXIS_ALIGNED_OBJECTS = {
    "double_bed",
    "single_bed",
    "kids_bed",
    "wardrobe",
    "bookshelf",
    "cabinet",
    "children_cabinet",
    "desk",
    "dressing_table",
    "tv_stand",
    "table",
}

# TODO: Ashok I have changed this function to take extra room_type parameter to make it more flexible, fix everywhere this function is used.
def compute_axis_alignment_reward(parsed_scenes, **kwargs):
    """
    Reward objects for being axis-aligned (parallel/perpendicular to walls).

    Penalizes angular deviation from 0°, 90°, 180°, 270° for furniture that
    should typically be placed against walls.

    Args:
        parsed_scenes: Dict with keys:
            - 'orientations': (B, N, 2) - [cos_theta, sin_theta] for each object
            - 'object_indices': (B, N) - object class indices
            - 'is_empty': (B, N) - mask for empty slots
        idx_to_labels: Dict mapping indices to object class names

    Returns:
        rewards: (B,) - alignment reward per scene (negative of total violation)
    """
    room_type = kwargs["room_type"]
    orientations = parsed_scenes["orientations"]  # (B, N, 2)
    object_indices = parsed_scenes["object_indices"]  # (B, N)
    is_empty = parsed_scenes["is_empty"]  # (B, N)

    batch_size, num_objects = orientations.shape[0], orientations.shape[1]
    device = orientations.device
    room_type = kwargs["room_type"]
    # Convert [cos, sin] to angles in radians
    axis_aligned_indices = [
        int(idx)
        for idx, label in idx_to_labels[room_type].items()
        if label in AXIS_ALIGNED_OBJECTS
    ]

    # Create mask for objects that should be axis-aligned
    should_align = torch.zeros_like(is_empty, dtype=torch.bool)
    for idx in axis_aligned_indices:
        should_align |= object_indices == idx

    # Mask for valid objects to check (non-empty AND should be aligned)
    valid_mask = ~is_empty & should_align  # (B, N)
    cos_theta = orientations[:, :, 0]  # (B, N)
    sin_theta = orientations[:, :, 1]  # (B, N)
    angles = torch.atan2(sin_theta, cos_theta)  # (B, N) in range [-pi, pi]
    # Print only angles of valid positions (axis-aligned and not empty)
    # valid_mask: (B, N), angles: (B, N)
    angles_deg = np.round(np.rad2deg(angles.cpu().numpy())).astype(int)  # (B, N)
    valid_mask_np = valid_mask.cpu().numpy()
    # object_indices_np = object_indices.cpu().numpy()
    # abnormal_indices = []
    # for b in range(batch_size):
    #     for n in range(num_objects):
    #         if valid_mask_np[b, n]:
    #             angle = angles_deg[b, n]
    #             # Acceptable: within 3 deg of 0, 90, 180, -90, -180
    #             if not any(abs(angle - ref) <= 3 for ref in [0, 90, 180, -90, -180]):
    #                 abnormal_indices.append((b, n, object_indices_np[b, n], angle))
    # if abnormal_indices:
    #     print("Abnormal angles detected (batch, obj_idx, class_idx, angle):")
    #     for b, n, class_idx, angle in abnormal_indices:
    #         print(f"  Batch {b}, Object {n}, ClassIdx {class_idx}, Angle {angle}")
    # Identify which objects should be axis-aligned

    # Compute deviation from nearest axis-aligned angle (0, 90, 180, -90, -180) in degrees
    axis_angles = np.array([0, 90, 180, -90, -180])
    deviation = np.zeros_like(angles_deg, dtype=float)
    for b in range(batch_size):
        for n in range(num_objects):
            if valid_mask_np[b, n]:
                angle = angles_deg[b, n]
                deviation[b, n] = np.min(np.abs(angle - axis_angles))
            else:
                deviation[b, n] = 0.0
    total_violation = (deviation * valid_mask_np).sum(axis=1)  # (B,)
    reward = -total_violation

    return reward


if __name__ == "__main__":
    args = np.load(
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/reward_func_args_for_first_10_scenes.npy",
        allow_pickle=True,
    )
    # print("loaded ", args)
    start = 0
    end = 15
    # for key in args.item().keys():
    #     if isinstance(args.item()[key], np.ndarray):
    #         args.item()[key] = args.item()[key][start:end]
    #     elif isinstance(args.item()[key], list):
    #         args.item()[key] = args.item()[key][start:end]
    #     print(f"{key}: {args.item()[key].shape if hasattr(args.item()[key], 'shape') else len(args.item()[key])}")
    # Enable visualization for testing
    result = compute_axis_alignment_reward(
        **args.item(),
        # save_viz=True, viz_dir="./viz"
    )
    # print(result[start:end])
    print(result)
