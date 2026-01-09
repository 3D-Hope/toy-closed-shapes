"""
Room Layout Constraint (φlayout) for PhyScene.
Ensures objects stay within room boundaries and follow proper layout constraints.

Compatible with parsed_scene format used throughout the project.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from tqdm import trange

from .common import cal_iou_3d


def room_layout_constraint(
    parsed_scene: Dict, floor_plan_args: Dict, **kwargs
) -> torch.Tensor:
    """
    Calculate room layout loss (φlayout) for each scene in the batch.

    This function penalizes objects that are outside the room boundaries.

    Args:
        parsed_scene: Dict with scene data including:
            - positions: (B, N, 3) - object centroids
            - sizes: (B, N, 3) - object dimensions (half-extents)
            - orientations: (B, N, 2) - cos/sin of rotation angles
            - is_empty: (B, N) - boolean mask for empty slots
            - one_hot: (B, N, C) - one-hot encoded class labels
        room_outer_box: torch.Tensor - room boundary boxes [B, M, 7]
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Room layout loss value (negative reward) for each scene (shape: [B])
    """
    positions = parsed_scene["positions"]  # (B, N, 3)
    sizes = parsed_scene["sizes"]  # (B, N, 3) - half-extents
    orientations = parsed_scene["orientations"]  # (B, N, 2) - [cos_theta, sin_theta]
    is_empty = parsed_scene["is_empty"]  # (B, N)
    one_hot = parsed_scene["one_hot"]  # (B, N, C)

    # Convert to bbox format [B, N, 7] (x, y, z, w, l, h, angle)
    # Note: sizes are half-extents, so full size is 2 * sizes
    batch_size, num_objects = positions.shape[:2]
    device = positions.device

    # Compute angle from cos/sin
    angles = torch.atan2(orientations[:, :, 1], orientations[:, :, 0])  # (B, N)

    # Construct bbox [B, N, 7]
    bbox = torch.cat(
        [
            positions,  # x, y, z
            sizes * 2,  # w, l, h (convert half-extents to full size)
            angles.unsqueeze(-1),  # angle
        ],
        dim=-1,
    )

    # Construct objectness from is_empty [B, N, 1]
    objectness = (~is_empty).unsqueeze(-1)
    class_labels = one_hot

    print("Calculating Room-layout Guidance ...")
    device = bbox.device
    loss_room_layout = torch.zeros(len(bbox), device=device)
    bbox_outer = torch.tensor(
        floor_plan_args["room_outer_box"], device=device, dtype=torch.float32
    )
    bbox_cnt_room = bbox_outer.shape[1]

    for j in trange(len(bbox)):
        bbox_cur = bbox[j : j + 1, :, :]
        objectness_cur = objectness[j : j + 1, :, :]

        bbox_cur = bbox_cur[:, objectness_cur[0, :, 0], :]
        bbox_cur_cnt = bbox_cur.shape[1]
        bbox_outer_cur = bbox_outer[j : j + 1, :, :]

        for i in range(bbox_cur_cnt):
            bbox_target = bbox_cur[:, i, :]  # [1, 7]
            bbox_target = torch.tile(bbox_target[:, None, :], [1, bbox_cnt_room, 1])
            loss_room_layout[j] += (
                cal_iou_3d(bbox_outer_cur, bbox_target).sum() / len(bbox) / bbox_cur_cnt
            )
    return loss_room_layout


def test_room_layout_constraint():
    """Test cases for room layout constraint."""
    print("\n" + "=" * 70)
    print("Testing Room Layout Constraint (φlayout)")
    print("=" * 70)

    device = "cpu"
    num_classes = 22
    num_objects = 12

    # Helper to create a scene with specific positions and sizes
    def create_scene_with_positions(objects_data):
        """
        objects_data: list of tuples (obj_type, position, size)
        """
        scene = torch.zeros(num_objects, 30, device=device)

        for i, (obj_type, pos, size) in enumerate(objects_data):
            if i >= num_objects:
                break
            scene[i, obj_type] = 1.0  # One-hot
            scene[i, num_classes : num_classes + 3] = torch.tensor(pos, device=device)
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                size, device=device
            )
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        # Fill remaining with empty
        for i in range(len(objects_data), num_objects):
            scene[i, 21] = 1.0  # Empty class

        return scene

    # Test Case 1: Objects completely inside room bounds (should have low IoU with walls)
    print("\nTest 1: Objects within room bounds")
    scene1 = create_scene_with_positions(
        [
            (4, [0.0, 1.0, 0.0], [0.3, 0.3, 0.3]),  # Chair at room center
            (18, [1.0, 0.5, 1.0], [0.4, 0.2, 0.4]),  # Table near center
        ]
    )

    # Test Case 2: Objects completely outside room bounds (should have high IoU with walls)
    print("Test 2: Objects outside room bounds")
    scene2 = create_scene_with_positions(
        [
            (4, [10.0, 10.0, 10.0], [0.3, 0.3, 0.3]),  # Chair far outside
            (18, [-10.0, -5.0, -10.0], [0.4, 0.2, 0.4]),  # Table far outside
        ]
    )

    # Test Case 3: Mixed - one inside, one outside (should have medium IoU)
    print("Test 3: Mixed - some inside, some outside")
    scene3 = create_scene_with_positions(
        [
            (4, [0.0, 1.0, 0.0], [0.3, 0.3, 0.3]),  # Chair inside
            (18, [10.0, 5.0, 10.0], [0.4, 0.2, 0.4]),  # Table outside
        ]
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # TODO: room_outer_box ??
    # b, num of sides of floor polygon, 7
    room_outer_box = torch.rand(4, 50, 7)  # Dummy data for testing

    # Compute room layout loss for each scene
    losses = room_layout_constraint(parsed, room_outer_box)

    print(f"\nResults:")
    print(f"Scene 1 (within bounds): {losses[0]:.4f} (expected: ~0)")
    print(f"Scene 2 (outside bounds): {losses[1]:.4f} (expected: > 0)")
    print(f"Scene 3 (mixed): {losses[2]:.4f} (expected: > 0)")

    # Verify
    assert losses[0] >= 0, "Room layout loss should be non-negative"
    assert losses[1] > losses[0], "Objects outside bounds should have higher loss"
    assert losses[2] > 0, "Mixed scene should have positive loss"

    print("\n✓ All room layout constraint tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_room_layout_constraint()
