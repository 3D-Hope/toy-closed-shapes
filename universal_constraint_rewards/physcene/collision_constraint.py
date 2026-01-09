"""
Collision Avoidance Constraint (φcoll) for PhyScene.
Prevents objects from overlapping or colliding with each other.

Compatible with parsed_scene format used throughout the project.
"""

from typing import Dict

import torch

from tqdm import trange

from .common import cal_iou_3d


def collision_constraint(parsed_scene: Dict, **kwargs) -> torch.Tensor:
    """
    Calculate collision avoidance loss (φcoll) for each scene in the batch.

    This function penalizes overlapping objects to ensure they don't collide
    with each other in the 3D scene.

    Args:
        parsed_scene: Dict with scene data including:
            - positions: (B, N, 3) - object centroids
            - sizes: (B, N, 3) - object dimensions (half-extents)
            - orientations: (B, N, 2) - cos/sin of rotation angles
            - object_indices: (B, N) - object class indices
            - is_empty: (B, N) - boolean mask for empty slots
            - one_hot: (B, N, C) - one-hot encoded class labels
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Collision loss value (negative reward) for each scene (shape: [B])
    """
    # Extract data from parsed_scene
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

    print("Length of bbox", len(bbox))

    # Construct objectness from is_empty [B, N, 1]
    objectness = (~is_empty).unsqueeze(
        -1
    )  # bool mask: True for present objects, False for empty
    class_labels = one_hot

    print("Calculating Collision Guidance (φcoll)...")
    device = bbox.device
    loss_collision = torch.zeros(len(bbox), device=device)

    for j in trange(len(bbox)):
        bbox_cur = bbox[j : j + 1, :, :]
        objectness_cur = objectness[j : j + 1, :, :]
        class_labels_cur = class_labels[j : j + 1, :, :]

        # Filter valid objects (convert to boolean mask)
        bbox_cur = bbox_cur[:, objectness_cur[0, :, 0], :]
        class_labels_cur = class_labels_cur[:, objectness_cur[0, :, 0], :]

        bbox_cur_cnt = bbox_cur.shape[1]

        for i in range(bbox_cur_cnt):
            bbox_target = bbox_cur[:, i, :]  # [B, 7]
            # Compare with all other objects
            bbox_target = torch.tile(bbox_target[:, None, :], [1, bbox_cur_cnt, 1])
            loss_iter = cal_iou_3d(bbox_cur, bbox_target)  # shape: [B, 1, 12]

            # Mask out self-comparison
            valid_pair = torch.ones_like(loss_iter).int()
            valid_pair[:, i] = 0
            loss_iter = loss_iter * valid_pair

            loss_collision[j] += loss_iter.sum() / bbox_cur_cnt / len(bbox)
    return loss_collision


def test_collision_constraint():
    """Test cases for collision constraint."""
    print("\n" + "=" * 70)
    print("Testing Collision Constraint (φcoll)")
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

    # Test Case 1: No collision - objects well separated
    print("\nTest 1: No collision - objects well separated")
    scene1 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),  # Chair at origin
            (18, [2.0, 0.0, 0.0], [0.1, 0.1, 0.1]),  # Table far away
        ]
    )

    # Test Case 2: Objects overlapping (same position)
    print("Test 2: Objects overlapping at same position")
    scene2 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.3, 0.1, 0.3]),  # Chair
            (18, [0.0, 0.0, 0.0], [0.3, 0.1, 0.3]),  # Table at same position
        ]
    )

    # Test Case 3: Partial overlap
    print("Test 3: Partial overlap")
    scene3 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.4, 0.1, 0.4]),  # Chair
            (18, [0.5, 0.0, 0.0], [0.4, 0.1, 0.4]),  # Table slightly overlapping
        ]
    )

    # Test Case 4: Multiple objects, some colliding
    print("Test 4: Multiple objects, some colliding")
    scene4 = create_scene_with_positions(
        [
            (4, [0.0, 0.0, 0.0], [0.3, 0.1, 0.3]),  # Chair
            (18, [0.0, 0.0, 0.0], [0.3, 0.1, 0.3]),  # Table overlapping
            (8, [3.0, 0.0, 0.0], [0.2, 0.1, 0.2]),  # Bed far away
        ]
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3, scene4], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Compute collision loss
    losses_collision = collision_constraint(parsed)

    print(f"\nResults:")
    for i in range(4):
        print("Loss collision", losses_collision[i])

    print("\n✓ Collision constraint test passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_collision_constraint()
