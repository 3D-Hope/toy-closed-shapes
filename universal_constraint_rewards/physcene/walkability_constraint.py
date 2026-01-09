"""
Reachability Constraint (φreach) for PhyScene.
Ensures objects are reachable by an agent/robot in the scene.

Compatible with parsed_scene format used throughout the project.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from tqdm import trange

try:
    from .common import cal_iou_3d, draw_2d_gaussian, find_shortest_path
except:
    from common import cal_iou_3d, draw_2d_gaussian, find_shortest_path


def walkability_constraint(parsed_scene, floor_plan_args, **kwargs):
    print("Calculating Walkability Constraint ...")
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
    loss_walkable = torch.zeros(len(bbox), device=device)
    robot_width_real = 0.2
    robot_height_real = 1.5

    floor_plan_vertices = floor_plan_args["floor_plan_vertices"]
    floor_plan_faces = floor_plan_args["floor_plan_faces"]
    floor_plan_centroid = floor_plan_args["floor_plan_centroid"]
    # wakable loss can not be parallelized.
    # larger batch size will cause longer time.
    for i in trange(len(bbox)):
        bbox_cur = bbox[i : i + 1, :, :]
        objectness_cur = objectness[i : i + 1, :, :]
        bbox_cur = bbox_cur[:, objectness_cur[0, :, 0], :]
        class_labels_cur = class_labels[i : i + 1, objectness_cur[0, :, 0], :]
        bbox_cur_cnt = bbox_cur.shape[1]

        vertices, faces = floor_plan_vertices[i], floor_plan_faces[i]
        vertices = vertices - floor_plan_centroid[i]
        vertices = vertices[:, 0::2]
        scale = np.abs(vertices).max() + 0.2
        # remove objects in a high level, such as ceiling lamp
        bbox_floor = bbox_cur[0, bbox_cur[0, :, 2] < robot_height_real]
        class_labels_cur = class_labels_cur[0, bbox_cur[0, :, 2] < robot_height_real]

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_width_real / scale * image_size / 2)

        def map_to_image_coordinate(point):
            x, y = point
            x_image = int(x / scale * image_size / 2) + image_size / 2
            y_image = int(y / scale * image_size / 2) + image_size / 2
            return x_image, y_image

        def image_to_map_coordinate(point):
            x, y = point
            x_map = (x - image_size / 2) * 2 / image_size * scale
            y_map = (y - image_size / 2) * 2 / image_size * scale
            return x_map, y_map

        # draw floor plan
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [map_to_image_coordinate(v) for v in face_vertices]
            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)

        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        # draw bboxes
        floor_plan_mask = image[:, :, 0] == 255

        box_heat_map = np.zeros((image_size, image_size), dtype=np.uint8)
        handle_lst = []
        for box, class_label in zip(bbox_floor, class_labels_cur):
            box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            box = box.cpu().detach().numpy()
            center = map_to_image_coordinate(box[:2])
            # full size
            size = (
                int(box[3] / scale * image_size / 2),
                int(box[4] / scale * image_size / 2),
            )
            angle = box[-1]

            # calculate box vertices
            # # [1] #original box
            box_points = cv2.boxPoints(
                ((center[0], center[1]), size, -angle / np.pi * 180)
            )

            box_points = np.intp(box_points)

            cv2.drawContours(
                image, [box_points], 0, (0, 255, 0), robot_width
            )  # Green (BGR)

            cv2.fillPoly(image, [box_points], (0, 255, 0))
            cv2.drawContours(
                box_mask, [box_points], 0, (0, 255, 0), robot_width
            )  # Green (BGR)

            cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
            box_mask = box_mask[:, :, 1] == 255
            if min(size) != 0:
                gaussian = draw_2d_gaussian(
                    (int(center[0]), int(center[1])), size, -angle, image_size
                )
                box_heat_map = box_heat_map + gaussian * box_mask
            # box_heat_map = box_heat_map + gaussian

        box_heat_map = floor_plan_mask * box_heat_map

        # add wall boundary
        wall_dist_transform = cv2.distanceTransform(
            floor_plan_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5
        )
        wall_dist_transform[wall_dist_transform != 0] = (
            1.0 / wall_dist_transform[wall_dist_transform != 0]
        )
        box_wall_heat_map = box_heat_map  # + wall_dist_transform
        box_wall_heat_map = (
            box_wall_heat_map + (1 - floor_plan_mask) * box_heat_map.max()
        )

        # visual
        box_wall_heat_map_image = cv2.normalize(
            box_wall_heat_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
        )
        box_wall_heat_map_image = cv2.applyColorMap(
            box_wall_heat_map_image, cv2.COLORMAP_JET
        )
        cv2.imwrite("mesh_image_with_boxes2.png", box_wall_heat_map_image)
        cv2.imwrite("mesh_image_with_boxes1.png", image)

        walkable_map = image[:, :, 0].copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8
        )

        # [1] region reachable,
        # find shortest path from region_1 to region_2
        if num_labels > 2:
            area_1 = np.zeros_like(walkable_map)
            area_2 = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 1
                if mask.sum() > area_2.sum():
                    area_2 = mask.copy()
                if area_2.sum() > area_1.sum():
                    area_2, area_1 = area_1.copy(), area_2.copy()
            if area_2.sum() > 100:
                minimum_area_1 = np.argmax(
                    cv2.distanceTransform(
                        area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5
                    )
                )
                minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
                minimum_area_2 = np.argmax(
                    cv2.distanceTransform(
                        area_2.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5
                    )
                )
                minimum_area_2_position = np.unravel_index(minimum_area_2, area_2.shape)

                shortest_path_image = box_wall_heat_map_image.copy()
                # draw circle for start and end point
                cv2.circle(
                    shortest_path_image,
                    [minimum_area_1_position[1], minimum_area_1_position[0]],
                    2,
                    (255, 255, 255),
                    -1,
                )
                cv2.circle(
                    shortest_path_image,
                    [minimum_area_2_position[1], minimum_area_2_position[0]],
                    2,
                    (255, 255, 255),
                    -1,
                )
                cv2.imwrite("walkable_map.png", shortest_path_image)
                shortest_path = find_shortest_path(
                    box_wall_heat_map,
                    (minimum_area_1_position[0], minimum_area_1_position[1]),
                    (minimum_area_2_position[0], minimum_area_2_position[1]),
                )
                if shortest_path == None:
                    continue
                for r, c in shortest_path:
                    shortest_path_image[r, c] = (255, 255, 255)  # white

                cv2.imwrite("shortest_path_image.png", shortest_path_image)
                loss_walkable[i] += calc_loss_on_path(
                    image,
                    shortest_path,
                    robot_width,
                    robot_width_real,
                    robot_height_real,
                    map_to_image_coordinate,
                    image_to_map_coordinate,
                    scale,
                    image_size,
                    bbox,
                    bbox_floor,
                )

        # [2] handle reachable
        # find shortest path from region_1 to handle_1(front center point of the open bbox)
        if len(handle_lst) > 0:
            shortest_path_image = box_wall_heat_map_image.copy()
            area_1 = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 1
                if mask.sum() > area_1.sum():
                    area_1 = mask.copy()
            # start point
            minimum_area_1 = np.argmax(
                cv2.distanceTransform(
                    area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5
                )
            )
            minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
            cv2.circle(
                shortest_path_image,
                [minimum_area_1_position[1], minimum_area_1_position[0]],
                2,
                (255, 255, 255),
                -1,
            )
            for handle_point in handle_lst:
                # # [2] #
                # handle_point = random.choice(handle_lst)
                # find the largest area

                if area_1.sum() > 100:
                    # end point
                    cv2.circle(
                        shortest_path_image,
                        [int(handle_point[0]), int(handle_point[1])],
                        2,
                        (255, 255, 255),
                        -1,
                    )
                    shortest_path = find_shortest_path(
                        box_wall_heat_map,
                        (minimum_area_1_position[0], minimum_area_1_position[1]),
                        (int(handle_point[1]), int(handle_point[0])),
                    )
                    if shortest_path == None:
                        continue
                    for r, c in shortest_path:
                        shortest_path_image[r, c] = (255, 255, 255)  # white
                    loss_walkable[i] += calc_loss_on_path(
                        image,
                        shortest_path,
                        robot_width,
                        robot_width_real,
                        robot_height_real,
                        map_to_image_coordinate,
                        image_to_map_coordinate,
                        scale,
                        image_size,
                        bbox,
                        bbox_floor,
                    )
            cv2.imwrite("shortest_path_image.png", shortest_path_image)

    return loss_walkable


def calc_loss_on_path(
    image,
    shortest_path,
    robot_width,
    robot_width_real,
    robot_height_real,
    map_to_image_coordinate,
    image_to_map_coordinate,
    scale,
    image_size,
    bbox,
    bbox_floor,
):
    loss_walkable = 0.0
    box_mask = image[:, :, 1] == 255
    bbox_path = []
    path_count = 0
    for i in range(len(shortest_path)):
        if box_mask[shortest_path[i]]:
            if path_count % robot_width == 0:
                center_map = image_to_map_coordinate(
                    (shortest_path[i][1], shortest_path[i][0])
                )
                angle = 0.0
                box = np.array(
                    [
                        *center_map,
                        0,
                        robot_width_real,
                        robot_width_real,
                        robot_height_real,
                        angle,
                    ]
                )
                bbox_path.append(box)

            path_count += 1
    if bbox_path == []:
        return loss_walkable
    for box in bbox_path:
        center = map_to_image_coordinate(box[:2])
        size = (
            int(box[3] / scale * image_size / 2),
            int(box[4] / scale * image_size / 2),
        )
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(((center[0], center[1]), size, -angle / np.pi * 180))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0, (0, 255, 255), robot_width)  # Yellow
    cv2.imwrite("shortest_path_box_image.png", image)
    bbox_path = np.expand_dims(np.stack(bbox_path, 0), 0)

    # xyz
    bbox_cnt_path = bbox_path.shape[1]
    bbox_floor = bbox_floor[None, :, :]
    bbox_floor_cur_cnt = bbox_floor.shape[1]
    for bbox_cnt_idx in range(bbox_floor_cur_cnt):
        bbox_target = bbox_floor[:, bbox_cnt_idx, :]
        bbox_target = torch.tile(bbox_target[:, None, :], [1, bbox_cnt_path, 1])

        loss_walkable += (
            cal_iou_3d(
                torch.tensor(
                    bbox_path, device=bbox_target.device, dtype=bbox_target.dtype
                ),
                bbox_target,
            ).sum()
            / len(bbox)
            / bbox_floor_cur_cnt
        )
    return loss_walkable


def test_reachability_constraint():
    """Test cases for reachability constraint."""
    print("\n" + "=" * 70)
    print("Testing Reachability Constraint (φreach)")
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

    # Test Case 1: Objects with clear paths (reachable)
    print("\nTest 1: Objects with clear paths (reachable)")
    scene1 = create_scene_with_positions(
        [
            (4, [0.0, 0.5, 0.0], [0.2, 0.3, 0.2]),  # Chair at center
            (18, [2.0, 0.5, 0.0], [0.3, 0.2, 0.3]),  # Table with space
        ]
    )

    # Test Case 2: Densely packed objects (may be unreachable)
    print("Test 2: Densely packed objects (harder to reach)")
    scene2 = create_scene_with_positions(
        [
            (4, [0.0, 0.5, 0.0], [0.5, 0.3, 0.5]),  # Large chair
            (18, [0.6, 0.5, 0.0], [0.5, 0.2, 0.5]),  # Large table nearby
            (8, [1.2, 0.5, 0.0], [0.5, 0.3, 0.5]),  # Large bed
        ]
    )

    # Test Case 3: Sparse layout (very reachable)
    print("Test 3: Sparse layout (easily reachable)")
    scene3 = create_scene_with_positions(
        [
            (4, [-2.0, 0.5, -2.0], [0.2, 0.3, 0.2]),  # Chair in corner
            (18, [2.0, 0.5, 2.0], [0.3, 0.2, 0.3]),  # Table in opposite corner
        ]
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3], dim=0)

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    import pickle

    with open(
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/universal_constraint_rewards/floor_plan_args.pkl",
        "rb",
    ) as f:
        floor_plan_args = pickle.load(f)

    print(f"Floor plan args: {floor_plan_args.keys()}")
    print(f"Floor plan vertices: {len(floor_plan_args['floor_plan_vertices'])}")
    print(f"Floor plan vertices: {floor_plan_args['floor_plan_vertices'][0].shape}")
    print(f"Floor plan faces: {len(floor_plan_args['floor_plan_faces'])}")
    print(f"Floor plan faces: {floor_plan_args['floor_plan_faces'][0].shape}")
    print(f"Floor plan centroid: {len(floor_plan_args['floor_plan_centroid'])}")
    print(f"Floor plan centroid: {floor_plan_args['floor_plan_centroid'][0].shape}")
    print(f"Room outer box: {len(floor_plan_args['room_outer_box'])}")
    print(f"Room outer box: {floor_plan_args['room_outer_box'][0].shape}")

    walkability_loss = walkability_constraint(parsed, floor_plan_args=floor_plan_args)
    print(f"Walkability loss: {walkability_loss}")


if __name__ == "__main__":
    test_reachability_constraint()
