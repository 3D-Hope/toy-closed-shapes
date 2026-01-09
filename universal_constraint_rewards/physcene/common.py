"""
Common utilities for PhyScene reward functions.
Contains shared functions and utilities used across all constraint implementations.
"""

import heapq

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from tqdm import trange


def cal_iou_3d(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate 3D IoU between two sets of bounding boxes.

    Args:
        box1: First set of boxes [B, N, 7] (x, y, z, w, l, h, angle)
        box2: Second set of boxes [B, M, 7] (x, y, z, w, l, h, angle)

    Returns:
        IoU values [B, N, M]
    """

    # Implementation of 3D IoU calculation
    # This is a simplified version - in practice you'd use a more robust implementation
    def box_volume(box):
        return box[..., 3] * box[..., 4] * box[..., 5]  # w * l * h

    def box_intersection_volume(box1, box2):
        # Simplified intersection calculation
        # In practice, this would be more complex for rotated boxes
        x1_min, x1_max = (
            box1[..., 0] - box1[..., 3] / 2,
            box1[..., 0] + box1[..., 3] / 2,
        )
        y1_min, y1_max = (
            box1[..., 1] - box1[..., 4] / 2,
            box1[..., 1] + box1[..., 4] / 2,
        )
        z1_min, z1_max = (
            box1[..., 2] - box1[..., 5] / 2,
            box1[..., 2] + box1[..., 5] / 2,
        )

        x2_min, x2_max = (
            box2[..., 0] - box2[..., 3] / 2,
            box2[..., 0] + box2[..., 3] / 2,
        )
        y2_min, y2_max = (
            box2[..., 1] - box2[..., 4] / 2,
            box2[..., 1] + box2[..., 4] / 2,
        )
        z2_min, z2_max = (
            box2[..., 2] - box2[..., 5] / 2,
            box2[..., 2] + box2[..., 5] / 2,
        )

        inter_x_min = torch.max(x1_min, x2_min)
        inter_x_max = torch.min(x1_max, x2_max)
        inter_y_min = torch.max(y1_min, y2_min)
        inter_y_max = torch.min(y1_max, y2_max)
        inter_z_min = torch.max(z1_min, z2_min)
        inter_z_max = torch.min(z1_max, z2_max)

        inter_x = torch.clamp(inter_x_max - inter_x_min, min=0)
        inter_y = torch.clamp(inter_y_max - inter_y_min, min=0)
        inter_z = torch.clamp(inter_z_max - inter_z_min, min=0)

        return inter_x * inter_y * inter_z

    vol1 = box_volume(box1)
    vol2 = box_volume(box2)
    inter_vol = box_intersection_volume(box1.unsqueeze(-2), box2.unsqueeze(-3))
    union_vol = vol1.unsqueeze(-1) + vol2.unsqueeze(-2) - inter_vol

    return inter_vol / (union_vol + 1e-8)


def draw_2d_gaussian(
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle: float,
    image_size: int = 256,
) -> np.ndarray:
    """
    Draw a 2D Gaussian distribution on an image.

    Args:
        center: Center coordinates (x, y)
        size: Size parameters (sigma_x, sigma_y)
        angle: Rotation angle in radians
        image_size: Size of the output image

    Returns:
        2D Gaussian array
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    covariance_matrix = np.array([[size[0] ** 2, 0], [0, size[1] ** 2]])
    rotation_covariance_matrix = rotation_matrix @ covariance_matrix @ rotation_matrix.T

    x = np.arange(0, image_size)
    y = np.arange(0, image_size)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()]).T - center

    try:
        inv_cov = np.linalg.inv(rotation_covariance_matrix)
        gaussian = np.exp(-0.5 * np.sum(xy @ inv_cov * xy, axis=1))
    except np.linalg.LinAlgError:
        gaussian = np.zeros(xy.shape[0])

    gaussian = gaussian.reshape(xx.shape)
    return gaussian


def heuristic_distance(node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
    """
    Calculate heuristic distance between two nodes (for A* pathfinding).

    Args:
        node1: First node coordinates (x, y)
        node2: Second node coordinates (x, y)

    Returns:
        Euclidean distance
    """
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def find_shortest_path(
    matrix: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    Find shortest path using A* algorithm.

    Args:
        matrix: 2D binary matrix (0=obstacle, 1=free)
        start: Starting coordinates (x, y)
        end: Ending coordinates (x, y)

    Returns:
        List of path coordinates or None if no path found
    """
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def is_valid(x, y):
        return (
            0 <= x < matrix.shape[0] and 0 <= y < matrix.shape[1] and matrix[x, y] == 1
        )

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_distance(start, end)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            new_node = (current[0] + dx, current[1] + dy)

            if not is_valid(new_node[0], new_node[1]):
                continue

            tentative_g = g_score[current] + 1

            if new_node not in g_score or tentative_g < g_score[new_node]:
                came_from[new_node] = current
                g_score[new_node] = tentative_g
                f_score[new_node] = tentative_g + heuristic_distance(new_node, end)
                heapq.heappush(open_set, (f_score[new_node], new_node))

    return None


def map_to_image_coordinate(
    point: Tuple[float, float], scale: float, image_size: int
) -> Tuple[int, int]:
    """
    Convert world coordinates to image coordinates.

    Args:
        point: World coordinates (x, y)
        scale: Scale factor
        image_size: Size of the image

    Returns:
        Image coordinates (x, y)
    """
    x, y = point
    x_image = int(x / scale * image_size / 2) + image_size // 2
    y_image = int(y / scale * image_size / 2) + image_size // 2
    return x_image, y_image


def image_to_map_coordinate(
    point: Tuple[int, int], scale: float, image_size: int
) -> Tuple[float, float]:
    """
    Convert image coordinates to world coordinates.

    Args:
        point: Image coordinates (x, y)
        scale: Scale factor
        image_size: Size of the image

    Returns:
        World coordinates (x, y)
    """
    x, y = point
    x_map = (x - image_size // 2) * 2 / image_size * scale
    y_map = (y - image_size // 2) * 2 / image_size * scale
    return x_map, y_map


def create_occupancy_map(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    orientations: torch.Tensor,
    scene_center: torch.Tensor,
    scale: float,
    image_size: int = 256,
) -> np.ndarray:
    """
    Create 2D occupancy map from 3D objects.

    Args:
        positions: Object positions [N, 3]
        sizes: Object sizes [N, 3]
        orientations: Object orientations [N, 2] (cos, sin)
        scene_center: Scene center coordinates [2]
        scale: Scale factor
        image_size: Size of the output image

    Returns:
        2D occupancy map
    """
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for pos, size, orient in zip(positions, sizes, orientations):
        center = map_to_image_coordinate(pos[:2].cpu().numpy(), scale, image_size)
        size_px = (
            int(size[0].item() / scale * image_size / 2),
            int(size[2].item() / scale * image_size / 2),
        )
        angle = torch.atan2(orient[1], orient[0]).item()

        # Draw rotated rectangle
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size_px, -angle / np.pi * 180)
        )
        pts = np.array(box_points, np.int32)
        pts = pts.reshape(-1, 1, 2)
        cv2.fillPoly(image, [pts], (255, 255, 255))

    return image


def get_region_center(region_mask: np.ndarray) -> Tuple[int, int]:
    """
    Get the center coordinates of a binary region.

    Args:
        region_mask: Binary mask of the region

    Returns:
        Center coordinates (x, y)
    """
    moments = cv2.moments(region_mask.astype(np.uint8))
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    else:
        # Fallback to geometric center
        y_coords, x_coords = np.where(region_mask)
        if len(x_coords) > 0:
            return int(np.mean(x_coords)), int(np.mean(y_coords))
        return 0, 0


def calc_path_loss(path: List[Tuple[int, int]], heat_map: np.ndarray) -> float:
    """
    Calculate loss along a path based on heat map values.

    Args:
        path: List of path coordinates
        heat_map: Heat map with cost values

    Returns:
        Total path loss
    """
    if not path:
        return 0.0

    total_loss = 0.0
    for x, y in path:
        if 0 <= x < heat_map.shape[0] and 0 <= y < heat_map.shape[1]:
            total_loss += heat_map[x, y]

    return total_loss / len(path) if path else 0.0
