import argparse
import os
import pickle
import multiprocessing
from multiprocessing import Pool, cpu_count, get_context
from concurrent.futures import ProcessPoolExecutor
import functools

import cv2
import numpy as np
import open3d as o3d
import torch

multiprocessing.set_start_method('spawn', force=True)

from kaolin.ops.mesh import check_sign
from physcene_utils import (
    bbox_overlap,
    cal_walkable_metric,
    calc_bbox_masks,
    get_textured_objects,
    map_to_image_coordinate,
)
from threed_front.datasets.threed_future_dataset import ThreedFutureDataset
from threed_front.evaluation import ThreedFrontResults
from threed_front.utils import PATH_TO_PICKLED_3D_FUTURE_MODEL_REMESH as PATH_TO_PICKLED_3D_FUTURE_MODEL
from tqdm import tqdm

from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
    update_data_file_paths,
)
from steerable_scene_generation.datasets.custom_scene.threed_front_encoding import (
    get_dataset_raw_and_encoded,
)


def _process_single_scene_wall_overlap(args):
    """
    Process wall overlap and walkable metrics for a single scene.
    Returns (scene_idx, walkable_rate, accessable_rate, box_wall_count, box_count)
    """
    scene_idx, scene_layout, encoded_dataset, cfg, robot_real_width = args
    
    raw_item = encoded_dataset[scene_idx]
    floor_plan_vertices = raw_item["floor_plan_vertices"]
    floor_plan_faces = raw_item["floor_plan_faces"]
    floor_plan_centroid = raw_item["floor_plan_centroid"]

    valid_idx = np.ones_like(scene_layout["class_labels"][:, 0], dtype=bool)
    class_labels = scene_layout["class_labels"][valid_idx]

    bbox = np.concatenate(
        [
            scene_layout["translations"][valid_idx],
            scene_layout["sizes"][valid_idx],
            scene_layout["angles"][valid_idx],
        ],
        axis=-1,
    )

    vertices, faces = floor_plan_vertices, floor_plan_faces
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max() + 0.2

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    robot_width = int(robot_real_width / scale * image_size / 2)

    # Draw floor plan
    for face in faces:
        face_vertices = vertices[face]
        face_vertices_image = [
            map_to_image_coordinate(v, scale, image_size) for v in face_vertices
        ]
        pts = np.array(face_vertices_image, np.int32)
        pts = pts.reshape(-1, 1, 2)
        cv2.fillPoly(image, [pts], (255, 0, 0))

    floor_plan_mask = (image[:, :, 0] == 255) * 255
    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)

    box_masks, _, box_wall_count, image = calc_bbox_masks(
        bbox, class_labels, image, image_size, scale, robot_width, floor_plan_mask, 0
    )

    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8
    )

    accessable_rate = 0
    for label in range(1, num_labels):
        mask = np.zeros_like(walkable_map)
        mask[labels == label] = 1
        accessable_count = 0
        for box_mask in box_masks:
            if (box_mask * mask).sum() > 0:
                accessable_count += 1
        if len(box_masks) > 0 and (labels != 0).sum() > 0:
            accessable_rate += (
                accessable_count / len(box_masks) * mask.sum() / (labels != 0).sum()
            )

    walkable_rate = cal_walkable_metric(
        floor_plan_vertices, floor_plan_faces, floor_plan_centroid, bbox, robot_real_width
    )

    return scene_idx, walkable_rate, accessable_rate, box_wall_count, len(box_masks)


def _process_single_scene_overlap(args):
    """
    Process overlap detection for a single scene.
    Returns (scene_idx, overlap_flag, obj_cnt, overlap_depths)
    Note: Uses CPU in worker processes to avoid CUDA fork issues
    """
    scene_idx, scene_layout, objects_dataset, classes, cfg = args
    
    device = "cpu"  # Use CPU in worker process to avoid CUDA fork issues
    
    boxes = scene_layout
    bbox_params = np.concatenate(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"],
        ],
        axis=-1,
    )[None, :, :]

    renderables, _, _, renderables_remesh, _ = get_textured_objects(
        bbox_params, objects_dataset, classes, cfg
    )
    obj_cnt = len(renderables)
    overlap_flag = np.zeros(obj_cnt)
    overlap_depths = np.zeros(obj_cnt)

    # Batch collect all meshes first
    mesh_data = []
    for i in range(obj_cnt):
        try:
            points, faces = renderables_remesh[i].to_points_and_faces()
        except:
            mesh_cnt = len(renderables_remesh[i].renderables)
            points = []
            faces = []
            point_cnt = 0
            for s in range(mesh_cnt):
                p, f = renderables_remesh[i].renderables[s].to_points_and_faces()
                points.append(p)
                faces.append(f + point_cnt)
                point_cnt += p.shape[0]
            points = np.concatenate(points, axis=0)
            faces = np.concatenate(faces, axis=0)
        mesh_data.append((points, faces))

    # Vectorized overlap detection
    for i in range(obj_cnt):
        verts = torch.tensor(mesh_data[i][0], device=device).unsqueeze(0)
        faces = torch.tensor(mesh_data[i][1], device=device).long()
        
        for j in range(i + 1, obj_cnt):
            if overlap_flag[i] and overlap_flag[j]:
                continue
            if not bbox_overlap(renderables_remesh[i], renderables[j]):
                continue
            
            pointscuda = torch.tensor(mesh_data[j][0], device=device).unsqueeze(0)
            occupancy = check_sign(verts, faces, pointscuda)
            if occupancy.max() > 0:
                overlap_flag[i] = 1
                overlap_flag[j] = 1

    return scene_idx, overlap_flag, obj_cnt, overlap_depths


def calc_wall_overlap_parallel(
    threed_front_results,
    raw_dataset,
    encoded_dataset,
    cfg,
    robot_real_width=0.3,
    calc_object_area=False,
    classes=None,
    num_workers=None,
):
    """Parallelized wall overlap calculation using ProcessPoolExecutor"""
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    task_args = [
        (scene_idx, scene_layout, encoded_dataset, cfg, robot_real_width)
        for scene_idx, scene_layout in threed_front_results
    ]

    walkable_metric_list = []
    accessable_rate_list = []
    box_wall_count_total = 0
    box_count_total = 0

    print(f"Using {num_workers} workers for wall overlap detection...")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
        for (scene_idx, walkable_rate, accessable_rate, 
             box_wall_count, box_count) in tqdm(
            executor.map(_process_single_scene_wall_overlap, task_args),
            total=len(task_args),
            desc="Processing wall overlaps"
        ):
            walkable_metric_list.append(walkable_rate)
            accessable_rate_list.append(accessable_rate)
            box_wall_count_total += box_wall_count
            box_count_total += box_count

    walkable_average_rate = sum(walkable_metric_list) / len(walkable_metric_list) if walkable_metric_list else 0
    accessable_rate = sum(accessable_rate_list) / len(accessable_rate_list) if accessable_rate_list else 0
    box_wall_rate = box_wall_count_total / box_count_total if box_count_total > 0 else 0

    print(f"walkable_average_rate: {walkable_average_rate:.4f}")
    print(f"accessable_rate: {accessable_rate:.4f}")
    print(f"box_wall_rate: {box_wall_rate:.4f}")

    if calc_object_area:
        return walkable_average_rate, accessable_rate, box_wall_rate, 0
    else:
        return walkable_average_rate, accessable_rate, box_wall_rate


def calc_overlap_parallel(
    threed_front_results, raw_dataset, encoded_dataset, cfg, num_workers=None, visualize_overlap=False
):
    """Parallelized overlap calculation using ProcessPoolExecutor"""
    
    classes = raw_dataset.class_labels[:-1]
    print("classes:", classes)
    print("num classes:", len(list(classes)))
    
    path_to_pickled_3d_future_models = PATH_TO_PICKLED_3D_FUTURE_MODEL.format(cfg.data.room_type)
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        path_to_pickled_3d_future_models
    )
    print(f"Loaded {len(objects_dataset)} 3D-FUTURE models")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    task_args = [
        (scene_idx, scene_layout, objects_dataset, classes, cfg)
        for scene_idx, scene_layout in threed_front_results
    ]

    overlap_cnt_total = 0
    obj_cnt_total = 0
    overlap_scene = 0
    scene_cnt = 0
    overlap_area = 0
    overlap_area_max = 0
    obj_overlap_cnt = 0

    print(f"Using {num_workers} workers for overlap detection...")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
        for scene_idx, overlap_flag, obj_cnt, overlap_depths in tqdm(
            executor.map(_process_single_scene_overlap, task_args),
            total=len(task_args),
            desc="Processing overlaps"
        ):
            overlap_cnt_total += overlap_flag.sum()
            obj_cnt_total += obj_cnt
            overlap_scene += overlap_flag.sum() > 0
            scene_cnt += 1
            overlap_area += overlap_depths.sum()
            overlap_area_max += overlap_depths.max()
            obj_overlap_cnt += sum(overlap_depths > 0)

    overlap_ratio = overlap_cnt_total / obj_cnt_total if obj_cnt_total > 0 else 0
    overlap_scene_rate = overlap_scene / scene_cnt if scene_cnt > 0 else 0

    print(f"overlap object: {overlap_ratio:.4f} (count: {overlap_cnt_total}/{obj_cnt_total})")
    print(f"overlap scene rate: {overlap_scene_rate:.4f}")
    print(f"overlap_area_mean: {overlap_area / obj_cnt_total:.4f}" if obj_cnt_total > 0 else "overlap_area_mean: N/A")
    print(f"overlap_area_max: {overlap_area_max / scene_cnt:.4f}" if scene_cnt > 0 else "overlap_area_max: N/A")
    print(f"overlap_area_mean_only_overlapped: {overlap_area / obj_overlap_cnt:.4f}" if obj_overlap_cnt > 0 else "overlap_area_mean_only_overlapped: N/A")

    return overlap_ratio, overlap_scene_rate


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compute physcene metrics with parallel processing"
    )
    parser.add_argument(
        "result_file", help="Path to a pickled result file (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 2)"
    )

    args = parser.parse_args(argv)

    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    config = threed_front_results.config
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=True,
    )

    print("\n=== COMPUTING WALL OVERLAP METRICS ===")
    walkable_average_rate, accessable_rate, box_wall_rate = calc_wall_overlap_parallel(
        threed_front_results,
        raw_dataset,
        encoded_dataset,
        config,
        robot_real_width=0.3,
        calc_object_area=False,
        classes=None,
        num_workers=args.workers,
    )
    
    print("\n=== COMPUTING OVERLAP METRICS ===")
    overlap_ratio, overlap_scene_rate = calc_overlap_parallel(
        threed_front_results,
        raw_dataset,
        encoded_dataset,
        cfg=config,
        num_workers=args.workers,
        visualize_overlap=False,
    )

    print("\n" + "="*50)
    print("=== FINAL METRICS ===")
    print("="*50)
    print(f"Rwalkable (walkable_average_rate): {walkable_average_rate:.4f}")
    print(f"Rreach (accessable_rate): {accessable_rate:.4f}")
    print(f"Rout (box_wall_rate): {box_wall_rate:.4f}")
    print(f"Colobj (overlap_ratio): {overlap_ratio:.4f}")
    print(f"Colscene (overlap_scene_rate): {overlap_scene_rate:.4f}")
    print("="*50)


if __name__ == "__main__":
    main(None)
