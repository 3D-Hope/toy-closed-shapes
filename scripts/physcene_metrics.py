import argparse

# from threed_front.evaluation.utils import count_out_of_boundary, compute_bbox_iou
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
import torch

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

"""Script for calculating physcene metrics. """


def calc_wall_overlap(
    threed_front_results,
    raw_dataset,
    encoded_dataset,
    cfg,
    robot_real_width=0.3,
    calc_object_area=False,
    classes=None,
):
    box_wall_count = 0
    accessable_count = 0
    box_count = 0
    walkable_metric_list = []
    accessable_rate_list = []

    for scene_idx, scene_layout in tqdm(threed_front_results):
        raw_item = encoded_dataset[scene_idx]
        # print(f"Scene {scene_idx} - Raw Item: {raw_item.keys()}")

        floor_plan_vertices = raw_item["floor_plan_vertices"]
        floor_plan_faces = raw_item["floor_plan_faces"]
        floor_plan_centroid = raw_item["floor_plan_centroid"]

        # print(scene_layout["class_labels"].shape, scene_layout["translations"].shape, scene_layout["sizes"].shape, scene_layout["angles"].shape)
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
        # vertices = vertices[:, :2]
        scale = np.abs(vertices).max() + 0.2

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_real_width / scale * image_size / 2)

        # draw face
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [
                map_to_image_coordinate(v, scale, image_size) for v in face_vertices
            ]

            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)

        floor_plan_mask = (image[:, :, 0] == 255) * 255
        # cv2.imwrite(os.path.join(save_path, "debug_floor.png"), floor_plan_mask)
        # 缩小墙边界，机器人行动范围
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)

        box_masks, handle_points, box_wall_count, image = calc_bbox_masks(
            bbox,
            class_labels,
            image,
            image_size,
            scale,
            robot_width,
            floor_plan_mask,
            box_wall_count,
        )
        # cv2.imwrite(os.path.join(save_path, "debug.png"), image)
        # breakpoint()
        walkable_map = image[:, :, 0].copy()
        # cv2.imwrite(os.path.join(save_path, "debug2.png"), walkable_map)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8
        )
        # 遍历每个连通域

        accessable_rate = 0
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_count = 0
            for box_mask in box_masks:
                if (box_mask * mask).sum() > 0:
                    accessable_count += 1
            accessable_rate += (
                accessable_count / len(box_masks) * mask.sum() / (labels != 0).sum()
            )
        accessable_rate_list.append(accessable_rate)
        box_count += len(box_masks)

        # walkable map area rate
        if calc_object_area:
            walkable_rate, object_area_ratio = cal_walkable_metric(
                floor_plan_vertices,
                floor_plan_faces,
                floor_plan_centroid,
                bbox,
                robot_width=0.3,
                calc_object_area=True,
            )
        else:
            walkable_rate = cal_walkable_metric(
                floor_plan_vertices,
                floor_plan_faces,
                floor_plan_centroid,
                bbox,
                robot_width=0.3,
            )
        walkable_metric_list.append(walkable_rate)

    walkable_average_rate = sum(walkable_metric_list) / len(walkable_metric_list)
    accessable_rate = sum(accessable_rate_list) / len(accessable_rate_list)
    # accessable_handle_rate = sum(accessable_handle_rate_list)/len(accessable_handle_rate_list)
    box_wall_rate = box_wall_count / box_count

    print("walkable_average_rate:", walkable_average_rate)
    print("accessable_rate:", accessable_rate)
    # print('accessable_handle_rate:', accessable_handle_rate)
    print("box_wall_rate:", box_wall_rate)
    if calc_object_area:
        print("object_area_ratio:", object_area_ratio)
        return walkable_average_rate, accessable_rate, box_wall_rate, object_area_ratio
    else:
        return walkable_average_rate, accessable_rate, box_wall_rate


def calc_overlap(
    threed_front_results, raw_dataset, encoded_dataset, cfg, visualize_overlap=False
):
    # Get classes here
    classes = raw_dataset.class_labels[:-1]  # Removing the empty class (end)
    print("classes:", classes)
    print("classes shape:", len(list(classes)))
    device = "cuda"
    # class_num = classes.shape[0]-1
    a = 0
    overlap_cnt_total = 0
    obj_cnt_total = 0
    overlap_scene = 0
    scene_cnt = 0

    overlap_area = 0
    overlap_area_max = 0
    obj_overlap_cnt = 0

    path_to_pickled_3d_future_models = PATH_TO_PICKLED_3D_FUTURE_MODEL.format(cfg.data.room_type)
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        path_to_pickled_3d_future_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    cnt = min(100, len(threed_front_results))
    for scene_idx, scene_layout in tqdm(threed_front_results):
        # raw_item = encoded_dataset[scene_idx]
        a += 1
        print("visualizing scene ", a, "/", cnt)
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

            if visualize_overlap:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(points)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(points)
                pcd0.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

            verts = torch.tensor(points, device=device).unsqueeze(0)
            faces = torch.tensor(faces, device=device).long()
            for j in range(i + 1, obj_cnt):
                if overlap_flag[i] and overlap_flag[j]:
                    continue
                # obj2
                if not bbox_overlap(renderables_remesh[i], renderables[j]):
                    continue
                try:
                    points = renderables[j].to_points_and_faces()[0][None, :, :]
                except:
                    mesh_cnt = len(renderables[j].renderables)
                    points = [
                        renderables[j].renderables[s].to_points_and_faces()[0]
                        for s in range(mesh_cnt)
                    ]
                    points = np.concatenate(points, axis=0)[None, :, :]
                pointscuda = torch.tensor(points, device=device)
                occupancy = check_sign(verts, faces, pointscuda)

                if occupancy.max() > 0:
                    overlap_flag[i] = 1
                    overlap_flag[j] = 1

        print(overlap_flag)
        print(overlap_depths)
        overlap_cnt_total += overlap_flag.sum()
        obj_cnt_total += obj_cnt

        overlap_scene += overlap_flag.sum() > 0
        scene_cnt += 1

        overlap_area += overlap_depths.sum()
        overlap_area_max += overlap_depths.max()

        obj_overlap_cnt += sum(overlap_depths > 0)

        # if visualize_overlap:
        #     show_renderables(renderables)
    overlap_ratio = overlap_cnt_total / obj_cnt_total
    print(
        "overlap object: ", overlap_ratio, "cnt ", overlap_cnt_total, "/", obj_cnt_total
    )
    overlap_scene_rate = overlap_scene / scene_cnt
    print("overlap scene rate: ", overlap_scene_rate)
    print("overlap_area_mean: ", overlap_area / obj_cnt_total)
    print("overlap_area_max : ", overlap_area_max / scene_cnt)
    print("overlap_area_mean_only_overlaped : ", overlap_area / obj_overlap_cnt)

    return overlap_ratio, overlap_scene_rate


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Compute the FID scores between the real and the " "synthetic images"
        )
    )
    parser.add_argument(
        "result_file", help="Path to a pickled result file (ThreedFrontResults object)"
    )

    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    # Load dataset
    config = threed_front_results.config
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=True,
    )

    walkable_average_rate, accessable_rate, box_wall_rate = calc_wall_overlap(
        threed_front_results,
        raw_dataset,
        encoded_dataset,
        config,
        robot_real_width=0.3,
        calc_object_area=False,
        classes=None,
    )
    
    # overlap_ratio, overlap_scene_rate = calc_overlap(
    #     threed_front_results,
    #     raw_dataset,
    #     encoded_dataset,
    #     cfg=config,
    #     visualize_overlap=False,
    # )
    print("walkable_average_rate, Rwalkable:", walkable_average_rate)
    print("accessable_rate, Rreach:", accessable_rate)
    print("box_wall_rate, Rout:", box_wall_rate)
    # print("overlap_ratio, Colobj:", overlap_ratio)
    # print("overlap_scene_rate, Colscene:", overlap_scene_rate)


# Walkable Average Rate(walkable_average_rate): Path exists in the scene for the robot to walk through
# Accessable Rate(accessable_rate): The robot can access every object in the scene
# Box Wall Rate(box_wall_rate): Objects in scene are going out of bounds or not
# Overlap Ratio(overlap_ratio) (Col Object): The ratio of the number of overlapping objects to the total number of objects
# Overlap Scene Rate(overlap_scene_rate) (Col Scene): The ratio of the number of scenes with overlapping objects to the total number of scenes
if __name__ == "__main__":
    main(None)
