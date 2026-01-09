import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.patches import FancyArrow, Polygon as MplPolygon, Rectangle
from universal_constraint_rewards.commons import idx_to_labels

# TODO: IF distance of headboard is less than 5 0 reward

# Objects that should have back against wall
WALL_BACKED_OBJECTS = {
    "double_bed",
    "single_bed",
    "kids_bed",
    "wardrobe",
    "bookshelf",
    "cabinet",
    "children_cabinet",
    "dressing_table",
    "tv_stand",
}


def compute_wall_proximity_reward(
    parsed_scenes, floor_polygons, **kwargs
):
    """
    Reward furniture for having their back/headboard close to walls.

    For each furniture:
    1. Calculate headboard position based on rotation
    2. Cast ray from headboard toward wall
    3. Find nearest wall edge intersection
    4. Measure distance

    Args:
        parsed_scenes: Dict with keys:
            - 'positions': (B, N, 3) - object positions (x, y, z)
            - 'orientations': (B, N, 2) - [cos_theta, sin_theta]
            - 'sizes': (B, N, 3) - object half-extents (already halved)
            - 'object_indices': (B, N) - object class indices
            - 'is_empty': (B, N) - mask for empty slots
        floor_polygons: List of length B, each containing (M, 2) polygon vertices [x, z]
        idx_to_labels: Dict mapping indices to object class names
        viz_batch_idx: (optional, int) If provided, will save a visualization for this batch index.

    Returns:
        rewards: (B,) - wall proximity reward per scene (negative of total distance)
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]  # (B, N, 3)
    orientations = parsed_scenes["orientations"]  # (B, N, 2)
    sizes = parsed_scenes["sizes"]  # (B, N, 3) - already half-extents
    object_indices = parsed_scenes["object_indices"]  # (B, N)
    is_empty = parsed_scenes["is_empty"]  # (B, N)

    batch_size, num_objects = positions.shape[0], positions.shape[1]
    device = positions.device

    # Identify which objects should have back against wall
    wall_backed_indices = [
        int(idx) for idx, label in idx_to_labels[room_type].items() if label in WALL_BACKED_OBJECTS
    ]
    # print("wall_backed_indices:", wall_backed_indices)
    should_back_wall = torch.zeros_like(is_empty, dtype=torch.bool)
    for idx in wall_backed_indices:
        should_back_wall |= object_indices == idx

    # Mask for valid objects (non-empty AND should be against wall)
    valid_mask = ~is_empty & should_back_wall  # (B, N)
    # print("valid_mask:", valid_mask)
    # Compute distances for each scene
    total_distances = torch.zeros(batch_size, device=device)

    # def visualize_bed_against_wall(idx, positions, sizes, orientations, polygon, valid_mask, min_distances, headboard_xs, headboard_zs, nearest_wall_points):
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle, Polygon as MplPolygon
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     # Draw floor polygon
    #     poly_np = polygon.cpu().numpy() if hasattr(polygon, 'cpu') else np.array(polygon)
    #     room_patch = MplPolygon(poly_np, fill=True, facecolor='lightgray', edgecolor='black', linewidth=2)
    #     ax.add_patch(room_patch)
    #     # Flip z axis for bed objects (but not for floor polygon)
    #     z_min, z_max = poly_np[:,1].min(), poly_np[:,1].max()
    #     for n in range(positions.shape[1]):
    #         if not valid_mask[idx, n]:
    #             continue
    #         x = positions[idx, n, 0].item()
    #         z = positions[idx, n, 2].item()
    #         flipped_z = z_max - (z - z_min)
    #         width = sizes[idx, n, 0].item() * 2
    #         depth = sizes[idx, n, 2].item() * 2
    #         angle_rad = torch.atan2(orientations[idx, n, 1], orientations[idx, n, 0]).item()
    #         angle_deg = np.degrees(angle_rad)
    #         # Rectangle position: flip z for y coordinate
    #         bed_rect = Rectangle((x - width/2, flipped_z - depth/2), width, depth, angle=angle_deg, facecolor='purple', alpha=0.7, edgecolor='black', linewidth=2)
    #         ax.add_patch(bed_rect)
    #         # Mark headboard
    #         flipped_headboard_z = z_max - (headboard_zs[n] - z_min) if headboard_zs[n] is not None else None
    #         ax.plot(headboard_xs[n], flipped_headboard_z, 'go', markersize=10, label='Headboard' if n==0 else None, zorder=5)
    #         # Draw line to wall
    #         if nearest_wall_points[n] is not None:
    #             flipped_wall_z = z_max - (nearest_wall_points[n][1] - z_min)
    #             ax.plot([headboard_xs[n], nearest_wall_points[n][0]], [flipped_headboard_z, flipped_wall_z], 'b--', linewidth=2, label='Distance to wall' if n==0 else None)
    #             ax.plot(nearest_wall_points[n][0], flipped_wall_z, 'rx', markersize=12, markeredgewidth=3, zorder=5)
    #         ax.text(x, flipped_z, f"d={min_distances[n]:.2f}", color='blue', fontsize=10)
    #     ax.set_xlim(poly_np[:,0].min()-1, poly_np[:,0].max()+1)
    #     ax.set_ylim(poly_np[:,1].min()-1, poly_np[:,1].max()+1)
    #     ax.invert_yaxis()  # Flip z axis so higher z is at the top
    #     ax.set_aspect('equal')
    #     ax.set_title(f"Beds against wall (Scene {idx})")
    #     ax.legend()
    #     plt.savefig(f'bed_against_wall_scene_{idx}.png', dpi=150, bbox_inches='tight')
    #     plt.close(fig)

    # # Get batch index to visualize, if provided
    # viz_batch_idx = kwargs.get('viz_batch_idx', 29)

    for b in range(batch_size):
        if floor_polygons[b] is None or len(floor_polygons[b]) == 0:
            continue
        polygon = floor_polygons[b]  # (M, 2) - [x, z] coordinates
        if not isinstance(polygon, torch.Tensor):
            polygon = torch.tensor(polygon, dtype=torch.float32, device=device)
        else:
            polygon = polygon.to(device)

        num_edges = polygon.shape[0]

        # For visualization
        min_distances = []
        headboard_xs = []
        headboard_zs = []
        nearest_wall_points = []

        for n in range(num_objects):
            if not valid_mask[b, n]:
                min_distances.append(None)
                headboard_xs.append(None)
                headboard_zs.append(None)
                nearest_wall_points.append(None)
                continue
            pos_x = positions[b, n, 0]  # X position
            pos_z = positions[b, n, 2]  # Z position
            size_x = sizes[b, n, 0]  # half-width
            size_z = sizes[b, n, 2]  # half-depth

            cos_theta = orientations[b, n, 0]
            sin_theta = orientations[b, n, 1]

            # Convert to angle in degrees
            angle_rad = torch.atan2(sin_theta, cos_theta)
            angle_deg = (angle_rad * 180 / torch.pi).item()
            # Round to nearest 90 degrees
            rounded_angle = round(angle_deg / 90) * 90
            normalized_angle = ((rounded_angle % 360) + 360) % 360

            # Calculate headboard position and ray direction
            if normalized_angle == 0:
                headboard_x = pos_x
                headboard_z = pos_z - size_z
                ray_dx = 0.0
                ray_dz = -1.0
            elif normalized_angle == 90:
                headboard_x = pos_x - size_x
                headboard_z = pos_z
                ray_dx = -1.0
                ray_dz = 0.0
            elif normalized_angle == 180:
                headboard_x = pos_x
                headboard_z = pos_z + size_z
                ray_dx = 0.0
                ray_dz = 1.0
            else:  # 270
                headboard_x = pos_x + size_x
                headboard_z = pos_z
                ray_dx = 1.0
                ray_dz = 0.0

            # Cast ray and find nearest wall intersection
            min_distance = float("inf")
            nearest_wall_point = None
            for i in range(num_edges):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % num_edges]
                edge_x = p2[0] - p1[0]
                edge_z = p2[1] - p1[1]
                denom = ray_dx * edge_z - ray_dz * edge_x
                if abs(denom) < 1e-8:
                    continue
                diff_x = p1[0].item() - headboard_x.item()
                diff_z = p1[1].item() - headboard_z.item()
                t = (diff_x * edge_z.item() - diff_z * edge_x.item()) / denom
                s = (diff_x * ray_dz - diff_z * ray_dx) / denom
                if t > 0 and 0 <= s <= 1:
                    if t < min_distance:
                        min_distance = t
                        nearest_wall_point = (
                            headboard_x + t * ray_dx,
                            headboard_z + t * ray_dz,
                        )
            if min_distance < float("inf"):
                total_distances[b] += min_distance
                min_distances.append(min_distance)
                headboard_xs.append(headboard_x.item())
                headboard_zs.append(headboard_z.item())
                nearest_wall_points.append(nearest_wall_point)
            else:
                min_distances.append(None)
                headboard_xs.append(headboard_x.item())
                headboard_zs.append(headboard_z.item())
                nearest_wall_points.append(None)
        # Visualization for requested batch index
        # if viz_batch_idx is not None and b == viz_batch_idx:
        #     # print("polygon of idx:", b, polygon)
        #     visualize_bed_against_wall(b, positions, sizes, orientations, polygon, valid_mask, min_distances, headboard_xs, headboard_zs, nearest_wall_points)

    # Reward is negative of total distance
    reward = -total_distances
    return reward


def visualize_bed_placement():
    """
    Visualize bed at different rotations, both close and far from walls.
    Uses ray-casting to find actual nearest wall.
    Also computes wall proximity reward for each setup using compute_wall_proximity_reward().
    """
    # L-shaped room
    room_polygon = np.array(
        [
            [0, 0],
            [6, 0],
            [6, 4],
            [3, 4],
            [3, 6],
            [0, 6],
        ]
    )

    # Bed dimensions (half-extents)
    bed_width = 1.0
    bed_depth = 0.9

    # Test configurations: (rotation, x, z, description)
    configs = [
        (0, 3.0, 3.5, "0° near internal wall"),
        (0, 3.0, 1.0, "0° near bottom edge"),
        (90, 1.0, 2.0, "90° close to left"),
        (90, 2.0, 2.0, "90° near internal wall"),
        (180, 4.0, 2.0, "180° close to top"),
        (180, 4, 3.0, "180° far from top"),
        (270, 5.0, 2.0, "270° close to right"),
        (270, 2.0, 3.0, "270° near internal wall"),
    ]

    # Prepare dummy label mapping
    idx_to_labels = {0: "double_bed"}  # use the bed as wall-backed object

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (rotation, x, z, desc) in enumerate(configs):
        ax = axes[idx]

        # Draw room
        room_patch = MplPolygon(
            room_polygon,
            fill=True,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(room_patch)

        # Calculate headboard position and ray direction (for drawing)
        angle_rad = np.radians(rotation)
        if rotation == 0:
            headboard_x, headboard_z = x, z - bed_depth
            ray_dx, ray_dz = 0.0, -1.0
        elif rotation == 90:
            headboard_x, headboard_z = x - bed_width, z
            ray_dx, ray_dz = -1.0, 0.0
        elif rotation == 180:
            headboard_x, headboard_z = x, z + bed_depth
            ray_dx, ray_dz = 0.0, 1.0
        else:
            headboard_x, headboard_z = x + bed_width, z
            ray_dx, ray_dz = 1.0, 0.0

        # -----------------------------
        # Call reward function here
        # -----------------------------
        parsed_scenes = {
            "positions": torch.tensor([[[x, 0.0, z]]], dtype=torch.float32),
            "orientations": torch.tensor(
                [[[np.cos(angle_rad), np.sin(angle_rad)]]], dtype=torch.float32
            ),
            "sizes": torch.tensor([[[bed_width, 0.0, bed_depth]]], dtype=torch.float32),
            "object_indices": torch.tensor([[0]]),  # single bed
            "is_empty": torch.tensor([[False]]),
        }
        floor_polygons = [torch.tensor(room_polygon, dtype=torch.float32)]
        reward = compute_wall_proximity_reward(
            parsed_scenes, floor_polygons, idx_to_labels
        )
        reward_value = reward.item()
        # -----------------------------

        # Ray-cast manually to visualize intersection
        min_distance = float("inf")
        nearest_wall_point, nearest_edge = None, None
        num_edges = len(room_polygon)
        for i in range(num_edges):
            p1 = room_polygon[i]
            p2 = room_polygon[(i + 1) % num_edges]
            edge_x, edge_z = p2[0] - p1[0], p2[1] - p1[1]
            denom = ray_dx * edge_z - ray_dz * edge_x
            if abs(denom) < 1e-8:
                continue
            diff_x, diff_z = p1[0] - headboard_x, p1[1] - headboard_z
            t = (diff_x * edge_z - diff_z * edge_x) / denom
            s = (diff_x * ray_dz - diff_z * ray_dx) / denom
            if t > 0 and 0 <= s <= 1:
                if t < min_distance:
                    min_distance = t
                    nearest_wall_point = (
                        headboard_x + t * ray_dx,
                        headboard_z + t * ray_dz,
                    )
                    nearest_edge = (p1, p2)

        # Draw bed rectangle
        bed_rect = Rectangle(
            (x - bed_width, z - bed_depth),
            bed_width * 2,
            bed_depth * 2,
            angle=rotation,
            rotation_point="center",
            facecolor="purple",
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(bed_rect)

        # Draw arrows
        ax.arrow(
            x,
            z,
            np.sin(angle_rad) * 0.5,
            np.cos(angle_rad) * 0.5,
            head_width=0.2,
            head_length=0.15,
            fc="red",
            ec="red",
            linewidth=2,
        )
        ax.arrow(
            x,
            z,
            -np.sin(angle_rad) * 0.5,
            -np.cos(angle_rad) * 0.5,
            head_width=0.2,
            head_length=0.15,
            fc="green",
            ec="green",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

        # Mark headboard
        ax.plot(
            headboard_x, headboard_z, "go", markersize=10, label="Headboard", zorder=5
        )

        # Highlight wall edge and ray
        if nearest_edge is not None:
            ax.plot(
                [nearest_edge[0][0], nearest_edge[1][0]],
                [nearest_edge[0][1], nearest_edge[1][1]],
                "r-",
                linewidth=4,
                alpha=0.5,
                label="Nearest wall",
            )
        if nearest_wall_point is not None:
            ax.plot(
                [headboard_x, nearest_wall_point[0]],
                [headboard_z, nearest_wall_point[1]],
                "b--",
                linewidth=2,
                label=f"Dist={min_distance:.2f}",
            )
            ax.plot(
                nearest_wall_point[0],
                nearest_wall_point[1],
                "rx",
                markersize=12,
                markeredgewidth=3,
                zorder=5,
            )

        # Plot settings
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 6.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        color = (
            "green"
            if reward_value < -0.3
            else ("orange" if reward_value < -1.0 else "red")
        )
        ax.set_title(
            f"{desc}\nReward: {reward_value:.3f}",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        "bed_placement_visualization_with_rewards.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Visualization saved as 'bed_placement_visualization_with_rewards.png'")


if __name__ == "__main__":
    # visualize_bed_placement()
    args = np.load(
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/reward_func_args_for_first_10_scenes.npy",
        allow_pickle=True,
    )
    # print("loaded ", args)
    # only take start to end scenes
    start = 40
    end = 55

    print(compute_wall_proximity_reward(**args.item()))
