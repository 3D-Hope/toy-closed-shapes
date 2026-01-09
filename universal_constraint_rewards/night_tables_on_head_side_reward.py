import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.patches import Polygon as MplPolygon, Rectangle
from universal_constraint_rewards.commons import idx_to_labels


def compute_nightstand_placement_reward(
    parsed_scenes, **kwargs
):
    """
    Penalizes scenes where:
      - Both nightstands are placed on the same side of the bed.
      - A nightstand is at the foot side of the bed (soft penalty based on distance).

    For each bed-nightstand pair:
    1. Find headboard position of bed (same logic as wall proximity)
    2. Determine which side of bed each nightstand is on (left/right relative to headboard)
    3. Compute how far toward foot side it is; apply distance-based penalty
    4. Penalize if both nightstands on same side or too far from head side

    Args:
        parsed_scenes: Dict with keys:
            - 'positions': (B, N, 3) - object positions (x, y, z)
            - 'orientations': (B, N, 2) - [cos_theta, sin_theta]
            - 'sizes': (B, N, 3) - object half-extents (already halved)
            - 'object_indices': (B, N) - object class indices
            - 'is_empty': (B, N) - mask for empty slots
        floor_polygons: List of length B (not used but kept for signature compatibility)
        idx_to_labels: Dict mapping indices to object class names

    Returns:
        rewards: (B,) - nightstand placement reward per scene (negative penalty)
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]  # (B, N, 3)
    orientations = parsed_scenes["orientations"]  # (B, N, 2)
    sizes = parsed_scenes["sizes"]  # (B, N, 3)
    object_indices = parsed_scenes["object_indices"]  # (B, N)
    is_empty = parsed_scenes["is_empty"]  # (B, N)

    batch_size, num_objects = positions.shape[0], positions.shape[1]
    device = positions.device

    # Identify bed and nightstand indices
    bed_indices = [
        int(idx) for idx, label in idx_to_labels[room_type].items() if "bed" in label.lower()
    ]
    nightstand_indices = [
        int(idx)
        for idx, label in idx_to_labels[room_type].items()
        if "nightstand" in label.lower() or "night_stand" in label.lower()
    ]

    rewards = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        beds, nightstands = [], []

        for n in range(num_objects):
            if is_empty[b, n]:
                continue
            obj_idx = object_indices[b, n].item()
            if obj_idx in bed_indices:
                beds.append(n)
            elif obj_idx in nightstand_indices:
                nightstands.append(n)

        if len(beds) == 0 or len(nightstands) == 0:
            continue

        for bed_n in beds:
            pos_x = positions[b, bed_n, 0]
            pos_z = positions[b, bed_n, 2]
            size_x = sizes[b, bed_n, 0]
            size_z = sizes[b, bed_n, 2]
            cos_theta = orientations[b, bed_n, 0]
            sin_theta = orientations[b, bed_n, 1]

            angle_rad = torch.atan2(sin_theta, cos_theta)
            angle_deg = (angle_rad * 180 / torch.pi).item()
            rounded_angle = round(angle_deg / 90) * 90
            normalized_angle = ((rounded_angle % 360) + 360) % 360

            if normalized_angle == 0:
                perp_x, perp_z = 1.0, 0.0
                foot_dir_x, foot_dir_z = 0.0, 1.0
            elif normalized_angle == 90:
                perp_x, perp_z = 0.0, 1.0
                foot_dir_x, foot_dir_z = 1.0, 0.0
            elif normalized_angle == 180:
                perp_x, perp_z = 1.0, 0.0
                foot_dir_x, foot_dir_z = 0.0, -1.0
            else:  # 270
                perp_x, perp_z = 0.0, 1.0
                foot_dir_x, foot_dir_z = -1.0, 0.0

            nightstand_sides = []

            for ns_n in nightstands:
                ns_x = positions[b, ns_n, 0]
                ns_z = positions[b, ns_n, 2]

                dx = ns_x - pos_x
                dz = ns_z - pos_z

                side_projection = dx.item() * perp_x + dz.item() * perp_z
                foot_projection = dx.item() * foot_dir_x + dz.item() * foot_dir_z

                bed_length = size_z if normalized_angle in [0, 180] else size_x

                # --- Soft distance-based penalty ---
                # Normalize distance: negative = head side, positive = foot side
                # Ideal region: slightly negative (near headboard)
                # Penalty grows smoothly as you move toward foot
                dist_ratio = foot_projection / (bed_length + 1e-6)
                if dist_ratio > 0:
                    # Smooth penalty increases with distance toward foot
                    penalty = 10.0 * torch.tanh(
                        torch.tensor(dist_ratio * 2.0, device=device)
                    )
                    rewards[b] -= penalty.item()
                    nightstand_sides.append("foot")
                else:
                    # Closer to headboard side → small bonus based on proximity
                    # proximity_score = torch.exp(-((dist_ratio + 0.3) ** 2) * 10.0)
                    # rewards[b] += proximity_score.item()
                    if side_projection > 0:
                        nightstand_sides.append("right")
                    else:
                        nightstand_sides.append("left")

            if len(nightstand_sides) >= 2:
                non_foot_sides = [s for s in nightstand_sides if s != "foot"]
                if len(non_foot_sides) >= 2:
                    if all(s == "left" for s in non_foot_sides) or all(
                        s == "right" for s in non_foot_sides
                    ):
                        rewards[b] -= 5.0

    return rewards


def visualize_nightstand_placement(
    idx,
    positions,
    sizes,
    orientations,
    object_indices,
    is_empty,
    beds,
    nightstands,
    bed_indices,
    nightstand_indices,
    reward,
    scene_name="",
):
    """Visualize bed and nightstand placement"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw beds
    for bed_n in beds:
        x = positions[idx, bed_n, 0].item()
        z = positions[idx, bed_n, 2].item()
        width = sizes[idx, bed_n, 0].item() * 2
        depth = sizes[idx, bed_n, 2].item() * 2

        angle_rad = torch.atan2(
            orientations[idx, bed_n, 1], orientations[idx, bed_n, 0]
        ).item()
        angle_deg = np.degrees(angle_rad)

        bed_rect = Rectangle(
            (x - width / 2, z - depth / 2),
            width,
            depth,
            angle=angle_deg,
            facecolor="purple",
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
            label="Bed",
        )
        ax.add_patch(bed_rect)

        # Mark headboard
        rounded_angle = round(angle_deg / 90) * 90
        normalized_angle = ((rounded_angle % 360) + 360) % 360

        if normalized_angle == 0:
            hb_x, hb_z = x, z - sizes[idx, bed_n, 2].item()
        elif normalized_angle == 90:
            hb_x, hb_z = x - sizes[idx, bed_n, 0].item(), z
        elif normalized_angle == 180:
            hb_x, hb_z = x, z + sizes[idx, bed_n, 2].item()
        else:  # 270
            hb_x, hb_z = x + sizes[idx, bed_n, 0].item(), z

        ax.plot(hb_x, hb_z, "go", markersize=15, label="Headboard", zorder=5)
        ax.text(x, z, "BED", ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw nightstands
    for ns_n in nightstands:
        x = positions[idx, ns_n, 0].item()
        z = positions[idx, ns_n, 2].item()
        width = sizes[idx, ns_n, 0].item() * 2
        depth = sizes[idx, ns_n, 2].item() * 2

        angle_rad = torch.atan2(
            orientations[idx, ns_n, 1], orientations[idx, ns_n, 0]
        ).item()
        angle_deg = np.degrees(angle_rad)

        ns_rect = Rectangle(
            (x - width / 2, z - depth / 2),
            width,
            depth,
            angle=angle_deg,
            facecolor="orange",
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
            label="Nightstand",
        )
        ax.add_patch(ns_rect)
        ax.plot(x, z, "ro", markersize=8, zorder=5)
        ax.text(x, z, "NS", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    title = (
        f"Scene {idx}"
        + (f" - {scene_name}" if scene_name else "")
        + f" - Reward: {reward:.2f}"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    filename = (
        f"nightstand_placement_scene_{idx}"
        + (f'_{scene_name.replace(" ", "_").replace(":", "")}' if scene_name else "")
        + ".png"
    )
    plt.show()
    # plt.savefig(filename, dpi=150, bbox_inches='tight')
    # plt.close(fig)
    # print(f"  Saved visualization: {filename}")


# Test scenarios
def create_test_scenarios():
    """Create test scenarios with different nightstand placements"""
    device = "cpu"

    # Scenario parameters: (bed_angle, ns1_offset, ns2_offset, description)
    # Offsets are (dx, dz) relative to bed center
    scenarios = [
        # Good placements
        (0, (-2, -1.5), (2, -1.5), "Good: Both nightstands at head, different sides"),
        (90, (-1.5, -2), (-1.5, 2), "Good: Bed rotated 90°, correct placement"),
        (0, (-2, -1.5), None, "Good: Single nightstand at head"),
        # Bad placements - same side
        (0, (-2, -1.5), (-2.5, -1.5), "Bad: Both nightstands on same side (left)"),
        (0, (2, -1.5), (2.5, -1.5), "Bad: Both nightstands on same side (right)"),
        # Bad placements - at foot
        (0, (0, 2), None, "Bad: Nightstand at foot"),
        (0, (-2, -1.5), (2, 2), "Bad: One at head, one at foot"),
        (0, (-1, 2), (1, 2), "Bad: Both at foot, different sides"),
    ]

    results = []

    # Mock idx_to_labels
    idx_to_labels = {0: "bed", 1: "nightstand"}

    for scene_idx, (bed_angle, ns1_offset, ns2_offset, description) in enumerate(
        scenarios
    ):
        # Create scene - always use 3 objects for consistency
        num_objects = 3
        positions = torch.zeros((1, num_objects, 3), device=device)
        orientations = torch.zeros((1, num_objects, 2), device=device)
        sizes = torch.zeros((1, num_objects, 3), device=device)
        object_indices = torch.zeros((1, num_objects), dtype=torch.long, device=device)
        is_empty = torch.zeros((1, num_objects), dtype=torch.bool, device=device)

        # Bed at origin
        positions[0, 0] = torch.tensor([5.0, 0.0, 5.0])
        sizes[0, 0] = torch.tensor([1.5, 0.5, 2.0])  # Half-extents: wider in Z
        angle_rad = np.radians(bed_angle)
        orientations[0, 0] = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)])
        object_indices[0, 0] = 0

        # Nightstand 1
        positions[0, 1] = torch.tensor([5.0 + ns1_offset[0], 0.0, 5.0 + ns1_offset[1]])
        sizes[0, 1] = torch.tensor([0.4, 0.4, 0.4])
        orientations[0, 1] = torch.tensor([1.0, 0.0])
        object_indices[0, 1] = 1

        # Nightstand 2 (if exists)
        if ns2_offset:
            positions[0, 2] = torch.tensor(
                [5.0 + ns2_offset[0], 0.0, 5.0 + ns2_offset[1]]
            )
            sizes[0, 2] = torch.tensor([0.4, 0.4, 0.4])
            orientations[0, 2] = torch.tensor([1.0, 0.0])
            object_indices[0, 2] = 1
        else:
            is_empty[0, 2] = True

        parsed_scenes = {
            "positions": positions,
            "orientations": orientations,
            "sizes": sizes,
            "object_indices": object_indices,
            "is_empty": is_empty,
        }

        # Compute reward
        reward = compute_nightstand_placement_reward(
            parsed_scenes, [None], idx_to_labels, viz_batch_idx=None
        )

        # Manually call visualization for each scene
        beds = [0]  # Bed is always at index 0
        nightstands = [1] if not ns2_offset else [1, 2]
        visualize_nightstand_placement(
            0,
            positions,
            sizes,
            orientations,
            object_indices,
            is_empty,
            beds,
            nightstands,
            [0],
            [1],
            reward[0].item(),
        )

        results.append((description, reward[0].item()))
        print(f"Scene {scene_idx}: {description}")
        print(f"  Reward: {reward[0].item():.2f}\n")

    return results


if __name__ == "__main__":
    # print("Testing Nightstand Placement Reward Function\n")
    # print("=" * 60)
    # results = create_test_scenarios()
    # print("=" * 60)
    # print("\nSummary:")
    # for desc, reward in results:
    #     status = "✓" if reward == 0 else "✗"
    #     print(f"{status} {desc}: {reward:.2f}")
    args = np.load(
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/reward_func_args_for_first_10_scenes.npy",
        allow_pickle=True,
    )
    # print("loaded ", args)
    # only take start to end scenes
    start = 40
    end = 55

    print(compute_nightstand_placement_reward(**args.item()))
