import os
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tqdm import tqdm


def compute_boundary_violation_reward(
    parsed_scene: Dict[str, torch.Tensor],
    floor_polygons: torch.Tensor,  # (B, num_vertices, 2) - only used if cache miss
    is_val: bool,
    indices: List[int],  # (B,) - indices to lookup cached SDFs
    sdf_cache,
    grid_resolution: float = 0.05,
    # sdf_cache_dir: str = "./sdf_cache",
    **kwargs,  #
) -> torch.Tensor:
    """
    Compute boundary violation reward using cached SDF grids.

    **IMPORTANT**: Call `precompute_sdf_cache()` once before training to generate cache!

    Args:
        parsed_scene: Dictionary with positions, sizes, is_empty, device
        floor_polygons: (B, num_vertices, 2) - only needed if cache doesn't exist
        indices: (B,) - scene indices for SDF lookup
        grid_resolution: SDF grid resolution
        sdf_cache_dir: Directory containing cached SDF grids

    Returns:
        rewards: (B, 1) - sum of negative violation distances per scene
    """
    positions = parsed_scene["positions"]  # (B, N, 3)
    sizes = parsed_scene["sizes"]  # (B, N, 3) - half-extents
    is_empty = parsed_scene["is_empty"]  # (B, N)
    device = parsed_scene["device"]

    B, N = positions.shape[0], positions.shape[1]
    rewards = torch.zeros(B, device=device)

    # Process each scene in batch
    for b in range(B):
        scene_idx = indices[b] if indices is not None else b

        # Try to load cached SDF (from memory)
        sdf_data = sdf_cache.load(scene_idx)
        # print(f"sdf cache {sdf_data} for scene idx {scene_idx}")

        if sdf_data is None:
            # Cache miss - compute on the fly (slow path)
            if floor_polygons is None:
                raise ValueError(
                    f"SDF cache miss for scene {scene_idx} but no floor_polygons provided!"
                )

            print(
                f"Warning: SDF cache miss for scene {scene_idx}. Computing on the fly (slow)..."
            )
            floor_verts = floor_polygons[b]
            sdf_checker = SDFBoundaryChecker(
                floor_vertices=floor_verts.tolist(), grid_resolution=grid_resolution
            )

            # Save for next time
            sdf_cache.save(scene_idx, sdf_checker.get_cache_data())
        else:
            # Cache hit - fast path
            sdf_checker = SDFBoundaryChecker.from_cache_data(sdf_data)

        # Check each object in the scene
        for n in range(N):
            if is_empty[b, n]:
                continue

            obj_pos = positions[b, n].cpu().detach().numpy()
            obj_size = sizes[b, n].cpu().detach().numpy()

            # Check 4 corners of object footprint
            x_center, z_center = obj_pos[0], obj_pos[2]
            dx, dz = obj_size[0], obj_size[2]

            corners = [
                (x_center - dx, z_center - dz),
                (x_center + dx, z_center - dz),
                (x_center + dx, z_center + dz),
                (x_center - dx, z_center + dz),
            ]

            max_violation = 0.0
            for corner in corners:
                violation = sdf_checker.check_violation(corner)
                max_violation = max(max_violation, violation)
            max_violation = min(max_violation, 5.0)  # Cap at 5 meters
            rewards[b] -= max_violation 
    rewards = torch.where(
        rewards == 0,
        torch.ones_like(rewards),
        rewards
    )
    return rewards


def _compute_single_sdf(args):
    """Worker function for parallel SDF computation."""
    idx, floor_verts, grid_resolution, split_name = args
    try:
        checker = SDFBoundaryChecker(
            floor_vertices=floor_verts.tolist(), grid_resolution=grid_resolution
        )
        return idx, checker.get_cache_data(), None
    except Exception as e:
        return idx, None, str(e)


def precompute_sdf_cache(
    config,
    sdf_cache_dir: str = "./sdf_cache",
    grid_resolution: float = 0.05,
    verbose: bool = True,
    validate: bool = False,
    num_workers: int = None,  # None = use all available CPUs
):
    """
    Precompute and save SDF grids for all floor polygons in your dataset.
    Handles both train/val (combined) and test splits separately.
    Uses multiprocessing for fast parallel computation.

    **Call this ONCE before training** to generate the cache!

    Args:
        config: Config object with dataset parameters
        sdf_cache_dir: Directory to save cached SDF grids
        grid_resolution: SDF grid resolution (must match training)
        verbose: Print progress
        validate: Run sanity checks on each polygon
        num_workers: Number of parallel workers (None = all CPUs)

    Example:
        # Before training
        precompute_sdf_cache(config, sdf_cache_dir="./sdf_cache", num_workers=32)
    """
    from steerable_scene_generation.datasets.custom_scene import CustomDataset

    if num_workers is None:
        num_workers = os.cpu_count()

    # Precompute for train/val split (combined as one cache)
    train_val_dir = os.path.join(sdf_cache_dir, "train_val")
    os.makedirs(train_val_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print(
            f"Precomputing SDF cache for TRAIN/VAL split (using {num_workers} workers)..."
        )
        print("=" * 60)

    train_val_dataset = CustomDataset(
        cfg=config.dataset,
        split=["train", "val"],
        ckpt_path=None,
    )

    _precompute_split(
        dataset=train_val_dataset,
        sdf_cache_dir=train_val_dir,
        grid_resolution=grid_resolution,
        split_name="train_val",
        verbose=verbose,
        validate=validate,
        num_workers=num_workers,
    )

    # Precompute for test split
    test_dir = os.path.join(sdf_cache_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Precomputing SDF cache for TEST split (using {num_workers} workers)...")
        print("=" * 60)

    test_dataset = CustomDataset(
        cfg=config.dataset,
        split=["test"],
        ckpt_path=None,
    )

    _precompute_split(
        dataset=test_dataset,
        sdf_cache_dir=test_dir,
        grid_resolution=grid_resolution,
        split_name="test",
        verbose=verbose,
        validate=validate,
        num_workers=num_workers,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("✓ SDF cache precomputation complete!")
        print(f"  Train/Val: {train_val_dir}")
        print(f"  Test: {test_dir}")
        print("=" * 60)


def _precompute_split(
    dataset,
    sdf_cache_dir: str,
    grid_resolution: float,
    split_name: str,
    verbose: bool,
    validate: bool,
    num_workers: int,
):
    """
    Internal helper to precompute SDF cache for a single dataset split using parallel processing.

    Args:
        dataset: Dataset instance
        sdf_cache_dir: Directory to save cache
        grid_resolution: SDF grid resolution
        split_name: Name of split for logging
        verbose: Print progress
        validate: Run sanity checks
        num_workers: Number of parallel workers
    """
    floor_polygons_list = [
        dataset.get_floor_polygon_points(idx) for idx in range(len(dataset))
    ]

    # Save metadata
    metadata = {
        "num_scenes": len(floor_polygons_list),
        "grid_resolution": grid_resolution,
        "split": split_name,
    }
    with open(os.path.join(sdf_cache_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    failed_scenes = []

    # Prepare arguments for parallel processing
    compute_args = [
        (idx, floor_verts, grid_resolution, split_name)
        for idx, floor_verts in enumerate(floor_polygons_list)
    ]

    # Parallel computation with progress bar
    if verbose:
        print(
            f"[{split_name}] Computing {len(floor_polygons_list)} SDFs with {num_workers} workers..."
        )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_compute_single_sdf, args): args[0] for args in compute_args
        }

        if verbose:
            progress_bar = tqdm(
                total=len(futures), desc=f"[{split_name}] Precomputing SDFs"
            )

        for future in as_completed(futures):
            idx, cache_data, error = future.result()

            if error is not None:
                print(f"ERROR: [{split_name}] Scene {idx} failed: {error}")
                failed_scenes.append(idx)
            elif cache_data is not None:
                # Validate if requested
                if validate:
                    floor_verts = floor_polygons_list[idx]
                    center = floor_verts.mean(axis=0)
                    # Quick validation using the cached data
                    checker = SDFBoundaryChecker.from_cache_data(cache_data)
                    sdf_val = checker._query_sdf((center[0], center[1]))
                    if sdf_val < 0:
                        print(
                            f"Warning: [{split_name}] Scene {idx} - polygon center is outside! SDF={sdf_val:.3f}"
                        )
                        failed_scenes.append(idx)

                # Save to disk
                cache_path = os.path.join(sdf_cache_dir, f"sdf_{idx}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)

            if verbose:
                progress_bar.update(1)

        if verbose:
            progress_bar.close()

    if verbose:
        print(
            f"✓ [{split_name}] Precomputed {len(floor_polygons_list)} SDF grids in {sdf_cache_dir}"
        )
        if failed_scenes:
            print(
                f"⚠ [{split_name}] {len(failed_scenes)} scenes had issues: {failed_scenes[:10]}..."
            )

    # Save failed scenes list
    if failed_scenes:
        with open(os.path.join(sdf_cache_dir, "failed_scenes.txt"), "w") as f:
            f.write("\n".join(map(str, failed_scenes)))


class SDFCache:
    """Manages loading/saving of cached SDF grids with split awareness. Loads all data into RAM on init."""

    def __init__(
        self, cache_dir: str, grid_resolution: float = 0.05, split: str = "train_val"
    ):
        """
        Args:
            cache_dir: Base cache directory (e.g., "./sdf_cache")
            grid_resolution: Grid resolution for validation
            split: Either "train_val" or "test" to determine which cache to use
        """
        self.base_cache_dir = Path(cache_dir)
        self.grid_resolution = grid_resolution
        self.split = split

        # Determine actual cache directory based on split
        if split in ["train", "val", "train_val"]:
            self.cache_dir = self.base_cache_dir / "train_val"
        elif split == "test":
            self.cache_dir = self.base_cache_dir / "test"
        else:
            raise ValueError(
                f"Unknown split: {split}. Expected 'train', 'val', 'train_val', or 'test'"
            )

        # Verify cache metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            if metadata["grid_resolution"] != grid_resolution:
                print(
                    f"Warning: Cache resolution {metadata['grid_resolution']} != requested {grid_resolution}"
                )
        else:
            print(
                f"Warning: No metadata found at {metadata_path}. Cache may not exist."
            )

        # Load all SDF data into memory
        self._memory_cache = {}
        self._load_all_to_memory()

    def _load_all_to_memory(self):
        """Load all cached SDFs into RAM for fast access."""
        if not self.cache_dir.exists():
            print(
                f"Warning: Cache directory {self.cache_dir} does not exist. No data loaded."
            )
            return

        sdf_files = sorted(self.cache_dir.glob("sdf_*.pkl"))
        print(
            f"Loading {len(sdf_files)} SDF caches into memory from {self.cache_dir}..."
        )

        for sdf_file in tqdm(sdf_files, desc=f"Loading {self.split} SDFs to RAM"):
            # Extract scene index from filename
            scene_idx = int(sdf_file.stem.split("_")[1])

            with open(sdf_file, "rb") as f:
                self._memory_cache[scene_idx] = pickle.load(f)
        # print(f"[Ashok] Loaded sdf files, testing for idx 1 {self._memory_cache.get(1,None)}")

        # Calculate memory usage
        total_bytes = sum(
            data["sdf_grid"].nbytes
            + len(pickle.dumps(data["world_bounds"]))
            + data["x_range"].nbytes
            + data["z_range"].nbytes
            for data in self._memory_cache.values()
        )
        print(
            f"✓ Loaded {len(self._memory_cache)} SDFs into RAM (~{total_bytes / 1024 / 1024:.1f} MB)"
        )
        # print(f"[Ashok] str load idx 1 {self.load('1')}")
        # print(f"[Ashok] int load idx 1 {self.load(int(1))}")

    def load(self, scene_idx: int) -> Optional[Dict]:
        """Load cached SDF data for scene_idx from memory."""
        # print(f"[Ashok] Loading sdf cache for scene idx {scene_idx}, len of memory cache {len(self._memory_cache)}, {self._memory_cache[int(scene_idx)]}, {type(scene_idx)}")
        return self._memory_cache.get(int(scene_idx), None)

    def save(self, scene_idx: int, cache_data: Dict):
        """Save SDF data for scene_idx to memory and disk."""
        # Save to memory
        self._memory_cache[scene_idx] = cache_data

        # Save to disk
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"sdf_{scene_idx}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)


class SDFBoundaryChecker:
    """Lightweight SDF checker with caching support."""

    def __init__(self, floor_vertices, grid_resolution=0.05):
        self.floor_vertices = np.array(floor_vertices, dtype=np.float32)
        self.grid_resolution = grid_resolution

        # Validate polygon
        self._validate_polygon()

        # Auto-compute bounds with adaptive padding
        padding = max(2.0, self.grid_resolution * 10)  # At least 10 cells padding
        min_x = self.floor_vertices[:, 0].min() - padding
        max_x = self.floor_vertices[:, 0].max() + padding
        min_z = self.floor_vertices[:, 1].min() - padding
        max_z = self.floor_vertices[:, 1].max() + padding

        self.world_bounds = (min_x, max_x, min_z, max_z)
        self.sdf_grid, self.x_range, self.z_range = self._compute_sdf_grid()

    def _validate_polygon(self):
        """Validate polygon is well-formed."""
        if len(self.floor_vertices) < 3:
            raise ValueError(
                f"Polygon must have at least 3 vertices, got {len(self.floor_vertices)}"
            )

        # Check for duplicate consecutive vertices
        for i in range(len(self.floor_vertices)):
            j = (i + 1) % len(self.floor_vertices)
            if np.allclose(self.floor_vertices[i], self.floor_vertices[j], atol=1e-6):
                print(f"Warning: Duplicate vertices at indices {i}, {j}")

        # Compute signed area to check orientation
        area = 0.0
        n = len(self.floor_vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.floor_vertices[i, 0] * self.floor_vertices[j, 1]
            area -= self.floor_vertices[j, 0] * self.floor_vertices[i, 1]
        area *= 0.5

        if abs(area) < 1e-6:
            raise ValueError("Polygon has zero area (degenerate)")

        # Store orientation for consistent winding
        self.is_ccw = area > 0

    def get_cache_data(self) -> Dict:
        """Get data to save to cache."""
        return {
            "sdf_grid": self.sdf_grid,
            "world_bounds": self.world_bounds,
            "grid_resolution": self.grid_resolution,
            "x_range": self.x_range,
            "z_range": self.z_range,
            "is_ccw": self.is_ccw,
        }

    @classmethod
    def from_cache_data(cls, cache_data: Dict) -> "SDFBoundaryChecker":
        """Create checker from cached data (avoids recomputation)."""
        obj = cls.__new__(cls)
        obj.sdf_grid = cache_data["sdf_grid"]
        obj.world_bounds = cache_data["world_bounds"]
        obj.grid_resolution = cache_data["grid_resolution"]
        obj.x_range = cache_data["x_range"]
        obj.z_range = cache_data["z_range"]
        obj.is_ccw = cache_data.get("is_ccw", True)  # Backwards compatibility
        obj.floor_vertices = None  # Not needed for queries
        return obj

    def check_violation(self, point: Tuple[float, float]) -> float:
        """
        Check boundary violation at point.
        Returns 0.0 if inside, positive distance if outside.
        """
        sdf_value = self._query_sdf(point)
        return max(0.0, -sdf_value)

    def _query_sdf(self, point: Tuple[float, float]) -> float:
        """Query SDF value using bilinear interpolation."""
        x, z = point

        # NEW: Check for NaN inputs
        if np.isnan(x) or np.isnan(z):
            return -999.0  # Treat as far outside (penalize heavily)

        min_x, max_x, min_z, max_z = self.world_bounds

        # Check if way out of bounds (beyond grid)
        margin = self.grid_resolution * 5  # 5 cells beyond grid
        if (
            x < min_x - margin
            or x > max_x + margin
            or z < min_z - margin
            or z > max_z + margin
        ):
            return -999.0  # Far outside

        # Convert to grid coordinates
        x_idx = (x - min_x) / self.grid_resolution
        z_idx = (z - min_z) / self.grid_resolution

        # Handle edge cases - clamp to valid range
        x_idx = np.clip(x_idx, 0, len(self.x_range) - 1.001)
        z_idx = np.clip(z_idx, 0, len(self.z_range) - 1.001)

        # Bilinear interpolation
        x0, x1 = int(np.floor(x_idx)), int(np.ceil(x_idx))
        z0, z1 = int(np.floor(z_idx)), int(np.ceil(z_idx))

        # Ensure indices are valid
        x0 = np.clip(x0, 0, self.sdf_grid.shape[1] - 1)
        x1 = np.clip(x1, 0, self.sdf_grid.shape[1] - 1)
        z0 = np.clip(z0, 0, self.sdf_grid.shape[0] - 1)
        z1 = np.clip(z1, 0, self.sdf_grid.shape[0] - 1)

        # Fractional parts for interpolation
        fx = x_idx - np.floor(x_idx)
        fz = z_idx - np.floor(z_idx)

        # Bilinear interpolation weights
        v00 = self.sdf_grid[z0, x0]
        v10 = self.sdf_grid[z0, x1]
        v01 = self.sdf_grid[z1, x0]
        v11 = self.sdf_grid[z1, x1]

        # Check for any -999 in the interpolation corners (edge artifact)
        if any(v < -500 for v in [v00, v10, v01, v11]):
            return min(v00, v10, v01, v11)  # Return worst case

        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx

        return v0 * (1 - fz) + v1 * fz

    def _point_in_polygon(self, point: np.ndarray) -> bool:
        """Ray casting algorithm."""
        x, z = point
        vertices = self.floor_vertices
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, zi = vertices[i]
            xj, zj = vertices[j]

            if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            j = i

        return inside

    def _distance_to_segment(
        self, point: np.ndarray, v1: np.ndarray, v2: np.ndarray
    ) -> float:
        """Distance from point to line segment."""
        px, pz = point
        v1x, v1z = v1
        v2x, v2z = v2

        dx = v2x - v1x
        dz = v2z - v1z
        length_sq = dx * dx + dz * dz

        if length_sq == 0:
            return np.sqrt((px - v1x) ** 2 + (pz - v1z) ** 2)

        t = ((px - v1x) * dx + (pz - v1z) * dz) / length_sq
        t = np.clip(t, 0, 1)

        proj_x = v1x + t * dx
        proj_z = v1z + t * dz

        return np.sqrt((px - proj_x) ** 2 + (pz - proj_z) ** 2)

    def _distance_to_polygon(self, point: np.ndarray) -> float:
        """Minimum distance to polygon boundary."""
        min_dist = float("inf")
        n = len(self.floor_vertices)

        for i in range(n):
            j = (i + 1) % n
            dist = self._distance_to_segment(
                point, self.floor_vertices[i], self.floor_vertices[j]
            )
            min_dist = min(min_dist, dist)

        return min_dist

    def _compute_sdf_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute SDF grid."""
        min_x, max_x, min_z, max_z = self.world_bounds

        x_range = np.arange(min_x, max_x, self.grid_resolution)
        z_range = np.arange(min_z, max_z, self.grid_resolution)

        sdf_grid = np.zeros((len(z_range), len(x_range)), dtype=np.float32)

        for i, z in enumerate(z_range):
            for j, x in enumerate(x_range):
                point = np.array([x, z])
                dist = self._distance_to_polygon(point)
                is_inside = self._point_in_polygon(point)
                sdf_grid[i, j] = dist if is_inside else -dist

        return sdf_grid, x_range, z_range


def visualize_sdf_grid(
    scene_idx: int,
    sdf_cache_dir: str = "./sdf_cache",
    split: str = "train_val",
    save_path: str = None,
):
    """
    Debug helper: Visualize the SDF grid for a scene.
    Helps identify issues with complex polygons.

    Args:
        scene_idx: Scene index to visualize
        sdf_cache_dir: Base cache directory
        split: "train_val" or "test"
        save_path: Optional path to save image (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    cache = SDFCache(sdf_cache_dir, grid_resolution=0.05, split=split)
    sdf_data = cache.load(scene_idx)

    if sdf_data is None:
        print(f"Scene {scene_idx} not found in {split} cache!")
        return

    sdf_grid = sdf_data["sdf_grid"]

    # Plot SDF
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Clamp for visualization
    plot_grid = np.clip(sdf_grid, -5, 5)

    im = ax.imshow(plot_grid, origin="lower", cmap="RdBu", vmin=-5, vmax=5)
    ax.set_title(f"SDF Grid - Scene {scene_idx} ({split})\nRed=Outside, Blue=Inside")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.colorbar(im, ax=ax, label="Signed Distance")

    # Draw zero contour (boundary)
    ax.contour(sdf_grid, levels=[0], colors="black", linewidths=2, origin="lower")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print stats
    print(f"\nSDF Stats for scene {scene_idx} ({split}):")
    print(f"  Grid shape: {sdf_grid.shape}")
    print(f"  Min: {sdf_grid.min():.3f}")
    print(f"  Max: {sdf_grid.max():.3f}")
    print(f"  Mean: {sdf_grid.mean():.3f}")
    print(f"  % Inside (>0): {(sdf_grid > 0).mean() * 100:.1f}%")
