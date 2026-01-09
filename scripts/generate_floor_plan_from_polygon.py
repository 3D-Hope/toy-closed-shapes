"""
Utility script to generate a ThreedFront-style floor-plan mesh NPZ file
from a simple polygon description.

The goal is to let you quickly sketch a floor shape (e.g., as grid
corner points) and convert that into the same `floor_plan_*.npz` format
used by:
  - `scripts/save_floor_plan_from_dataset.py`
  - `ThreedFront/scripts/render_results_3d_custom_floor.py`

The generated NPZ will contain:
  - `floor_plan_vertices`: (V, 3) array of XYZ vertices (Y=0).
  - `floor_plan_faces`: (F, 3) triangle indices (integer).
  - `floor_plan_centroid`: (3,) centroid of the polygon in XYZ.

This is sufficient for `render_results_3d_custom_floor.py`, which:
  - Reads `floor_plan_vertices` and `floor_plan_faces`.
  - Optionally recenters using `floor_plan_centroid`.

Input polygon formats supported:
  1. `.npy` file with shape (N, 2): rows are [x, z] coordinates in the
     same world units as you want to render (e.g., meters).
  2. `.txt` file: one vertex per line, as either:
        x z
     or:
        i j
     grid coordinates. You can then use `--cell_size` and
     `--grid_origin` to map grid indices to world XZ coordinates.

For simple convex polygons (rectangles, L-shapes that are still convex,
etc.) we triangulate with a fan from vertex 0:
    faces = [[0, 1, 2], [0, 2, 3], ...]
If your polygon is concave, you may need a more sophisticated
triangulation; this script is intentionally kept simple.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def load_polygon(path: Path) -> np.ndarray:
    """
    Load a polygon as an (N, 2) array of [x, z] coordinates.

    Supports:
      - .npy: expected shape (N, 2).
      - .txt: whitespace-separated pairs per line, e.g. "x z" or "i j".
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected (N,2) array in {path}, got shape {arr.shape}"
            )
        return arr

    if suffix == ".txt":
        xs, zs = [], []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Line '{line}' in {path} does not have at least 2 values"
                    )
                x_val = float(parts[0])
                z_val = float(parts[1])
                xs.append(x_val)
                zs.append(z_val)
        if not xs:
            raise ValueError(f"No valid vertices found in {path}")
        return np.stack([np.array(xs, dtype=float), np.array(zs, dtype=float)], axis=1)

    raise ValueError(f"Unsupported polygon file type: {path.suffix}")


def grid_to_world(
    ij_vertices: np.ndarray,
    cell_size: float,
    grid_origin: Tuple[float, float],
) -> np.ndarray:
    """
    Convert integer grid indices (i,j) to world (x,z) coordinates.

    Args:
        ij_vertices: (N, 2) array of grid coordinates [i, j].
        cell_size: size of one grid cell in world units (e.g., meters).
        grid_origin: (x0, z0) world coordinate of grid cell (0,0).

    Returns:
        (N, 2) array of [x, z] in world units.
    """
    ij_vertices = np.asarray(ij_vertices, dtype=float)
    if ij_vertices.ndim != 2 or ij_vertices.shape[1] != 2:
        raise ValueError(
            f"Expected (N,2) ij grid vertices, got shape {ij_vertices.shape}"
        )
    x0, z0 = grid_origin
    xs = x0 + ij_vertices[:, 0] * cell_size
    zs = z0 + ij_vertices[:, 1] * cell_size
    return np.stack([xs, zs], axis=1)


def polygon_to_mesh(
    xz_vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a simple polygon (in XZ) to a floor mesh.

    - Y is set to 0 for all vertices.
    - Triangulation is done with a fan from vertex 0:
        faces = [[0, 1, 2], [0, 2, 3], ...]
      which assumes the polygon is convex and vertices are ordered
      consistently (clockwise or counter-clockwise).
    - Centroid is computed as the mean of the vertex positions (XYZ).
    - Face winding is adjusted so triangles face upward (Y+).

    Returns:
        floor_plan_vertices: (V, 3)
        floor_plan_faces: (F, 3)
        floor_plan_centroid: (3,)
    """
    xz_vertices = np.asarray(xz_vertices, dtype=float)
    if xz_vertices.ndim != 2 or xz_vertices.shape[1] != 2:
        raise ValueError(
            f"Expected (N,2) XZ vertices, got shape {xz_vertices.shape}"
        )
    if xz_vertices.shape[0] < 3:
        raise ValueError("Polygon must have at least 3 vertices")

    # Build XYZ vertices WITHOUT sharing (replicate vertices for each triangle).
    # This matches the dataset's floor mesh where vertices can be duplicated across faces.
    V = xz_vertices.shape[0]
    
    # Build fan triangulation: each triangle gets its own 3 vertices (no index sharing)
    # IMPORTANT: Use CCW winding [0, i, i+1] for upward-pointing normals (when polygon is CCW from above)
    verts = []
    faces = []
    
    for i in range(1, V - 1):
        # Triangle uses polygon vertices [0, i, i+1] (CCW order for upward normal)
        v0 = xz_vertices[0]      # fan center
        v1 = xz_vertices[i]      # current corner
        v2 = xz_vertices[i + 1]  # next corner
        
        # Append these 3 vertices as new entries (duplicate positions, not shared indices)
        base_idx = len(verts)
        verts.append([v0[0], 0.0, v0[1]])  # X, Y=0, Z
        verts.append([v1[0], 0.0, v1[1]])
        verts.append([v2[0], 0.0, v2[1]])
        
        # Face references these 3 new vertices
        faces.append([base_idx, base_idx + 1, base_idx + 2])
    
    vertices = np.asarray(verts, dtype=float)    # Shape: ((V-2)*3, 3)
    faces = np.asarray(faces, dtype=int)         # Shape: ((V-2), 3)
    
    # Centroid as mean of all vertex positions
    centroid = vertices.mean(axis=0)
    
    return vertices, faces, centroid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a ThreedFront-style floor_plan_*.npz file from a simple "
            "polygon (e.g., drawn on a grid)."
        )
    )
    parser.add_argument(
        "polygon_file",
        type=str,
        help=(
            "Path to a .npy (N,2) or .txt polygon file. For .npy, entries are "
            "[x,z] in world units. For .txt, entries are 'x z' or 'i j' per line."
        ),
    )
    parser.add_argument(
        "--assume_grid",
        action="store_true",
        help=(
            "Interpret polygon_file values as integer grid indices (i,j), and "
            "map them to world coordinates using --cell_size and --grid_origin."
        ),
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=0.5,
        help=(
            "World size of one grid cell (used only if --assume_grid is set). "
            "For example, 0.5 means each grid step is 0.5 meters."
        ),
    )
    parser.add_argument(
        "--grid_origin",
        type=str,
        default="0.0,0.0",
        help=(
            "World origin (x0,z0) for grid cell (0,0), as 'x0,z0'. "
            "Only used if --assume_grid is set."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output NPZ file (e.g., floor_plan_custom.npz).",
    )

    args = parser.parse_args()

    polygon_path = Path(args.polygon_file)
    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygon_path}")

    # Load raw (N,2) vertices.
    raw_verts = load_polygon(polygon_path)

    # Optionally map grid indices to world coordinates.
    if args.assume_grid:
        try:
            gx, gz = [float(v) for v in args.grid_origin.split(",")]
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Invalid --grid_origin '{args.grid_origin}', expected 'x0,z0'"
            ) from e
        xz_vertices = grid_to_world(raw_verts, cell_size=args.cell_size, grid_origin=(gx, gz))
    else:
        xz_vertices = raw_verts

    vertices, faces, centroid = polygon_to_mesh(xz_vertices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        floor_plan_vertices=vertices,
        floor_plan_faces=faces,
        floor_plan_centroid=centroid,
    )

    print(f"[INFO] Saved floor-plan NPZ to: {output_path}")
    print(f"  - floor_plan_vertices shape: {vertices.shape}")
    print(f"  - floor_plan_faces shape: {faces.shape}")
    print(f"  - floor_plan_centroid: {centroid}")


if __name__ == "__main__":
    main()
    
"""
python scripts/generate_floor_plan_from_polygon.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/polygon_world.npy --output /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/tmp/floor_plan_world.npz
"""

