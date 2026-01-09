"""Diagnostic script to compare two FPBPN files and identify differences."""

import numpy as np
import sys

def compare_fpbpn(file1_path, file2_path):
    """Compare two FPBPN files and print detailed differences."""
    
    fpbpn1 = np.load(file1_path)
    fpbpn2 = np.load(file2_path)
    
    print("=" * 80)
    print("FPBPN COMPARISON")
    print("=" * 80)
    print(f"\nFile 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    print(f"\nShapes:")
    print(f"  File 1: {fpbpn1.shape}")
    print(f"  File 2: {fpbpn2.shape}")
    
    if fpbpn1.shape != fpbpn2.shape:
        print(f"\n⚠️  WARNING: Shapes don't match!")
        return
    
    print(f"\n{'='*80}")
    print("COORDINATE RANGES (x, y)")
    print("=" * 80)
    print(f"\nFile 1 (generated):")
    print(f"  x: [{fpbpn1[:, 0].min():.6f}, {fpbpn1[:, 0].max():.6f}]")
    print(f"  y: [{fpbpn1[:, 1].min():.6f}, {fpbpn1[:, 1].max():.6f}]")
    print(f"  Range width x: {fpbpn1[:, 0].max() - fpbpn1[:, 0].min():.6f}")
    print(f"  Range width y: {fpbpn1[:, 1].max() - fpbpn1[:, 1].min():.6f}")
    
    print(f"\nFile 2 (dataset):")
    print(f"  x: [{fpbpn2[:, 0].min():.6f}, {fpbpn2[:, 0].max():.6f}]")
    print(f"  y: [{fpbpn2[:, 1].min():.6f}, {fpbpn2[:, 1].max():.6f}]")
    print(f"  Range width x: {fpbpn2[:, 0].max() - fpbpn2[:, 0].min():.6f}")
    print(f"  Range width y: {fpbpn2[:, 1].max() - fpbpn2[:, 1].min():.6f}")
    
    print(f"\n{'='*80}")
    print("NORMAL RANGES (nx, ny)")
    print("=" * 80)
    print(f"\nFile 1 (generated):")
    print(f"  nx: [{fpbpn1[:, 2].min():.6f}, {fpbpn1[:, 2].max():.6f}]")
    print(f"  ny: [{fpbpn1[:, 3].min():.6f}, {fpbpn1[:, 3].max():.6f}]")
    print(f"  Normal magnitudes: min={np.linalg.norm(fpbpn1[:, 2:4], axis=1).min():.6f}, "
          f"max={np.linalg.norm(fpbpn1[:, 2:4], axis=1).max():.6f}")
    
    print(f"\nFile 2 (dataset):")
    print(f"  nx: [{fpbpn2[:, 2].min():.6f}, {fpbpn2[:, 2].max():.6f}]")
    print(f"  ny: [{fpbpn2[:, 3].min():.6f}, {fpbpn2[:, 3].max():.6f}]")
    print(f"  Normal magnitudes: min={np.linalg.norm(fpbpn2[:, 2:4], axis=1).min():.6f}, "
          f"max={np.linalg.norm(fpbpn2[:, 2:4], axis=1).max():.6f}")
    
    print(f"\n{'='*80}")
    print("STATISTICS")
    print("=" * 80)
    print(f"\nFile 1 (generated):")
    print(f"  Mean: [{fpbpn1[:, 0].mean():.6f}, {fpbpn1[:, 1].mean():.6f}, "
          f"{fpbpn1[:, 2].mean():.6f}, {fpbpn1[:, 3].mean():.6f}]")
    print(f"  Std:  [{fpbpn1[:, 0].std():.6f}, {fpbpn1[:, 1].std():.6f}, "
          f"{fpbpn1[:, 2].std():.6f}, {fpbpn1[:, 3].std():.6f}]")
    
    print(f"\nFile 2 (dataset):")
    print(f"  Mean: [{fpbpn2[:, 0].mean():.6f}, {fpbpn2[:, 1].mean():.6f}, "
          f"{fpbpn2[:, 2].mean():.6f}, {fpbpn2[:, 3].mean():.6f}]")
    print(f"  Std:  [{fpbpn2[:, 0].std():.6f}, {fpbpn2[:, 1].std():.6f}, "
          f"{fpbpn2[:, 2].std():.6f}, {fpbpn2[:, 3].std():.6f}]")
    
    print(f"\n{'='*80}")
    print("SAMPLE VALUES (first 5 points)")
    print("=" * 80)
    print(f"\nFile 1 (generated):")
    for i in range(min(5, len(fpbpn1))):
        print(f"  Point {i}: x={fpbpn1[i, 0]:.6f}, y={fpbpn1[i, 1]:.6f}, "
              f"nx={fpbpn1[i, 2]:.6f}, ny={fpbpn1[i, 3]:.6f}")
    
    print(f"\nFile 2 (dataset):")
    for i in range(min(5, len(fpbpn2))):
        print(f"  Point {i}: x={fpbpn2[i, 0]:.6f}, y={fpbpn2[i, 1]:.6f}, "
              f"nx={fpbpn2[i, 2]:.6f}, ny={fpbpn2[i, 3]:.6f}")
    
    print(f"\n{'='*80}")
    print("DIFFERENCES")
    print("=" * 80)
    
    # Check if coordinate ranges are similar
    x_range1 = fpbpn1[:, 0].max() - fpbpn1[:, 0].min()
    x_range2 = fpbpn2[:, 0].max() - fpbpn2[:, 0].min()
    y_range1 = fpbpn1[:, 1].max() - fpbpn1[:, 1].min()
    y_range2 = fpbpn2[:, 1].max() - fpbpn2[:, 1].min()
    
    print(f"\nCoordinate range differences:")
    print(f"  X range ratio: {x_range1/x_range2:.3f} (should be ~1.0)")
    print(f"  Y range ratio: {y_range1/y_range2:.3f} (should be ~1.0)")
    
    if abs(x_range1/x_range2 - 1.0) > 0.2 or abs(y_range1/y_range2 - 1.0) > 0.2:
        print(f"\n⚠️  WARNING: Coordinate ranges differ significantly!")
        print(f"   This could cause the model to fail.")
    
    # Check normal magnitudes
    norm_mags1 = np.linalg.norm(fpbpn1[:, 2:4], axis=1)
    norm_mags2 = np.linalg.norm(fpbpn2[:, 2:4], axis=1)
    
    print(f"\nNormal magnitude check:")
    print(f"  File 1: all close to 1.0? {np.allclose(norm_mags1, 1.0, atol=0.01)}")
    print(f"  File 2: all close to 1.0? {np.allclose(norm_mags2, 1.0, atol=0.01)}")
    
    if not np.allclose(norm_mags1, 1.0, atol=0.01):
        print(f"  ⚠️  WARNING: File 1 normals are not normalized!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_fpbpn.py <file1.npy> <file2.npy>")
        print("\nExample:")
        print("  python scripts/compare_fpbpn.py tmp/fpbpn.npy outputs/2025-12-17/06-52-15/fpbpn_sample_idx_0.npy")
        sys.exit(1)
    
    compare_fpbpn(sys.argv[1], sys.argv[2])
