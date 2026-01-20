"""
Weighted MLS Smoothing on Noisy Grid PointCloud
================================================

This script demonstrates the use of Open3D's Weighted MLS (Moving Least Squares) 
smoothing on a noisy 3D grid. It compares the point cloud before and after smoothing,
shows the movement vectors, and visualizes the result with both Open3D and Matplotlib.

Dependencies:
-------------
- numpy
- open3d
- matplotlib

Usage:
------
python weighted_mls_demo.py

Author: Tsofnat Maman
Date: 2026-01-15
"""

import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt

# =========================
# 1. Create a noisy grid
# =========================
n_points = 60
x = np.linspace(-3, 3, n_points)
y = np.linspace(-3, 3, n_points)
xx, yy = np.meshgrid(x, y)
zz = np.sin(xx) * np.cos(yy)

# Stack into Nx3 points array and add Gaussian noise
points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
points += 0.1 * np.random.randn(*points.shape)

# =========================
# 2. Create Open3D PointCloud
# =========================
pcd_before = o3d.geometry.PointCloud()
pcd_before.points = o3d.utility.Vector3dVector(points)

# Print first few points before smoothing
print("=== BEFORE smoothing (first 5 points) ===")
print(np.asarray(pcd_before.points)[:5])

# =========================
# 3. Apply Weighted MLS smoothing
# =========================
# Open3D >= 0.17 required for smooth() method
pcd_after = pcd_before.smooth_mls(o3d.geometry.KDTreeSearchParamRadius(radius=0.6))

# Print first few points after smoothing
points_after = np.asarray(pcd_after.points)
print("\n=== AFTER smoothing (first 5 points) ===")
print(points_after[:5])

# =========================
# 4. Compute differences and statistics
# =========================
points_before = np.asarray(pcd_before.points)
diff = points_after - points_before
movement = np.linalg.norm(diff, axis=1)

print("\n=== DIFFERENCE (first 5 points) ===")
print(diff[:5])
print("\n=== MOVEMENT magnitude (first 10 points) ===")
print(movement[:10])
print("\n=== SUMMARY ===")
print(f"Min movement: {movement.min():.6f}")
print(f"Max movement: {movement.max():.6f}")
print(f"Mean movement: {movement.mean():.6f}")

# =========================
# 5. Create shifted point clouds for visualization
# =========================
shift_distance = 10

pcd_before_shifted = copy.deepcopy(pcd_before)
pcd_before_shifted.translate([-shift_distance, 0, 0], relative=False)
pcd_before.paint_uniform_color([1, 0, 0])  # red original
pcd_before_shifted.paint_uniform_color([1, 0, 0])  # red shifted

pcd_after_shifted = copy.deepcopy(pcd_after)
pcd_after_shifted.translate([shift_distance, 0, 0], relative=False)
pcd_after.paint_uniform_color([0, 1, 0])  # green original
pcd_after_shifted.paint_uniform_color([0, 1, 0])  # green shifted

# =========================
# 6. Create lines connecting original and smoothed points
# =========================
lines = [[i, i + len(points)] for i in range(len(points))]
colors = [[0, 0, 1] for _ in lines]  # blue lines
points_combined = np.vstack([points_before, points_after])
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points_combined),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

# =========================
# 7. Visualize with Open3D
# =========================
o3d.visualization.draw_geometries([
    pcd_before_shifted,
    pcd_after_shifted,
    pcd_before,
    pcd_after,
    line_set
], window_name="Weighted MLS: Comparison")

# =========================
# 8. 3D scatter plots with Matplotlib
# =========================
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(points_before[:, 0], points_before[:, 1], points_before[:, 2], s=2, c='r')
ax1.set_title("Before smoothing")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(points_after[:, 0], points_after[:, 1], points_after[:, 2], s=2, c='g')
ax2.set_title("After Weighted MLS smoothing")

plt.show()
