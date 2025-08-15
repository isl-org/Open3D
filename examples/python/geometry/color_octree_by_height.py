
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic point cloud 
xyz = np.random.rand(10000, 3) * 10  # Points with x, y, z in [0, 10]
rgb_colors = np.random.rand(10000, 3)  # Random RGB colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

# Normalize z-coordinates for height-based color mapping
z = xyz[:, 2]
z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

# Scale point cloud to unit cube
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())

# Create octree
print('Octree division')
octree = o3d.geometry.Octree(max_depth=5)
octree.convert_from_point_cloud(pcd, size_expand=0.01)

# Available color maps for height-based mode
color_maps = ['jet', 'hot', 'viridis', 'cool']
current_cmap_index = [0]
color_mode = ['height']

# Function to apply color to point cloud and octree
def apply_color(pcd, octree, mode, cmap_name=None, z_normalized=None, rgb_colors=None):
    if mode == 'height':
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(z_normalized)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    def color_octree_leaves(octree, pcd):
        def traverse_and_color(node, node_info):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                min_bound = node_info.origin
                size = node_info.size
                max_bound = min_bound + np.array([size, size, size])
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
                if np.sum(mask) > 0:
                    avg_color = np.mean(colors[mask], axis=0)
                    node.color = avg_color
                else:
                    node.color = np.array([0.5, 0.5, 0.5])
        octree.traverse(traverse_and_color)

    color_octree_leaves(octree, pcd)
    return pcd, octree

pcd, octree = apply_color(pcd, octree, 'height', cmap_name=color_maps[current_cmap_index[0]], z_normalized=z_normalized, rgb_colors=rgb_colors)

# Custom visualization with key callbacks
def custom_visualize(pcd, octree):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Octree and Point Cloud Visualization')
    vis.add_geometry(pcd)
    vis.add_geometry(octree)

    def toggle_color_map(vis):
        nonlocal pcd, octree
        if color_mode[0] == 'height':
            current_cmap_index[0] = (current_cmap_index[0] + 1) % len(color_maps)
            new_cmap = color_maps[current_cmap_index[0]]
            print(f"Switching to color map: {new_cmap} (height mode)")
            pcd, octree = apply_color(pcd, octree, 'height', cmap_name=new_cmap, z_normalized=z_normalized, rgb_colors=rgb_colors)
            vis.update_geometry(pcd)
            vis.update_geometry(octree)
            vis.update_renderer()
        else:
            print("In RGB mode; press '5' to switch to height-based coloring.")
        return False

    def toggle_color_mode(vis):
        nonlocal pcd, octree
        color_mode[0] = 'rgb' if color_mode[0] == 'height' else 'height'
        if color_mode[0] == 'height':
            print(f"Switching to height mode with color map: {color_maps[current_cmap_index[0]]}")
            pcd, octree = apply_color(pcd, octree, 'height', cmap_name=color_maps[current_cmap_index[0]], z_normalized=z_normalized, rgb_colors=rgb_colors)
        else:
            print("Switching to RGB mode")
            pcd, octree = apply_color(pcd, octree, 'rgb', rgb_colors=rgb_colors)
        vis.update_geometry(pcd)
        vis.update_geometry(octree)
        vis.update_renderer()
        return False

    vis.register_key_callback(ord('4'), toggle_color_map)
    vis.register_key_callback(ord('5'), toggle_color_mode)
    print("Visualizing with initial color map: jet (height mode)")
    print("Press '4' to cycle color maps (jet, hot, viridis, cool) in height mode")
    print("Press '5' to toggle between height-based and RGB-based coloring")
    vis.run()
    vis.destroy_window()

# Run visualization
custom_visualize(pcd, octree)