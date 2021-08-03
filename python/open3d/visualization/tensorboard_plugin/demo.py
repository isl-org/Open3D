import copy
import numpy as np
import tensorflow as tf
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary


def summary_format(o3d_trimesh_list):
    vertex_positions = []
    vertex_colors = []
    vertex_normals = []
    triangle_indices = []
    for trimesh in o3d_trimesh_list:
        vertex_positions.append(np.asarray(trimesh.vertices))
        vertex_colors.append(np.asarray(trimesh.vertex_colors))
        vertex_normals.append(np.asarray(trimesh.vertex_normals))
        triangle_indices.append(np.asarray(trimesh.triangles))

    return {
        'vertex_positions': np.stack(vertex_positions, axis=0),
        'vertex_colors': np.stack(vertex_colors, axis=0),
        'vertex_normals': np.stack(vertex_normals, axis=0),
        'triangle_indices': np.stack(triangle_indices, axis=0),
    }


def small_scale():

    writer = tf.summary.create_file_writer("demo_logs/small_scale")

    cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube.compute_vertex_normals()
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0,
                                                         height=2.0,
                                                         resolution=20,
                                                         split=4)
    cylinder.compute_vertex_normals()
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    with writer.as_default():
        for step in range(2):
            cube.paint_uniform_color(colors[step])
            summary.add_3d('cube', summary_format([cube]), step=step)
            cylinder.paint_uniform_color(colors[step])
            summary.add_3d('cylinder', summary_format([cylinder]), step=step)


def large_scale(n_steps=40, batch_size=8, base_resolution=20):
    """Generate a large scale summary. Geometry resolution increases linearly
    with step. Each element in a batch is painted a different color.
    """
    writer = tf.summary.create_file_writer("demo_logs/large_scale")
    colors = []
    for k in range(batch_size):
        t = k * np.pi / batch_size
        colors.append(((1 + np.sin(t)) / 2, (1 + np.cos(t)) / 2, t / np.pi))
    with writer.as_default():
        for step in range(n_steps):
            resolution = base_resolution * (step + 1)
            cylinder_list = []
            moebius_list = []
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=1.0, height=2.0, resolution=resolution, split=4)
            cylinder.compute_vertex_normals()
            moebius = o3d.geometry.TriangleMesh.create_moebius(
                length_split=int(3.5 * resolution),
                width_split=int(0.75 * resolution),
                twists=1,
                raidus=1,
                flatness=1,
                width=1,
                scale=1)
            moebius.compute_vertex_normals()
            for b in range(batch_size):
                cylinder_list.append(copy.deepcopy(cylinder))
                cylinder_list[b].paint_uniform_color(colors[b])
                moebius_list.append(copy.deepcopy(moebius))
                moebius_list[b].paint_uniform_color(colors[b])
            summary.add_3d('cylinder',
                           summary_format(cylinder_list),
                           max_outputs=batch_size,
                           step=step)
            summary.add_3d('moebius',
                           summary_format(moebius_list),
                           max_outputs=batch_size,
                           step=step)


if __name__ == "__main__":
    small_scale()
    large_scale()
