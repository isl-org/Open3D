import copy
import numpy as np
import tensorflow as tf
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary


def to_dict_batch(o3d_geometry_list):
    """
    Convert sequence of identical Open3D geometry types to attribute - tensor
    dictionary. The geometry seequence forms a batch of data.
    TODO: Add support for point cloud and line set.
    TODO: This involves a data copy. Add support for List[Open3D geometry]
    directoy to add_3d() if needed.

    Args:
        o3d_geometry_list (Iterable): Iterable (list / tuple / sequence
            generator) of Open3D Tensor geometry types.
    """
    if len(o3d_geometry_list) == 0:
        return {}
    if isinstance(o3d_geometry_list[0], o3d.geometry.PointCloud):
        vertex_positions = []
        vertex_colors = []
        vertex_normals = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.points))
            vertex_colors.append(np.asarray(geometry.colors))
            vertex_normals.append(np.asarray(geometry.normals))

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
        }
    if isinstance(o3d_geometry_list[0], o3d.geometry.TriangleMesh):
        vertex_positions = []
        vertex_colors = []
        vertex_normals = []
        triangle_indices = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.vertices))
            vertex_colors.append(np.asarray(geometry.vertex_colors))
            vertex_normals.append(np.asarray(geometry.vertex_normals))
            triangle_indices.append(np.asarray(geometry.triangles))

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
            'triangle_indices': np.stack(triangle_indices, axis=0),
        }

    if isinstance(o3d_geometry_list[0], o3d.geometry.LineSet):
        vertex_positions = []
        line_colors = []
        line_indices = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.points))
            line_colors.append(np.asarray(geometry.colors))
            line_indices.append(np.asarray(geometry.lines))

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'line_colors': np.stack(line_colors, axis=0),
            'line_indices': np.stack(line_indices, axis=0),
        }

    raise NotImplementedError(
        f"Geometry type {type(o3d_geometry_list[0])} is not suported yet.")


def small_scale(run_name="small_scale"):
    """Basic demo with cube and cylinder with normals and colors.
    """

    writer = tf.summary.create_file_writer("demo_logs/" + run_name)

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
            summary.add_3d('cube', to_dict_batch([cube]), step=step)
            cylinder.paint_uniform_color(colors[step])
            summary.add_3d('cylinder', to_dict_batch([cylinder]), step=step)


def property_reference(run_name="property_reference"):
    """Produces identical visualization to small_scale, but does not store
    repeated properties of ``vertex_positions`` and ``vertex_normals``.
    """

    writer = tf.summary.create_file_writer("demo_logs/" + run_name)

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
            cube_summary = to_dict_batch([cube])
            if step > 0:
                cube_summary['vertex_positions'] = 0
                cube_summary['vertex_normals'] = 0
            summary.add_3d('cube', cube_summary, step=step)
            cylinder.paint_uniform_color(colors[step])
            cylinder_summary = to_dict_batch([cylinder])
            if step > 0:
                cylinder_summary['vertex_positions'] = 0
                cylinder_summary['vertex_normals'] = 0
            summary.add_3d('cylinder', cylinder_summary, step=step)


def large_scale(n_steps=20,
                batch_size=1,
                base_resolution=200,
                run_name="large_scale"):
    """Generate a large scale summary. Geometry resolution increases linearly
    with step. Each element in a batch is painted a different color.
    """
    writer = tf.summary.create_file_writer("demo_logs/" + run_name)
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
                           to_dict_batch(cylinder_list),
                           max_outputs=batch_size,
                           step=step)
            summary.add_3d('moebius',
                           to_dict_batch(moebius_list),
                           max_outputs=batch_size,
                           step=step)


if __name__ == "__main__":
    small_scale()
    property_reference()
    # large_scale()
