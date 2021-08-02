import tensorflow as tf
import open3d as o3d
import numpy as np
from open3d.visualization.tensorboard_plugin import summary


def summary_format(o3d_trimesh):
    return {
        'vertex_positions':
            np.expand_dims(np.asarray(o3d_trimesh.vertices), axis=0),
        'vertex_colors':
            np.expand_dims(np.asarray(o3d_trimesh.vertex_colors), axis=0),
        'vertex_normals':
            np.expand_dims(np.asarray(o3d_trimesh.vertex_normals), axis=0),
        'triangle_indices':
            np.expand_dims(np.asarray(o3d_trimesh.triangles), axis=0)
    }


def main():
    writer = tf.summary.create_file_writer("demo_logs")

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
            summary.add_3d('cube', summary_format(cube), step=step)
            cylinder.paint_uniform_color(colors[step])
            summary.add_3d('cylinder', summary_format(cylinder), step=step)


if __name__ == "__main__":
    main()
