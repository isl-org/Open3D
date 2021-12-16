import open3d as o3d
import numpy as np

scale = 8
block_voxel_ratio = 4


def gen_box_lineset(origin=[0, 0, 0], size=1, color=[0, 0, 0]):
    origin = np.array(origin)
    assert origin.shape == (3,)
    origin = origin.reshape((1, 3))
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                       [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                      dtype=np.float64)
    points = origin + points * size
    lines = np.array(([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6],
                       [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]),
                     dtype=np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set


if __name__ == "__main__":
    # Load mesh
    # mesh = o3d.io.read_triangle_mesh("output.ply")
    # np.savez_compressed("output_mesh.npz",
    #                     vertices=np.asarray(mesh.vertices).astype(np.float16),
    #                     vertex_colors=np.asarray(mesh.vertex_colors).astype(
    #                         np.float16),
    #                     triangles=np.asarray(mesh.triangles).astype(np.int32))
    mesh_data = np.load("output_mesh.npz")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_data["vertices"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_data["vertex_colors"])
    mesh.triangles = o3d.utility.Vector3iVector(mesh_data["triangles"])

    block_size = 3.0 / 512 * 8 * scale
    voxel_size = block_size / block_voxel_ratio

    vertices = np.asarray(mesh.vertices)
    active_block_indices = np.floor(vertices / block_size).astype(int)
    active_block_indices = np.unique(active_block_indices, axis=0)
    active_block_origins = active_block_indices * block_size
    print(f"num blocks: {len(active_block_origins)}")

    block_line_sets = o3d.geometry.LineSet()
    for block_id in range(len(active_block_origins)):
        block_origin = active_block_origins[block_id]
        block_line_set = gen_box_lineset(block_origin, block_size, [0, 0, 0])
        block_line_sets += block_line_set

    max_index = active_block_indices[:, 0].argmax()
    select_block_index = active_block_indices[max_index]
    select_block_origin = select_block_index * block_size

    voxel_line_sets = o3d.geometry.LineSet()
    for i in range(block_voxel_ratio):
        for j in range(block_voxel_ratio):
            for k in range(block_voxel_ratio):
                voxel_origin = select_block_origin + np.array([i, j, k
                                                              ]) * voxel_size
                voxel_line_set = gen_box_lineset(voxel_origin, voxel_size,
                                                 [1, 0, 0])
                voxel_line_sets += voxel_line_set

    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries(
        [mesh, block_line_sets, voxel_line_sets, coords])
