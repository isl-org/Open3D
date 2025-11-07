import numpy as np
import os
import open3d as o3d

def extrude_along_normals(
    mesh: o3d.geometry.TriangleMesh,
    thickness: float = 0.1,
        color=None
) -> o3d.geometry.TriangleMesh:
    """
    将 mesh 沿自身顶点法线方向挤出（extrude）指定厚度，形成具有厚度的 3D 实体。
    适用于已经计算好法线的 TriangleMesh。

    :param mesh: 输入的三角网格（o3d.geometry.TriangleMesh），需已计算法线
    :param thickness: 挤出厚度
    :param color: 输出 mesh 的颜色
    :return: extruded mesh (o3d.geometry.TriangleMesh)
    """
    # === 1️⃣ 确保法线存在 ===
    if color is None:
        color = [1.0, 0.6, 0.3]
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    N = len(vertices)

    # === 2️⃣ 生成上下面的顶点 ===
    top_vertices = vertices + normals * thickness
    bottom_vertices = vertices  # 原 mesh 作为底面

    # === 3️⃣ 构建上下表面 ===
    top_triangles = triangles
    bottom_triangles = np.flip(triangles + N, axis=1)  # 反向法线

    # === 4️⃣ 构造侧面 ===
    edge_set = set()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edge_set.add(edge)

    side_tris = []
    for a, b in edge_set:
        side_tris.append([a, b, a + N])
        side_tris.append([b, b + N, a + N])

    # === 5️⃣ 合并所有部分 ===
    all_vertices = np.vstack([bottom_vertices, top_vertices])
    all_triangles = np.vstack([bottom_triangles, top_triangles, np.array(side_tris)])

    extruded_mesh = o3d.geometry.TriangleMesh()
    extruded_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    extruded_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    extruded_mesh.paint_uniform_color(color)
    extruded_mesh.compute_vertex_normals()

    return extruded_mesh