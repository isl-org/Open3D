import numpy as np
import open3d as o3d

def clean_mesh(mesh: o3d.geometry.TriangleMesh, weld_eps=1e-5):
    """Clean and normalize normals"""
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if weld_eps:
        mesh.merge_close_vertices(weld_eps)
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def extrude_shell(mesh, thickness: float):
    m = clean_mesh(mesh)
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    F = np.asarray(m.triangles)

    V_out = V + N * thickness
    V_comb = np.vstack([V, V_out])
    F_in = F
    F_out = F[:, ::-1] + len(V)

    # Construct sides
    def edges_of(F):
        E = set()
        for a, b, c in F:
            for u, v in [(a, b), (b, c), (c, a)]:
                e = (min(u, v), max(u, v))
                E.add(e)
        return list(E)

    E = edges_of(F)
    offset = len(V)
    side_faces = []
    for a, b in E:
        a2, b2 = a + offset, b + offset
        side_faces.append([a, b, b2])
        side_faces.append([a, b2, a2])

    F_all = np.vstack([F_in, F_out, np.array(side_faces)])
    shell = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V_comb),
        triangles=o3d.utility.Vector3iVector(F_all)
    )
    return clean_mesh(shell)