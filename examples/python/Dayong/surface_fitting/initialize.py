import numpy as np
import os
import open3d as o3d

def read_stl_to_mesh(stl_path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    initialize_mesh(mesh)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def initialize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # Translate the center of the mesh to the origin
    mesh.translate(-mesh.get_center())
    return mesh