import numpy as np
import os
import open3d as o3d

from initialize import *
from align import *
from project import *
from extrude import *
from utils import *

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    shoe_path = os.path.join(base_dir, "STLs", "shoe.stl")
    toy_path = os.path.join(base_dir, "STLs", "fish.stl")

    # Generate mesh for each stl
    shoe = read_stl_to_mesh(shoe_path)
    toy = read_stl_to_mesh(toy_path)
    shoe.paint_uniform_color([0.6, 0.8, 1.0]) # light blue
    toy.paint_uniform_color([1.0, 0.6, 0.3]) # orange
    # toy = toy.scale(5.0, center=toy.get_center()) # scale to 2x size; optional
    o3d.visualization.draw_geometries([shoe, toy], mesh_show_back_face=True)

    # Get o and its normal
    # o = select_o_point_from_ratio(shoe, 0.5, 0.3, 0.9) # front
    o = select_o_point_from_ratio(shoe, 0.5, 0.9, 0.5) # back
    visualize_point_on_mesh(shoe, o, 1.0)
    normal_o = get_normal_at_point(shoe, o)

    # Get p and its normal
    # p = select_o_point_from_ratio(toy, 0.5, 0.5, 0) # badge
    p = select_o_point_from_ratio(toy, 0.5, 1, 0.5) # fish
    visualize_point_on_mesh(toy, p, 0.1)
    normal_p = get_normal_at_point(toy, p)

    # Rotate toy
    R = get_rotation_matrix(from_vec=-normal_p, to_vec=normal_o)
    toy.rotate(R, center=p)  # rotate toy based on p

    # Move p to o
    offset = o - p
    toy.translate(offset)

    # Project the entire toy to the shoe
    all_points = np.asarray(toy.vertices) + normal_o * 10
    projected_points, hit_mask = project_points_onto_mesh(all_points, -normal_o, shoe)
    toy.vertices = o3d.utility.Vector3dVector(projected_points)
    o3d.visualization.draw_geometries([shoe, toy], mesh_show_back_face=True)

    toy.compute_triangle_normals()
    toy.compute_vertex_normals()

    toy = clean_mesh(toy, weld_eps=1e-5)
    toy.compute_triangle_normals()
    toy.compute_vertex_normals()

    thickness = 0.2
    extruded_toy = extrude_shell(toy, thickness)
    extruded_toy.paint_uniform_color([1.0, 0.6, 0.3])

    o3d.visualization.draw_geometries([extruded_toy], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([shoe, extruded_toy], mesh_show_back_face=True)

    # Extrude
    # extruded_toy = extrude_along_normals(toy, thickness=0.06)
    # o3d.visualization.draw_geometries([extruded_toy], mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([shoe, extruded_toy], mesh_show_back_face=True)

    # Export to stl and ply
    # combined = shoe + extruded_toy
    # combined.compute_vertex_normals()
    # o3d.io.write_triangle_mesh("output.stl", combined)