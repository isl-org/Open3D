from initialize import *
from examples.python.Dayong.surface_fitting.align import *
from project import *
from extrude import *
from utils import *
from pathlib import Path

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "Toys", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "Toys", "badge.stl")

    # Generate mesh for each stl
    shoe = read_stl_to_mesh(shoe_path)
    toy = read_stl_to_mesh(toy_path)
    shoe.paint_uniform_color([0.6, 0.8, 1.0]) # light blue
    toy.paint_uniform_color([1.0, 0.6, 0.3]) # orange
    toy = toy.scale(2.0, center=toy.get_center()) # scale to 2x size; optional
    o3d.visualization.draw_geometries([shoe, toy], mesh_show_back_face=True)

    # Get o and its normal
    o = select_o_point_from_ratio(shoe, 0.5, 0.2, 0.9) # front
    # o = select_o_point_from_ratio(shoe, 0.5, 0.9, 0.5) # back
    visualize_point_on_mesh(shoe, o, 1.0)
    normal_o = get_normal_at_point(shoe, o)

    # Get p and its normal
    p = select_o_point_from_ratio(toy, 0.5, 0.5, 0) # badge
    # p = select_o_point_from_ratio(toy, 0.5, 1, 0.5) # fish
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

    # Extrude
    thickness = 1
    extruded_toy = extrude_shell(toy, thickness)
    extruded_toy.paint_uniform_color([1.0, 0.6, 0.3])

    o3d.visualization.draw_geometries([extruded_toy], mesh_show_back_face=True)
    o3d.visualization.draw_geometries([shoe, extruded_toy], mesh_show_back_face=True)

    # # Export to stl and ply
    # combined = shoe + extruded_toy
    # combined.compute_vertex_normals()
    # o3d.io.write_triangle_mesh("output.stl", combined)


    # # === PREVIEW CUT RESULT (before projection / warp) ===
    # cut_dir = -normal_o
    # band_ratio = 0.02  # try 0.02~0.05 if slice looks too thin
    #
    # cut_mesh, cut_dbg = cut_preview_mesh(
    #     toy,
    #     cut_dir=cut_dir,
    #     keep="bottom",
    #     band_ratio=band_ratio
    # )
    #
    # cut_mesh.paint_uniform_color([1.0, 0.8, 0.2])  # yellow
    # cut_slice_pcd = make_debug_pcd(
    #     cut_dbg["slice_pts"],
    #     color=(1.0, 0.0, 0.0)  # red slice band
    # )
    #
    # o3d.visualization.draw_geometries(
    #     [shoe, cut_mesh, cut_slice_pcd],
    #     mesh_show_back_face=True
    # )
    #
    # # === WARP (closest_points on slice) ===
    # k = 0.1  # 先用 0.05~0.2 之间试
    # band_ratio = 0.02  # 你 preview 用多少，这里保持一致
    # keep = "bottom"  # 必须和 preview 一致
    # cut_dir = -normal_o
    #
    # warped_toy, dbg = warp_top_by_projected_slice_closest(
    #     toy=toy,
    #     shoe=shoe,
    #     cut_dir=cut_dir,
    #     move_dir=-normal_o,  # ✅ 关键
    #     keep="bottom",
    #     band_ratio=band_ratio,
    #     k=k,
    # )
    #
    # warped_toy.paint_uniform_color([1.0, 0.6, 0.3])  # orange
    #
    # slice_pcd = make_debug_pcd(dbg["slice_pts"], color=(1.0, 0.0, 0.0))  # red
    # proj_pcd = make_debug_pcd(dbg["projected_slice"], color=(0.0, 1.0, 0.0))  # green
    #
    # o3d.visualization.draw_geometries(
    #     [shoe, warped_toy, slice_pcd, proj_pcd],
    #     mesh_show_back_face=True
    # )



