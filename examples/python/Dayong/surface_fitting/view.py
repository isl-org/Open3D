import trimesh

from examples.python.Dayong.surface_fitting.extrude import *
from examples.python.Dayong.surface_fitting.utils import *


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    # final_stl_path = os.path.join(base_dir, "warp.stl")
    # shoe = read_stl_to_mesh(final_stl_path)
    # shoe.paint_uniform_color([0.6, 0.8, 1.0])  # light blue
    # o3d.visualization.draw_geometries([shoe], mesh_show_back_face=True)

    def o3d_view_glb_via_trimesh(glb_path):
        scene_or_mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(scene_or_mesh, trimesh.Scene):
            # 如果是 Scene，合并
            scene_or_mesh = trimesh.util.concatenate(
                [g for g in scene_or_mesh.geometry.values()]
            )

        tm = scene_or_mesh

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))

        # 顶点颜色（如果存在）
        if tm.visual is not None and hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None:
            vc = np.asarray(tm.visual.vertex_colors)
            if vc.shape[1] >= 3:
                colors = (vc[:, :3].astype(np.float64) / 255.0)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries(
            [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)],
            window_name=f"GLB Viewer: {glb_path}",
            width=1400,
            height=900,
            mesh_show_back_face=True
        )


    o3d_view_glb_via_trimesh("poisson_filled.glb")
