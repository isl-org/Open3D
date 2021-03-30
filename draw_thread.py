import numpy as np
import open3d as o3d
import threading


def add_draw(geometries, window_name, width, height):
    vis = o3d.visualization.O3DVisualizer(title=window_name,
                                          width=width,
                                          height=height)
    count = 0
    for geometry in geometries:
        vis.add_geometry(f"geometry_{count}", geometry)
        count += 1
    vis.reset_camera_to_default()
    o3d.visualization.gui.Application.instance.add_window(vis)


def get_boxes():
    pc_rad = 1.0
    r = 0.4
    big_bbox = o3d.geometry.AxisAlignedBoundingBox((-pc_rad, -3, -pc_rad),
                                                   (6.0 + r, 1.0 + r, pc_rad))
    return [big_bbox]


def get_spheres():
    pc_rad = 1.0
    r = 0.4
    sphere_lit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_lit.compute_vertex_normals()
    sphere_lit.translate((4, 1, 0))
    sphere_colored_lit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_colored_lit.compute_vertex_normals()
    sphere_colored_lit.paint_uniform_color((0.0, 1.0, 0.0))
    sphere_colored_lit.translate((6, 1, 0))
    return [sphere_lit, sphere_colored_lit]


def add_draw(geometries, window_name, width, height):
    vis = o3d.visualization.O3DVisualizer(title=window_name,
                                          width=width,
                                          height=height)
    count = 0
    for geometry in geometries:
        vis.add_geometry(f"geometry_{count}", geometry)
        count += 1
    vis.reset_camera_to_default()
    o3d.visualization.gui.Application.instance.add_window(vis)


if __name__ == "__main__":
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    add_draw(get_boxes(), "Open3D empty_box", 640, 480)
    add_draw(get_spheres(), "Open3D multi_objects", 640, 480)

    # def workload():
    while (app.run_one_tick()):
        pass

    thread = threading.Thread(target=workload)
    thread.start()
    # app.run()
