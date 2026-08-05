# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.visualization.rendering as rendering


def main():
    render = rendering.OffscreenRenderer(640, 480)

    yellow = rendering.MaterialRecord()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"

    green = rendering.MaterialRecord()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"

    grey = rendering.MaterialRecord()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"

    white = rendering.MaterialRecord()
    white.base_color = [1.0, 1.0, 1.0, 1.0]
    white.shader = "defaultLit"

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    render.scene.add_geometry("cyl", cyl, green)
    render.scene.add_geometry("sphere", sphere, yellow)
    render.scene.add_geometry("box", box, grey)
    render.scene.add_geometry("solid", solid, white)
    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                     75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    print("Saving image at test.png")
    o3d.io.write_image("test.png", img, 9)

    render.setup_camera(60.0, [0, 0, 0], [-10, 0, 0], [0, 0, 1])
    img = render.render_to_image()
    print("Saving image at test2.png")
    o3d.io.write_image("test2.png", img, 9)


if __name__ == "__main__":
    main()
