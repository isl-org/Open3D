#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.rendering as rendering


def main():
    render = rendering.OffscreenRenderer(640,
                                         480,
                                         resource_path="",
                                         headless=True)

    grey = rendering.Material()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])

    render.scene.add_geometry("box", box, grey)
    render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                     75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    o3d.io.write_image("test.png", img, 9)


if __name__ == "__main__":
    main()
