#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.rendering as render
import open3d.visualization.gui as gui
import threading
import time

class RenderToImage:
    def __init__(self, width, height):
        gui.Application.instance.initialize()
        self._width = width
        self._height = height
        self._window = gui.Window("Open3D.RenderToImage", width, height, 0, 0)
        if self._window.content_rect.width > width:
            scaling = self._window.scaling
            self._window.size = gui.Size(int(round(width / scaling)),
                                         int(round(height / scaling)))
        self._image = None

    def set_background_color(self, color4):
        self._window.renderer.set_clear_color(color4)

    def create_scene(self):
        scene = render.Open3DScene(self._window.renderer)
        scene.set_view_size(self._width, self._height)
        return scene

    def render(self, open3dscene):
        self._image = None

        # Getting the clear color actually set in Filament seems to require
        # running a tick for everything to propagate through.
        gui.Application.instance.run_one_tick()

        open3dscene.scene.render_to_image(self._width, self._height,
                                          self._on_image)
        self._window.post_redraw()
        while self._image is None:
            gui.Application.instance.run_one_tick()

        img = self._image
        self._image = None
        return img

    def _on_image(self, image):
        self._image = image

def main():
    r = RenderToImage(1024, 768)
    scene = r.create_scene()
    r.set_background_color([1.0, 1.0, 1.0, 1.0])

    yellow = render.Material()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"

    green = render.Material()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"

    grey = render.Material()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"

    white = render.Material()
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

    scene.add_geometry("cyl", cyl, green)
    scene.add_geometry("sphere", sphere, yellow)
    scene.add_geometry("box", box, grey)
    scene.add_geometry("solid", solid, white)
    scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    scene.scene.set_directional_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                      75000)
    scene.scene.enable_directional_light(True)
    scene.show_axes(True)

    img = r.render(scene)
    o3d.io.write_image("/tmp/test.png", img, 9)

if __name__ == "__main__":
    main()
