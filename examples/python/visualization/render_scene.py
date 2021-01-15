# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/visualization/render_scene.py

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def main():
    render = rendering.Renderer(640, 480)

    mat = rendering.Material()
    mat.base_color = [1.0, 0.75, 0.0, 1.0]
    mat.shader = "defaultLit"

    # Set the properties of a shape (sphere).
    sphere = o3d.geometry.TriangleMesh.create_sphere(.3)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    # Add the sphere to the scene.
    render.scene.add_geometry("sphere", sphere, yellow)

    # Set up the camera
    render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.setup_camera(60, sphere, [0, 0, 0])

    # Set up the lighting
    render.scene.scene.set_directional_light([0.707, 0.0, -.707],
                                             [1.0, 1.0, 1.0], 75000)
    render.scene.scene.enable_directional_light(True)
    render.scene.show_axes(True)

    # Intialize the window
    self.window = gui.Application.instance.create_window(
        "Scene Rendering", 1024, 768)

    # Add the scene widget to the GUI
    self.scene = gui.SceneWidget()
    self.scene.scene = rendering.Open3DScene(render)


if __name__ == "__main__":
    main()
