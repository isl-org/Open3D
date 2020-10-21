#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform
import random
import threading
import time

isMacOS = (platform.system() == "Darwin")


# This example shows two methods of adding geometry to an existing scene.
# 1) add via a UI callback (in this case a menu, but a button would be similar,
#    you would call `button.set_on_clicked(self.on_menu_sphere_)` when
#    configuring the button. See `on_menu_sphere()`.
# 2) add asynchronously by polling from another thread. GUI functions must be
#    called from the UI thread, so use Application.post_to_main_thread().
#    See `on_menu_random()`.
# Running the example will show a simple window with a Debug menu item with the
# two different options. Two second method will add random spheres for
# 20 seconds, during which time you can be interacting with the scene, rotating,
# etc.
class SpheresApp:
    MENU_SPHERE = 1
    MENU_RANDOM = 2
    MENU_QUIT = 3

    def __init__(self):
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "Add Spheres Example", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background_color([1, 1, 1, 1])
        self.scene.scene.scene.set_directional_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_directional_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.scene.setup_camera(60, bbox, [0, 0, 0])

        self.window.add_child(self.scene)

        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Quit", SpheresApp.MENU_QUIT)
            debug_menu = gui.Menu()
            debug_menu.add_item("Add Sphere", SpheresApp.MENU_SPHERE)
            debug_menu.add_item("Add Random Spheres", SpheresApp.MENU_RANDOM)
            if not isMacOS:
                debug_menu.add_separator()
                debug_menu.add_item("Quit", SpheresApp.MENU_QUIT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("Debug", debug_menu)
            else:
                menu.add_menu("Debug", debug_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(SpheresApp.MENU_SPHERE,
                                               self._on_menu_sphere)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_RANDOM,
                                               self._on_menu_random)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_QUIT,
                                               self._on_menu_quit)

    def add_sphere(self):
        self._id += 1
        mat = rendering.Material()
        mat.base_color = [
            random.random(),
            random.random(),
            random.random(), 1.0
        ]
        mat.shader = "defaultLit"
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.5)
        sphere.compute_vertex_normals()
        sphere.translate([
            10.0 * random.uniform(-1.0, 1.0), 10.0 * random.uniform(-1.0, 1.0),
            10.0 * random.uniform(-1.0, 1.0)
        ])
        self.scene.scene.add_geometry("sphere" + str(self._id), sphere, mat)

    def _on_menu_sphere(self):
        # GUI callbacks happen on the main thread, so we can do everything
        # normally here.
        self.add_sphere()

    def _on_menu_random(self):
        # This adds spheres asynchronously. This pattern is useful if you have
        # data coming in from another source than user interaction.
        def thread_main():
            for _ in range(0, 20):
                # We can only modify GUI objects on the main thread, so we
                # need to post the function to call to the main thread.
                gui.Application.instance.post_to_main_thread(
                    self.window, self.add_sphere)
                time.sleep(1)

        threading.Thread(target=thread_main).start()

    def _on_menu_quit(self):
        gui.Application.instance.quit()


def main():
    gui.Application.instance.initialize()
    SpheresApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
