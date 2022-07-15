# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2022 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform

isMacOS = (platform.system() == "Darwin")

class SceneGraphApp:
    MENU_SPHERE = 1
    MENU_TIME = 2
    MENU_UPDATE_TIME = 3
    MENU_QUIT = 4

    def __init__(self):
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "Scenegraph Example", 1024, 768)
        self.scene = gui.SceneWidget()

        # Create Scenegraph and add it to SceneWidget
        self.sg = rendering.Open3DScenegraph(self.window.renderer)
        self.scene.scene = sg

        # Setup global scene properties
        self.sg.set_background([1, 1, 1, 1])
        self.sg.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.sg.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.sg.setup_global_camera(60, bbox, [0, 0, 0])

        self.window.add_child(self.scene)

        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Quit", SpheresApp.MENU_QUIT)
            debug_menu = gui.Menu()
            debug_menu.add_item("Add Sphere", SpheresApp.MENU_SPHERE)
            debug_menu.add_item("Add Timed Sphere", SpheresApp.MENU_TIME)
            debug_menu.add_item("Update Time", SpheresApp.MENU_UPDATE_TIME)
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
        self.window.set_on_menu_item_activated(SpheresApp.MENU_TIME,
                                               self._on_menu_timed_sphere)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_UPDATE_TIME,
                                               self._on_menu_update_time)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_QUIT,
                                               self._on_menu_quit)

    def create_sphere_item(self):
        self._id += 1
        mat = rendering.MaterialRecord()
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
        # Create a Drawable SceneItem for the Scenegraph
        sphere_name = "sphere" + str(self._id)
        sphere_item = self.sg.create_drawable_item(sphere_name, sphere, mat)
        return (sphere_name, sphere_item)


    def _on_menu_sphere(self):
        # GUI callbacks happen on the main thread, so we can do everything
        # normally here.
        sn, si = self.create_sphere_item()
        # Add sphere to scene
        scene.sg.set_item("/spheres/" + sn, si)


    def _on_menu_timed_sphere(self):
        # create 5 spheres at different times
        for _ in range(5):
            s = self.create_sphere()
            sphere_name = 


    def _on_menu_udpate_time(self):
        # Add one second to the current scene time
        self.sg.current_time = self.sg.current_time + 1.0


    def _on_menu_quit(self):
        gui.Application.instance.quit()


def main():
    gui.Application.instance.initialize()
    SceneGraphApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
