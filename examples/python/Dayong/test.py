import open3d as o3d
import numpy as np
import threading
import time


class DynamicPointCloudApp:
    GEOM_NAME = "Geometry"

    def __init__(self, update_delay=-1):
        self.update_delay = update_delay
        self.is_done = False
        self.lock = threading.Lock()

        self.app = o3d.visualization.gui.Application.instance
        self.window = self.app.create_window("Open3D Python App", width=800, height=600, x=0, y=30)
        self.window.set_on_close(self.on_main_window_closing)
        if self.update_delay < 0:
            self.window.set_on_tick_event(self.on_main_window_tick_event)

        self.widget = o3d.visualization.gui.SceneWidget()
        self.widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.window.add_child(self.widget)

        self.geom_pcd = DynamicPointCloudApp.generate_point_cloud()
        self.geom_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(self.geom_pcd.points))

        self.geom_mat = o3d.visualization.rendering.MaterialRecord()
        self.geom_mat.shader = 'defaultUnlit'
        self.geom_mat.point_size = 2.0

        self.widget.scene.add_geometry(self.GEOM_NAME, self.geom_pcd, self.geom_mat)

        self.widget.setup_camera(60, self.widget.scene.bounding_box, [0, 0, 0])

    def generate_point_cloud():
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    def update_point_cloud(self):
        index_to_replace = np.random.randint(0, 100)
        rand_pt = np.random.rand(1, 3)
        self.geom_pcd.points[index_to_replace] = rand_pt[0]
        self.widget.enable_scene_caching(False)
        self.widget.scene.remove_geometry(self.GEOM_NAME)
        self.widget.scene.add_geometry(self.GEOM_NAME, self.geom_pcd, self.geom_mat)
        return True

    def startThread(self):
        if self.update_delay >= 0:
            threading.Thread(target=self.update_thread).start()

    def update_thread(self):
        def do_update():
            return self.update_point_cloud()

        while not self.is_done:
            time.sleep(self.update_delay)
            print("update_thread")
            with self.lock:
                if self.is_done:  # might have changed while sleeping.
                    break
                o3d.visualization.gui.Application.instance.post_to_main_thread(self.window, self.update_point_cloud)

    def on_main_window_closing(self):
        with self.lock:
            self.is_done = True
        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        print("tick")
        return self.update_point_cloud()


def main():
    o3d.visualization.gui.Application.instance.initialize()

    thread_delay = 0.1
    use_tick = -1

    dpcApp = DynamicPointCloudApp(use_tick)
    dpcApp.startThread()

    o3d.visualization.gui.Application.instance.run()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()