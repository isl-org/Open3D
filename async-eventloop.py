import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading


class AsyncEventLoop:

    class _Task:
        _g_next_id = 0

        def __init__(self, f):
            self.task_id = self._g_next_id
            self.func = f
            self._g_next_id += 1

    # Do not call this directly, use instance instead
    def __init__(self):
        self._lock = threading.Lock()
        self._run_queue = []
        self._return_vals = {}

    def start(self):
        self._thread = threading.Thread(target=self._thread_main)
        self._thread.start()

    def run_sync(self, f):
        with self._lock:
            task = self._Task(f)
            self._run_queue.append(task)

        while True:
            with self._lock:
                if task.task_id in self._return_vals:
                    return self._return_vals[task.task_id]

    def _thread_main(self):
        app = gui.Application.instance
        app.initialize()

        done = False
        while not done:
            with self._lock:
                for task in self._run_queue:
                    retval = task.func()
                    self._return_vals[task.task_id] = retval
                self._run_queue.clear()

            done = not app.run_one_tick()


def torus():
    app = gui.Application.instance

    torus = o3d.geometry.TriangleMesh.create_torus()
    torus.compute_vertex_normals()
    mat = rendering.Material()
    mat.shader = "defaultLit"

    w = o3d.visualization.O3DVisualizer("Open3D", 640, 480)
    w.add_geometry("Torus", torus, mat)
    w.reset_camera_to_default()
    app.add_window(w)


def box():
    app = gui.Application.instance

    torus = o3d.geometry.TriangleMesh.create_box()
    torus.compute_vertex_normals()
    mat = rendering.Material()
    mat.shader = "defaultLit"

    w = o3d.visualization.O3DVisualizer("Open3D", 640, 480)
    w.add_geometry("box", torus, mat)
    w.reset_camera_to_default()
    app.add_window(w)


if __name__ == "__main__":
    o3d.visualization.gui.Application.instance.enable_webrtc()
    eloop = AsyncEventLoop()
    eloop.start()

    eloop.run_sync(torus)
    eloop.run_sync(box)
