import open3d as o3d
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
        # TODO: find a better solution. We need to redirect
        # C++ prints to terminal when async loop + GIL + py::print
        # are used together under some scenarios.
        o3d.utility.reset_print_function()
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
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        done = False
        while not done:
            with self._lock:
                for task in self._run_queue:
                    retval = task.func()
                    self._return_vals[task.task_id] = retval
                self._run_queue.clear()

            done = not app.run_one_tick()
