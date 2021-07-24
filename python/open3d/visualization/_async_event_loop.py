"""Run the GUI event loop in a non-main thread. This allows using the
GUI from plugins to other apps (e.g.: Jupyter or Tensorboard) where the GUI
cannot be started in the main thread. Currently does not work in macOS.

.. note:: This is a singleton class implemented with this module as a
   holder. The ``_async_event_loop`` singleton is started whenever this
   module is imported.  If you are using remote visualization with WebRTC,
   you must call ``enable_webrtc()`` before importing this module.
"""

import threading
import functools
import open3d as o3d


class _AsyncEventLoop:

    class _Task:
        _g_next_id = 0

        def __init__(self, f):
            self.task_id = self._g_next_id
            self.func = f
            _AsyncEventLoop._Task._g_next_id += 1

    def __init__(self):
        # TODO (yixing): find a better solution. Currently py::print acquires
        # GIL which causes deadlock when AsyncEventLoop is used. By calling
        # reset_print_function(), all C++ prints will be directed to the
        # terminal while python print will still remain in the cell.
        o3d.utility.reset_print_function()
        self._lock = threading.Lock()
        self._run_queue = []
        self._return_vals = {}
        self._started = False
        self._start()

    def _start(self):
        if not self._started:
            self._thread = threading.Thread(name="GUIMain",
                                            target=self._thread_main)
            self._thread.start()
            self._started = True

    def run_sync(self, f):
        with self._lock:
            task = _AsyncEventLoop._Task(f)
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


# The _AsyncEventLoop class shall only be used to create a singleton instance.
# There are different ways to achieve this, here we use the module as a holder
# for singleton variables, see: https://stackoverflow.com/a/31887/1255535.
_async_event_loop = _AsyncEventLoop()
