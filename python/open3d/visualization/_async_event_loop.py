# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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
# ----------------------------------------------------------------------------
"""Run the GUI event loop in a non-main thread. This allows using the
GUI from plugins to other apps (e.g.: Jupyter or Tensorboard) where the GUI
cannot be started in the main thread. Currently does not work in macOS.

.. note:: This is a singleton class implemented with this module as a
   holder. The ``_async_event_loop`` singleton is started whenever this
   module is imported.  If you are using remote visualization with WebRTC,
   you must call ``enable_webrtc()`` before importing this module.
"""

import threading
from collections import deque
import open3d as o3d


class _AsyncEventLoop:

    class _Task:
        _g_next_id = 0

        def __init__(self, func):
            self.task_id = self._g_next_id
            self.func = func
            _AsyncEventLoop._Task._g_next_id += 1

    def __init__(self):
        # TODO (yixing): find a better solution. Currently py::print acquires
        # GIL which causes deadlock when AsyncEventLoop is used. By calling
        # reset_print_function(), all C++ prints will be directed to the
        # terminal while python print will still remain in the cell.
        o3d.utility.reset_print_function()
        self._lock = threading.Lock()
        self._run_queue = deque()
        self._return_vals = {}
        self._started = False
        self._start()

    def _start(self):
        if not self._started:
            self._thread = threading.Thread(name="GUIMain",
                                            target=self._thread_main)
            self._thread.start()
            self._started = True

    def run_sync(self, func):
        """Enqueue task, wait for completion and return result. Can run in any
        thread."""
        with self._lock:
            task = _AsyncEventLoop._Task(func)
            self._run_queue.append(task)

        while True:
            with self._lock:
                if task.task_id in self._return_vals:
                    return self._return_vals.pop(task.task_id)

    def _thread_main(self):
        """Main GUI thread event loop"""
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        done = False
        while not done:
            while len(self._run_queue) > 0:
                with self._lock:
                    task = self._run_queue.popleft()
                # Release lock while running task. This allows a task to add
                # more async tasks.
                retval = task.func()
                with self._lock:
                    self._return_vals[task.task_id] = retval

            done = not app.run_one_tick()


# The _AsyncEventLoop class shall only be used to create a singleton instance.
# There are different ways to achieve this, here we use the module as a holder
# for singleton variables, see: https://stackoverflow.com/a/31887/1255535.
_async_event_loop = _AsyncEventLoop()
