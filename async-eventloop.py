import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading


class AsyncEventLoop:

    class _Task:
        g_next_id = 1

        def __init__(self, f):
            self.task_id = self.g_next_id
            self.func = f
            self.g_next_id += 1

    # Do not call this directly, use instance instead
    def __init__(self, use_native=False):
        assert (AsyncEventLoop.instance is None)

        self._using_native = use_native

        self._lock = threading.Lock()
        self._finished_cv = threading.Condition(self._lock)
        self._run_queue = []
        self._return_vals = {}
        self._thread_finished = False

        if use_native:
            gui.Application.instance.initialize()

    instance = None

    def start(self):
        if self._using_native:
            self._thread = None
            self._thread_main()
        else:
            self._thread = threading.Thread(target=self._thread_main)
            self._thread.start()

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

        with self._lock:
            self._thread_finished = True
            self._finished_cv.notify_all()

    def run_sync(self, f):
        with self._lock:
            assert (not self._thread_finished)
            task = self._Task(f)
            self._run_queue.append(task)

        while True:
            with self._lock:
                if task.task_id in self._return_vals:
                    return self._return_vals[task.task_id]

    def app_run(self):
        pass
        # self._lock.acquire()
        # self._finished_cv.wait()
        # self._lock.release()


def monkey_patch_class(eloop, clazz):
    if clazz.__class__.__name__.startswith("__"):
        return
    to_patch = dir(clazz)
    for attr in to_patch:
        if attr.startswith("__") and attr != "__init__":
            continue
        f = getattr(clazz, attr)
        # TODO: need to proxy properties, too
        #        if callable(f):
        #            def patch(func):
        #                #signature = inspect.signature(func)
        #                #default_kwargs = {
        #                #    k: v.default
        #                #    for k, v in signature.parameters.items()
        #                #    if v.default is not inspect.Parameter.empty
        #                #}
        #                default_kwargs = {}  # can't get this from pybind
        #                def patched(*args, **kwargs):
        #                    merged_kwargs = kwargs
        #                    for k, v in default_kwargs.items():
        #                        if k not in merged_kwargs:
        #                            merged_kwargs[k] = v
        #
        #                    def call():
        #                        return func(args, merged_kwargs)
        #                    return eloop.run_sync(call)
        #                return patched
        #            try:
        #                setattr(clazz, attr, patch(f))
        #            except AttributeError as e:
        #                pass
        if callable(f):
            # pybind can't use inspect, but it does put the call signature
            # in the docstring
            docstr = getattr(f, "__doc__")
            if docstr is None:
                continue
            # Docstring signature always ends in \n, even if there is no
            # docstring specified in the pybind call.
            sigstr = docstr.split("\n")[0]
            arg_start_idx = docstr.find("(") + 1
            arg_end_idx = docstr.find(")") - 1
            argstr = docstr[arg_start_idx:arg_end_idx - arg_start_idx]
            if argstr != "":
                n_args = argstr.count(",") + 1
            else:
                n_args = 0
            try:
                setattr(clazz, attr, patch(eloop, f, n_args, clazz.__name__))
            except AttributeError as e:
                pass


def patch(eloop, f, n_args, clazzname):
    debug = clazzname + "." + f.__name__
    if n_args == 0:
        return patch_0_arg(eloop, f, debug)
    elif n_args == 1:
        return patch_1_arg(eloop, f, debug)
    elif n_args == 2:
        return patch_2_arg(eloop, f, debug)
    elif n_args == 3:
        return patch_3_arg(eloop, f, debug)
    elif n_args == 4:
        return patch_4_arg(eloop, f, debug)
    elif n_args == 5:
        return patch_5_arg(eloop, f, debug)
    elif n_args == 6:
        return patch_6_arg(eloop, f, debug)
    elif n_args == 7:
        return patch_7_arg(eloop, f, debug)
    elif n_args == 8:
        return patch_8_arg(eloop, f, debug)
    elif n_args == 12:
        return patch_12_arg(eloop, f, debug)
    else:
        print("[error] monkey patching for", n_args,
              "argument functions not implemented")
        print("[error]     ", clazzname + "." + f.__name__)
        return None


def patch_0_arg(eloop, f, debug):

    def patched():

        def call():
            print("[debug]", debug)
            return f()

        return eloop.run_sync(call)

    return patched


def patch_1_arg(eloop, f, debug):

    def patched(arg1):

        def call():
            print("[debug]", debug)
            return f(arg1)

        return eloop.run_sync(call)

    return patched


def patch_2_arg(eloop, f, debug):

    def patched(arg1, arg2):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2)

        return eloop.run_sync(call)

    return patched


def patch_3_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3)

        return eloop.run_sync(call)

    return patched


def patch_4_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4)

        return eloop.run_sync(call)

    return patched


def patch_5_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4, arg5):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4, arg5)

        return eloop.run_sync(call)

    return patched


def patch_6_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4, arg5, arg6):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4, arg5, arg6)

        return eloop.run_sync(call)

    return patched


def patch_7_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4, arg5, arg6, arg7):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4, arg5, arg6, arg7)

        return eloop.run_sync(call)

    return patched


def patch_8_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)

        return eloop.run_sync(call)

    return patched


def patch_12_arg(eloop, f, debug):

    def patched(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                arg11, arg12):

        def call():
            print("[debug]", debug)
            return f(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                     arg10, arg11, arg12)

        return eloop.run_sync(call)

    return patched


def gui_start_async(f, use_native=False):
    # We need some of the original functions
    app_run_one_tick = o3d.visualization.gui.Application.run_one_tick
    app_initialize = o3d.visualization.gui.Application.initialize

    AsyncEventLoop.instance = AsyncEventLoop(use_native)
    eloop = AsyncEventLoop.instance

    # debugging
    app_create_window = o3d.visualization.gui.Application.create_window

    def patched_create_window(app,
                              title="",
                              width=-1,
                              height=-1,
                              x=-1,
                              y=-1,
                              flags=0):

        def call():
            return app_create_window(app, title, width, height, x, y, flags)

        return eloop.run_sync(call)

    app_add_window = o3d.visualization.gui.Application.add_window

    def patched_add_window(app, w):

        def call():
            return app_add_window(app, w)

        return eloop.run_sync(call)

    o3dvis_init = o3d.visualization.O3DVisualizer.__init__

    def patched_o3dvis_init(o3dvis, title, width, height):

        def call():
            return o3dvis_init(o3dvis, title, width, height)

        return eloop.run_sync(call)

    o3dvis_add_geom = o3d.visualization.O3DVisualizer.add_geometry

    def patched_o3dvis_add_geom(o3dvis,
                                name,
                                geom,
                                mat=None,
                                group="",
                                time=0.0,
                                is_visible=True):

        def call():
            return o3dvis_add_geom(o3dvis, name, geom, mat, group, time,
                                   is_visible)

        return eloop.run_sync(call)

    o3dvis_reset_camera = o3d.visualization.O3DVisualizer.reset_camera_to_default

    def patched_o3dvis_reset_camera(o3dvis):

        def call():
            return o3dvis_reset_camera(o3dvis)

        return eloop.run_sync(call)

    mat_init = rendering.Material.__init__

    for attr in dir(o3d.visualization.rendering):
        if attr.startswith("__"):
            continue
        monkey_patch_class(eloop, getattr(o3d.visualization.rendering, attr))
    for attr in dir(o3d.visualization.gui):
        if attr.startswith("__"):
            continue
        monkey_patch_class(eloop, getattr(o3d.visualization.gui, attr))
    monkey_patch_class(eloop, o3d.visualization.O3DVisualizer)

    # Special overrides
    o3d.visualization.gui.Application.initialize = app_initialize
    o3d.visualization.gui.Application.run_one_tick = app_run_one_tick
    o3d.visualization.gui.Application.run = eloop.app_run

    # debugging
    o3d.visualization.gui.Application.create_window = patched_create_window
    #    o3d.visualization.gui.Application.add_window = patched_add_window
    #    o3d.visualization.O3DVisualizer.__init__ = patched_o3dvis_init
    o3d.visualization.O3DVisualizer.add_geometry = patched_o3dvis_add_geom
    #    o3d.visualization.O3DVisualizer.reset_camera_to_default = patched_o3dvis_reset_camera
    #    rendering.Material.__init__ = mat_init # Material doesn't use Filament

    if use_native:
        thread = threading.Thread(target=f)
        thread.start()
        eloop.start()  # blocks if use_native=True
        thread.join()
    else:
        eloop.start()  # returns
        f()


#-------------------------------------------------------------------------------
def main():
    app = gui.Application.instance
    app.initialize()
    app.run()

    torus = o3d.geometry.TriangleMesh.create_torus()
    torus.compute_vertex_normals()
    mat = rendering.Material()
    mat.shader = "defaultLit"

    w = o3d.visualization.O3DVisualizer("Open3D", 640, 480)
    w.add_geometry("Torus", torus, mat)
    w.reset_camera_to_default()
    app.add_window(w)

    print("hello")


if __name__ == "__main__":
    gui_start_async(main, use_native=False)
