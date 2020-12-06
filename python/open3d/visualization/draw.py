from . import gui
from . import O3DVisualizer


def draw(
        geometry=None,
        title="Open3D",
        width=1024,
        height=768,
        actions=None,
        #lookat=None,
        #eye=None,
        #up=None,
        #field_of_view=None,
        show_ui=None,
        point_size=None,
        rpc_interface=False):
    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if point_size is not None:
        w.point_size = point_size

    def add(g, n):
        if isinstance(g, dict):
            w.add_geometry(g)
        else:
            w.add_geometry("Object " + str(n), g)

    n = 1
    if isinstance(geometry, list):
        for g in geometry:
            add(g, n)
            n += 1
    elif geometry is not None:
        add(geometry, n)

    w.reset_camera_to_default()

    if show_ui is not None:
        w.show_settings = show_ui

    if rpc_interface:
        w.start_rpc_interface(address="tcp://127.0.0.1:51454", timeout=10000)

    gui.Application.instance.add_window(w)
    gui.Application.instance.run()

    if rpc_interface:
        w.stop_rpc_interface()
