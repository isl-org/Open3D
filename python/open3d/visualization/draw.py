from . import gui
from . import DrawVisualizer


def draw(geometry=None,
         title="Open3D",
         width=1024,
         height=768,
         actions=None,
         menu_actions=None,
         show_ui=None,
         point_size=None):
    gui.Application.instance.initialize()
    w = DrawVisualizer(title, width, height)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if menu_actions is not None:
        for a in menu_actions:
            w.add_menu_action(a[0], a[1])

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

    w.reset_camera()

    if show_ui is not None:
        w.show_settings = show_ui

    gui.Application.instance.add_window(w)
    gui.Application.instance.run()
