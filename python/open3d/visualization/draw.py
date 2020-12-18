from . import gui
from . import O3DVisualizer


def draw(
        geometry=None,
        title="Open3D",
        width=1024,
        height=768,
        background_color=(1.0, 1.0, 1.0, 1.0),
        actions=None,
        #lookat=None,
        #eye=None,
        #up=None,
        #field_of_view=None,
        bg_color=(1, 1, 1, 1),
        bg_image=None,
        show_ui=None,
        point_size=None,
        animation_time_step=1.0,
        animation_duration=None,
        rpc_interface=False,
        on_init=None,
        on_animation_frame=None,
        on_animation_tick=None):
    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(background_color, None)

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
    w.set_background(bg_color, bg_image)

    w.animation_time_step = animation_time_step
    if animation_duration is not None:
        w.animation_duration = animation_duration

    if show_ui is not None:
        w.show_settings = show_ui

    if rpc_interface:
        w.start_rpc_interface(address="tcp://127.0.0.1:51454", timeout=10000)

    if on_init is not None:
        on_init(w)
    if on_animation_frame is not None:
        w.set_on_animation_frame(on_animation_frame)
    if on_animation_tick is not None:
        w.set_on_animation_tick(on_animation_tick)

    gui.Application.instance.add_window(w)
    gui.Application.instance.run()

    if rpc_interface:
        w.stop_rpc_interface()
