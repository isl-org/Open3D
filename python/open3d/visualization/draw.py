# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from . import gui
from . import O3DVisualizer


def draw(geometry=None,
         title="Open3D",
         width=1024,
         height=768,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         intrinsic_matrix=None,
         extrinsic_matrix=None,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         ibl=None,
         ibl_intensity=None,
         show_skybox=None,
         show_ui=None,
         raw_mode=False,
         point_size=None,
         line_width=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None,
         non_blocking_and_return_uid=False):
    """Draw 3D geometry types and 3D models. This is a high level interface to
    :class:`open3d.visualization.O3DVisualizer`.

    The initial view may be specified either as a combination of (lookat, eye,
    up, and field of view) or (intrinsic matrix, extrinsic matrix) pair. A
    simple pinhole camera model is used.

    Args:
        geometry (List[Geometry] or List[Dict]): The 3D data to be displayed can be provided in different types:
            - A list of any Open3D geometry types (``PointCloud``, ``TriangleMesh``, ``LineSet`` or ``TriangleMeshModel``).
            - A list of dictionaries with geometry data and additional metadata. The following keys are used:
                - **name** (str): Geometry name.
                - **geometry** (Geometry): Open3D geometry to be drawn.
                - **material** (:class:`open3d.visualization.rendering.MaterialRecord`): PBR material for the geometry.
                - **group** (str): Assign the geometry to a group. Groups are shown in the settings panel and users can take take joint actions on a group as a whole.
                - **time** (float): If geometry elements are assigned times, a time bar is displayed and the elements can be animated.
                - **is_visible** (bool): Show this geometry?
        title (str): Window title.
        width (int): Viewport width.
        height (int): Viewport height.
        actions (List[(str, Callable)]): A list of pairs of action names and the
            corresponding functions to execute. These actions are presented as
            buttons in the settings panel. Each callable receives the window
            (``O3DVisualizer``) as an argument.
        lookat (array of shape (3,)): Camera principal axis direction.
        eye (array of shape (3,)): Camera location.
        up (array of shape (3,)): Camera up direction.
        field_of_view (float): Camera horizontal field of view (degrees).
        intrinsic_matrix (array of shape (3,3)): Camera intrinsic matrix.
        extrinsic_matrix (array of shape (4,4)): Camera extrinsic matrix (world
            to camera transformation).
        bg_color (array of shape (4,)): Background color float with range [0,1],
            default white.
        bg_image (open3d.geometry.Image): Background image.
        ibl (open3d.geometry.Image): Environment map for image based lighting
            (IBL).
        ibl_intensity (float): IBL intensity.
        show_skybox (bool): Show skybox as scene background (default False).
        show_ui (bool): Show settings user interface (default False). This can
            be toggled from the Actions menu.
        raw_mode (bool): Use raw mode for simpler rendering of the basic
            geometry (Default false).
        point_size (int): 3D point size (default 3).
        line_width (int): 3D line width (default 1).
        animation_time_step (float): Duration in seconds for each animation
            frame.
        animation_duration (float): Total animation duration in seconds.
        rpc_interface (bool): Start an RPC interface at http://localhost:51454 and
            listen for drawing requests. The requests can be made with
            :class:`open3d.visualization.ExternalVisualizer`.
        on_init (Callable): Extra initialization procedure for the underlying
            GUI window. The procedure receives a single argument of type
            :class:`open3d.visualization.O3DVisualizer`.
        on_animation_frame (Callable): Callback for each animation frame update
            with signature::

                Callback(O3DVisualizer, double time) -> None

        on_animation_tick (Callable): Callback for each animation time step with
            signature::

                Callback(O3DVisualizer, double tick_duration, double time) -> TickResult

            If the callback returns ``TickResult.REDRAW``, the scene is redrawn.
            It should return ``TickResult.NOCHANGE`` if redraw is not required.
        non_blocking_and_return_uid (bool): Do not block waiting for the user
            to close the window. Instead return the window ID. This is useful
            for embedding the visualizer and is used in the WebRTC interface and
            Tensorboard plugin.

    Example:
        See `examples/visualization/draw.py` for examples of advanced usage. The ``actions()``
        example from that file is shown below::

            import open3d as o3d
            import open3d.visualization as vis

            SOURCE_NAME = "Source"
            RESULT_NAME = "Result (Poisson reconstruction)"
            TRUTH_NAME = "Ground truth"

            bunny = o3d.data.BunnyMesh()
            bunny_mesh = o3d.io.read_triangle_mesh(bunny.path)
            bunny_mesh.compute_vertex_normals()

            bunny_mesh.paint_uniform_color((1, 0.75, 0))
            bunny_mesh.compute_vertex_normals()
            cloud = o3d.geometry.PointCloud()
            cloud.points = bunny_mesh.vertices
            cloud.normals = bunny_mesh.vertex_normals

            def make_mesh(o3dvis):
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    cloud)
                mesh.paint_uniform_color((1, 1, 1))
                mesh.compute_vertex_normals()
                o3dvis.add_geometry({"name": RESULT_NAME, "geometry": mesh})
                o3dvis.show_geometry(SOURCE_NAME, False)

            def toggle_result(o3dvis):
                truth_vis = o3dvis.get_geometry(TRUTH_NAME).is_visible
                o3dvis.show_geometry(TRUTH_NAME, not truth_vis)
                o3dvis.show_geometry(RESULT_NAME, truth_vis)

            vis.draw([{
                "name": SOURCE_NAME,
                "geometry": cloud
            }, {
                "name": TRUTH_NAME,
                "geometry": bunny_mesh,
                "is_visible": False
            }],
                 actions=[("Create Mesh", make_mesh),
                          ("Toggle truth/result", toggle_result)])
    """
    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if point_size is not None:
        w.point_size = point_size

    if line_width is not None:
        w.line_width = line_width

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

    w.reset_camera_to_default()  # make sure far/near get setup nicely
    if lookat is not None and eye is not None and up is not None:
        w.setup_camera(field_of_view, lookat, eye, up)
    elif intrinsic_matrix is not None and extrinsic_matrix is not None:
        w.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)

    w.animation_time_step = animation_time_step
    if animation_duration is not None:
        w.animation_duration = animation_duration

    if show_ui is not None:
        w.show_settings = show_ui

    if ibl is not None:
        w.set_ibl(ibl)

    if ibl_intensity is not None:
        w.set_ibl_intensity(ibl_intensity)

    if show_skybox is not None:
        w.show_skybox(show_skybox)

    if rpc_interface:
        w.start_rpc_interface(address="tcp://127.0.0.1:51454", timeout=10000)

        def stop_rpc():
            w.stop_rpc_interface()
            return True

        w.set_on_close(stop_rpc)

    if raw_mode:
        w.enable_raw_mode(True)

    if on_init is not None:
        on_init(w)
    if on_animation_frame is not None:
        w.set_on_animation_frame(on_animation_frame)
    if on_animation_tick is not None:
        w.set_on_animation_tick(on_animation_tick)

    gui.Application.instance.add_window(w)
    if non_blocking_and_return_uid:
        return w.uid
    else:
        gui.Application.instance.run()
