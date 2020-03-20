import open3d as o3d
import numpy as np
import PIL.Image
import IPython.display


def jupyter_draw_geometries(
        geoms,
        window_name="Open3D",
        width=1920,
        height=1080,
        left=50,
        top=50,
        point_show_normal=False,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name,
                      width=width,
                      height=height,
                      left=left,
                      top=top)
    vis.get_render_option().point_show_normal = point_show_normal
    for geom in geoms:
        vis.add_geometry(geom)
    vis.run()
    im = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im = (255 * np.asarray(im)).astype(np.uint8)
    IPython.display.display(PIL.Image.fromarray(im, "RGB"))


o3d.visualization.draw_geometries = jupyter_draw_geometries
