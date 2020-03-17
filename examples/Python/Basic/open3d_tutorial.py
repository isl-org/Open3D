import open3d as o3d
import numpy as np
import PIL.Image
import IPython.display


def jupyter_draw_geometries(geoms):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geoms:
        vis.add_geometry(geom)
    vis.run()
    im = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im = (255 * np.asarray(im)).astype(np.uint8)
    IPython.display.display(PIL.Image.fromarray(im, "RGB"))


o3d.visualization.draw_geometries = jupyter_draw_geometries
