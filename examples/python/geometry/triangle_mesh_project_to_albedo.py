import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering, O3DVisualizer
from open3d.core import Tensor


def create_dataset(meshfile):
    width, height = 1024, 768
    K = np.array([[512, 0, 512], [0, -512, 384], [0, 0, 1]])
    model = o3d.io.read_triangle_model(meshfile)
    t = np.array([0, 0, 0.5])  # origin / object in camera ref frame

    def move_camera_and_shoot(o3dvis):
        Rts = []
        images = []
        o3dvis.scene.scene.enable_sun_light(False)
        for n in range(12):
            theta = n * (2 * np.pi) / 12
            Rt = np.eye(4)
            Rt[:3, 3] = t
            Rt[:3, :
               3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_axis_angle(
                   [0, theta, 0])
            print(f"{Rt=}")
            Rts.append(Rt)
            o3dvis.setup_camera(K, Rt, width, height)
            o3dvis.post_redraw()
            o3dvis.export_current_image(f"render-{n}.jpg")
            images.append(f"render-{n}.jpg")
        np.savez("cameras.npz",
                 width=width,
                 height=height,
                 K=K,
                 Rts=Rts,
                 images=images)

    o3d.visualization.draw([model],
                           show_ui=False,
                           width=width,
                           height=height,
                           actions=[("Save Images", move_camera_and_shoot)])

    # Linux only :-(
    # render = rendering.OffscreenRenderer(width, height)
    # render.scene.add_geometry(model)
    # img = render.render_to_image()
    # o3d.io.write_image("render-image.jpg", img)


def albedo_from_images(meshfile, calib_data_file):

    model = o3d.io.read_triangle_model(meshfile)
    tmeshes = o3d.t.geometry.TriangleMesh.from_triangle_mesh_model(model)
    tmeshes = list(tmeshes.values())
    calib = np.load(calib_data_file)
    Ks = list(Tensor(calib["K"]) for _ in range(len(calib["Rts"])))[:2]
    Rts = list(Tensor(Rt) for Rt in calib["Rts"])[:2]
    images = list(o3d.t.io.read_image(imfile) for imfile in calib["images"])[:2]
    calib.close()
    tmeshes[0].project_images_to_albedo(images, Ks, Rts, 256)
    cam_vis = list({
        "name":
            f"camera-{i}",
        "geometry":
            o3d.geometry.LineSet.create_camera_visualization(
                images[0].columns, images[0].rows, K.numpy(), Rt.numpy(), 0.1)
    } for i, (K, Rt) in enumerate(zip(Ks, Rts)))
    o3d.visualization.draw(cam_vis + [{
        "name": meshfile.name,
        "geometry": tmeshes[0]
    }],
                           show_ui=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        choices=('create_dataset', 'albedo_from_images'))
    parser.add_argument("meshfile", type=Path)
    args = parser.parse_args()

    if args.action == "create_dataset":
        create_dataset(args.meshfile)
    else:
        albedo_from_images(args.meshfile, "cameras.npz")
