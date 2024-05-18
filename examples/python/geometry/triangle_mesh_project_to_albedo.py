import argparse
from pathlib import Path
import subprocess as sp
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering, O3DVisualizer
from open3d.core import Tensor


def create_dataset(meshfile, n_images=10, movie=False):
    SCALING = 2
    width, height = 1024, 1024
    focal_length = 512
    K = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2],
                  [0, 0, 1]])
    model = o3d.io.read_triangle_model(meshfile)
    unlit = rendering.MaterialRecord()
    unlit.shader = "unlit"
    t = np.array([0, 0, 0.3])  # origin / object in camera ref frame

    def rotate_camera_and_shoot(o3dvis):
        Rts = []
        images = []
        o3dvis.scene.scene.enable_sun_light(False)
        print("Rendering images: ", end='', flush=True)
        for n in range(n_images):
            theta = n * (2 * np.pi) / n_images
            Rt = np.eye(4)
            Rt[:3, 3] = t
            Rt[:3, :3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_zyx(
                [np.pi, theta, 0])
            Rts.append(Rt)
            o3dvis.setup_camera(K, Rt, width, height)
            o3dvis.post_redraw()
            o3dvis.export_current_image(f"render-{n:02}.jpg")
            images.append(f"render-{n:02}.jpg")
            print('.', end='', flush=True)
        np.savez("cameras.npz",
                 width=width,
                 height=height,
                 K=K,
                 Rts=Rts,
                 images=images)
        # Now create a movie from the saved images by calling ffmpeg with
        # subprocess
        if movie:
            print("\nCreating movie...", end='', flush=True)
            sp.run([
                "ffmpeg", "-framerate", f"{n_images/6}", "-pattern_type",
                "glob", "-i", "render-*.jpg", "-y", meshfile.stem + ".mp4"
            ],
                   check=True)
        print("\nDone.")

    o3d.visualization.draw([{
        'geometry': model,
        'name': meshfile.name,
        'material': unlit
    }],
                           show_ui=False,
                           width=int(width / SCALING),
                           height=int(height / SCALING),
                           actions=[("Save Images", rotate_camera_and_shoot)])

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
    Ks = list(Tensor(calib["K"]) for _ in range(len(calib["Rts"])))
    Rts = list(Tensor(Rt) for Rt in calib["Rts"])
    images = list(o3d.t.io.read_image(imfile) for imfile in calib["images"])
    calib.close()
    # breakpoint()
    albedo = tmeshes[0].project_images_to_albedo(images, Ks, Rts, 1024)
    o3d.t.io.write_image("albedo.png", albedo)
    cam_vis = list({
        "name":
            f"camera-{i:02}",
        "geometry":
            o3d.geometry.LineSet.create_camera_visualization(
                images[0].columns, images[0].rows, K.numpy(), Rt.numpy(), 0.1)
    } for i, (K, Rt) in enumerate(zip(Ks, Rts)))
    o3d.visualization.draw(cam_vis + [{
        "name": meshfile.name,
        "geometry": tmeshes[0]
    }],
                           show_ui=True)
    o3d.t.io.write_triangle_mesh(meshfile.stem + "_albedo.glb", tmeshes[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        choices=('create_dataset', 'albedo_from_images'))
    parser.add_argument("meshfile", type=Path)
    parser.add_argument("--n-images",
                        type=int,
                        default=10,
                        help="Number of images to render.")
    parser.add_argument(
        "--movie",
        action="store_true",
        help=
        "Create movie from rendered images with ffmpeg. ffmpeg must be installed and in path."
    )
    args = parser.parse_args()

    if args.action == "create_dataset":
        create_dataset(args.meshfile, n_images=args.n_images, movie=args.movie)
    else:
        albedo_from_images(args.meshfile, "cameras.npz")
