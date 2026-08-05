# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""This example demonstrates project_image_to_albedo. Use create_dataset mode to
render images of a 3D mesh or model from different viewpoints.
albedo_from_dataset mode then uses the calibrated images to re-create the albedo
texture for the mesh.
"""
import argparse
from pathlib import Path
import subprocess as sp
import time
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering, O3DVisualizer
from open3d.core import Tensor


def download_smithsonian_baluster_vase():
    """Download the Smithsonian Baluster Vase 3D model."""
    vase_url = 'https://3d-api.si.edu/content/document/3d_package:d8c62634-4ebc-11ea-b77f-2e728ce88125/resources/F1980.190%E2%80%93194_baluster_vase-150k-4096.glb'
    import urllib.request

    def show_progress(block_num, block_size, total_size):
        total_size = total_size >> 20 if total_size > 0 else "??"  # Convert to MB if known
        print(
            "Downloading F1980_baluster_vase.glb... "
            f"{(block_num * block_size) >>20}MB / {total_size}MB",
            end="\r")

    urllib.request.urlretrieve(vase_url,
                               filename="F1980_baluster_vase.glb",
                               reporthook=show_progress)
    print("\nDownload complete.")


def create_dataset(meshfile, n_images=10, movie=False, vary_exposure=False):
    """Render images of a 3D mesh from different viewpoints, covering the
    northern hemisphere. These form a synthetic dataset to test the
    project_images_to_albedo function.
    """
    # Adjust these parameters to properly frame your model.
    # Window system pixel scaling (e.g. 1 for normal, 2 for HiDPI / retina display)
    SCALING = 2
    width, height = 1024, 1024  # image width, height
    focal_length = 512
    d_camera_obj = 0.3  # distance from camera to object
    K = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2],
                  [0, 0, 1]])
    t = np.array([0, 0, d_camera_obj])  # origin / object in camera ref frame

    model = o3d.io.read_triangle_model(meshfile)
    # DefaultLit shader will produce non-uniform images with specular
    # highlights, etc. These should be avoided to accurately capture the diffuse
    # albedo
    unlit = rendering.MaterialRecord()
    unlit.shader = "unlit"

    def triangle_wave(n, period=1):
        """Triangle wave function between [0,1] with given period."""
        return abs(n % period - period / 2) / (period / 2)

    def rotate_camera_and_shoot(o3dvis):
        Rts = []
        images = []
        o3dvis.scene.scene.enable_sun_light(False)
        print("Rendering images: ", end='', flush=True)
        n_0 = 2 * n_images // 3
        n_1 = n_images - n_0 - 1
        for n in range(n_images):
            Rt = np.eye(4)
            Rt[:3, 3] = t
            if n < n_0:
                theta = n * (2 * np.pi) / n_0
                Rt[:3, :
                   3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_zyx(
                       [np.pi, theta, 0])
            elif n < n_images - 1:
                theta = (n - n_0) * (2 * np.pi) / n_1
                Rt[:3, :
                   3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(
                       [np.pi / 4, theta, np.pi])
            else:  # one image from the top
                Rt[:3, :
                   3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_zyx(
                       [np.pi, 0, -np.pi / 2])
            Rts.append(Rt)
            o3dvis.setup_camera(K, Rt, width, height)
            # Vary IBL intensity as a poxy for exposure value. IBL ranges from
            # [0,150000]. We vary it between 20000 and 100000.
            if vary_exposure:
                o3dvis.set_ibl_intensity(20000 +
                                         80000 * triangle_wave(n, n_images / 4))
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
        o3dvis.close()
        print("\nDone.")

    print("If the object is properly framed in the GUI window, click on the "
          "'Save Images' action in the menu.")
    o3d.visualization.draw([{
        'geometry': model,
        'name': meshfile.name,
        'material': unlit
    }],
                           show_ui=False,
                           width=int(width / SCALING),
                           height=int(height / SCALING),
                           actions=[("Save Images", rotate_camera_and_shoot)])


def albedo_from_images(meshfile, calib_data_file, albedo_contrast=1.25):

    model = o3d.io.read_triangle_model(meshfile)
    tmeshes = o3d.t.geometry.TriangleMesh.from_triangle_mesh_model(model)
    tmeshes = list(tmeshes.values())
    calib = np.load(calib_data_file)
    Ks = list(Tensor(calib["K"]) for _ in range(len(calib["Rts"])))
    Rts = list(Tensor(Rt) for Rt in calib["Rts"])
    images = list(o3d.t.io.read_image(imfile) for imfile in calib["images"])
    calib.close()
    start = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        albedo = tmeshes[0].project_images_to_albedo(images, Ks, Rts, 1024,
                                                     True)
    albedo = albedo.linear_transform(scale=albedo_contrast)  # brighten albedo
    tmeshes[0].material.texture_maps["albedo"] = albedo
    print(f"project_images_to_albedo ran in {time.time()-start:.2f}s")
    o3d.t.io.write_image("albedo.png", albedo)
    o3d.t.io.write_triangle_mesh(meshfile.stem + "_albedo.glb", tmeshes[0])
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=('create_dataset', 'albedo_from_images'))
    parser.add_argument("--meshfile",
                        type=Path,
                        default=".",
                        help="Path to mesh file.")
    parser.add_argument("--n-images",
                        type=int,
                        default=10,
                        help="Number of images to render.")
    parser.add_argument("--download_sample_model",
                        help="Download a sample 3D model for this example.",
                        action="store_true")
    parser.add_argument(
        "--movie",
        action="store_true",
        help=
        "Create movie from rendered images with ffmpeg. ffmpeg must be installed and in path."
    )
    args = parser.parse_args()

    if args.action == "create_dataset":
        if args.download_sample_model:
            download_smithsonian_baluster_vase()
            args.meshfile = "F1980_baluster_vase.glb"
        if args.meshfile == Path("."):
            parser.error("Please provide a path to a mesh file, or use "
                         "--download_sample_model.")
        if args.n_images < 10:
            parser.error("Atleast 10 images should be used!")
        create_dataset(args.meshfile,
                       n_images=args.n_images,
                       movie=args.movie,
                       vary_exposure=True)
    else:
        albedo_from_images(args.meshfile, "cameras.npz")
