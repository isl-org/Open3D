# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import sys
import argparse
from pathlib import Path
import open3d as o3d
import mitsuba as mi
import drjit as dr
import numpy as np
import math


def make_mitsuba_scene(mesh, cam_xform, fov, width, height, principle_pts,
                       envmap):
    # Camera transform
    t_from_np = mi.ScalarTransform4f(cam_xform)
    # Transform necessary to get from Open3D's environment map coordinate system
    # to Mitsuba's
    env_t = mi.ScalarTransform4f.rotate(axis=[0, 0, 1],
                                        angle=90).rotate(axis=[1, 0, 0],
                                                         angle=90)
    scene_dict = {
        "type": "scene",
        "integrator": {
            'type': 'path'
        },
        "light": {
            "type": "envmap",
            "to_world": env_t,
            "bitmap": mi.Bitmap(envmap),
        },
        "sensor": {
            "type": "perspective",
            "fov": fov,
            "to_world": t_from_np,
            "principal_point_offset_x": principle_pts[0],
            "principal_point_offset_y": principle_pts[1],
            "thefilm": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
            },
            "thesampler": {
                "type": "multijitter",
                "sample_count": 64,
            },
        },
        "themesh": mesh,
    }

    scene = mi.load_dict(scene_dict)
    return scene


def run_estimation(mesh, cam_info, ref_image, env_width, iterations, tv_alpha):
    # Make Mitsuba mesh from Open3D mesh -- conversion will attach a Mitsuba
    # Principled BSDF to the mesh
    mesh_opt = mesh.to_mitsuba('themesh')

    # Prepare empty environment map
    empty_envmap = np.ones((int(env_width / 2), env_width, 3))

    # Create Mitsuba scene
    scene = make_mitsuba_scene(mesh_opt, cam_info[0], cam_info[1], cam_info[2],
                               cam_info[3], cam_info[4], empty_envmap)

    def total_variation(image, alpha):
        diff1 = image[1:, :, :] - image[:-1, :, :]
        diff2 = image[:, 1:, :] - image[:, :-1, :]
        return alpha * (dr.sum(dr.abs(diff1)) / len(diff1) +
                        dr.sum(dr.abs(diff2)) / len(diff2))

    def mse(image, ref_img):
        return dr.mean(dr.sqr(image - ref_img))

    params = mi.traverse(scene)
    print(params)

    # Create a Mitsuba Optimizer and configure it to optimize albedo and
    # environment maps
    opt = mi.ad.Adam(lr=0.05, mask_updates=True)
    opt['themesh.bsdf.base_color.data'] = params['themesh.bsdf.base_color.data']
    opt['light.data'] = params['light.data']
    params.update(opt)

    integrator = mi.load_dict({'type': 'prb'})
    for i in range(iterations):
        img = mi.render(scene, params, spp=8, seed=i, integrator=integrator)

        # Compute loss
        loss = mse(img, ref_image)
        # Apply TV regularization if requested
        if tv_alpha > 0.0:
            loss = loss + total_variation(opt['themesh.bsdf.base_color.data'],
                                          tv_alpha)

        # Backpropogate and step. Note: if we were optimizing over a larger set
        # of inputs not just a single image we might want to step only every x
        # number of inputs
        dr.backward(loss)
        opt.step()

        # Make sure albedo values stay in allowed range
        opt['themesh.bsdf.base_color.data'] = dr.clamp(
            opt['themesh.bsdf.base_color.data'], 0.0, 1.0)
        params.update(opt)
        print(f'Iteration {i} complete')

    # Done! Return the estimated maps
    albedo_img = params['themesh.bsdf.base_color.data'].numpy()
    envmap_img = params['light.data'].numpy()
    return (albedo_img, envmap_img)


def load_input_mesh(model_path, tex_dim):
    mesh = o3d.t.io.read_triangle_mesh(model_path)
    mesh.material.set_default_properties()
    mesh.material.material_name = 'defaultLit'  # note: ignored by Mitsuba, just used to visualize in Open3D
    mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(0.5 + np.zeros(
        (tex_dim, tex_dim, 3), dtype=np.float32))
    return mesh


def load_input_data(object, camera_pose, input_image, tex_dim):
    print(f'Loading {object}...')
    mesh = load_input_mesh(object, tex_dim)

    print(f'Loading camera pose from {camera_pose}...')
    cam_npz = np.load(camera_pose)
    img_width = cam_npz['width'].item()
    img_height = cam_npz['height'].item()
    cam_xform = np.linalg.inv(cam_npz['T'])
    cam_xform = np.matmul(
        cam_xform,
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                 dtype=np.float32))
    fov = 2 * np.arctan(0.5 * img_width / cam_npz['K'][0, 0])
    fov = (180.0 / math.pi) * fov.item()
    camera = (cam_xform, fov, img_width, img_height, (0.0, 0.0))

    print(f'Loading reference image from {input_image}...')
    ref_img = o3d.t.io.read_image(str(input_image))
    ref_img = ref_img.as_tensor()[:, :, 0:3].to(o3d.core.Dtype.Float32) / 255.0
    bmp = mi.Bitmap(ref_img.numpy()).convert(srgb_gamma=False)
    ref_img = mi.TensorXf(bmp)
    return (mesh, camera, ref_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Script that estimates texture and environment map from an input image and geometry. You can find data to test this script here: https://github.com/isl-org/open3d_downloads/releases/download/mitsuba-demos/raven_mitsuba.zip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'object_path',
        type=Path,
        help=
        "Path to geometry for which to estimate albedo. It is assumed that in the same directory will be an object-name.npz which contains the camera pose information and an object-name.png which is the input image"
    )
    parser.add_argument('--env-width', type=int, default=1024)
    parser.add_argument('--tex-width',
                        type=int,
                        default=2048,
                        help="The dimensions of the texture")
    parser.add_argument(
        '--device',
        default='cuda' if o3d.core.cuda.is_available() else 'cpu',
        choices=('cpu', 'cuda'),
        help="Run Mitsuba on 'cuda' or 'cpu'")
    parser.add_argument('--iterations',
                        type=int,
                        default=40,
                        help="Number of iterations")
    parser.add_argument(
        '--total-variation',
        type=float,
        default=0.01,
        help="Factor to apply to total_variation loss. 0.0 disables TV")

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print("Arguments: ", vars(args))

    # Initialize Mitsuba
    if args.device == 'cpu':
        mi.set_variant('llvm_ad_rgb')
    else:
        mi.set_variant('cuda_ad_rgb')

    # Confirm that the 3 required inputs exist
    object_path = args.object_path
    object_name = object_path.stem
    datadir = args.object_path.parent
    camera_pose = datadir / (object_name + '.npz')
    input_image = datadir / (object_name + '.png')
    if not object_path.exists():
        print(f'{object_path} does not exist!')
        sys.exit()
    if not camera_pose.exists():
        print(f'{camera_pose} does not exist!')
        sys.exit()
    if not input_image.exists():
        print(f'{input_image} does not exist!')
        sys.exit()

    # Load input data
    mesh, cam_info, input_image = load_input_data(object_path, camera_pose,
                                                  input_image, args.tex_width)

    # Estimate albedo map
    print('Running material estimation...')
    albedo, envmap = run_estimation(mesh, cam_info, input_image, args.env_width,
                                    args.iterations, args.total_variation)

    # Save maps
    def save_image(img, name, output_dir):
        # scale to 0-255
        texture = o3d.core.Tensor(img * 255.0).to(o3d.core.Dtype.UInt8)
        texture = o3d.t.geometry.Image(texture)
        o3d.t.io.write_image(str(output_dir / name), texture)

    print('Saving final results...')
    save_image(albedo, 'estimated_albedo.png', datadir)
    mi.Bitmap(envmap).write(str(datadir / 'predicted_envmap.exr'))

    # Visualize result with Open3D
    mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(
        str(datadir / 'estimated_albedo.png'))
    o3d.visualization.draw(mesh)
