# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import tempfile
import os


def test_apply_depth_noise_model():

    class Simulator:
        """Original implementation of the simulator:
        http://redwood-data.org/indoor/data/simdepth.py
        """

        def loaddistmodel(self, fname):

            data = np.loadtxt(fname, comments='%', skiprows=5)
            dist = np.empty([80, 80, 5])

            for y in range(0, 80):
                for x in range(0, 80):
                    idx = (y * 80 + x) * 23 + 3
                    if (data[idx:idx + 5] < 8000).all():
                        dist[y, x, :] = 0
                    else:
                        dist[y, x, :] = data[idx + 15:idx + 20]

            self.model = dist

        def undistort(self, x, y, z):

            i2 = int((z + 1) / 2)
            i1 = i2 - 1
            a = (z - (i1 * 2 + 1)) / 2
            x = int(x / 8)
            y = int(y / 6)
            f = (1 - a) * self.model[y, x, min(max(i1, 0), 4)] + a * self.model[
                y, x, min(i2, 4)]

            if f == 0:
                return 0
            else:
                return z / f

        def simulate(self,
                     inputpng,
                     outputpng,
                     depth_scale=1000.0,
                     deterministic=False):

            a = o3d.t.io.read_image(inputpng).as_tensor()
            a = a.numpy().squeeze().astype(np.float32) / depth_scale
            b = np.copy(a)
            it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])

            while not it.finished:

                # pixel shuffle
                # here, 639 == width - 1, 479 == height - 1
                if deterministic:
                    x = min(max(round(it.multi_index[1] + 0), 0), 639)
                    y = min(max(round(it.multi_index[0] + 0), 0), 479)
                else:
                    x = min(
                        max(
                            round(it.multi_index[1] +
                                  np.random.normal(0, 0.25)), 0), 639)
                    y = min(
                        max(
                            round(it.multi_index[0] +
                                  np.random.normal(0, 0.25)), 0), 479)

                # downsample
                d = b[y - y % 2, x - x % 2]

                # distortion
                d = self.undistort(x, y, d)

                # quantization and high freq noise
                if d == 0:
                    it[0] = 0
                else:
                    if deterministic:
                        it[0] = 35.130 * 8 / round((35.130 / d + 0) * 8)
                    else:
                        it[0] = 35.130 * 8 / round(
                            (35.130 / d + np.random.normal(0, 0.027778)) * 8)

                it.iternext()

            a = (a * depth_scale).astype(np.uint16)
            a = np.expand_dims(a, axis=2)
            o3d.t.io.write_image(outputpng, o3d.t.geometry.Image(a))

    # Load dataset.
    data = o3d.data.RedwoodIndoorLivingRoom1()
    noise_model_path = data.noise_model_path
    depth_scale = 1000.0

    # Source image.
    im_src_path = data.depth_paths[0]
    im_src_uint16 = o3d.t.io.read_image(im_src_path).as_tensor().numpy()
    im_src_float32 = im_src_uint16.astype(np.float32) / depth_scale

    # Simulate "ground truth" noise depth image, with deterministic noise.
    # We use the original simulator, which requires input and output path.
    # See http://redwood-data.org/indoor/data/simdepth.py for the original
    # implementation.
    gt_simulator = Simulator()
    gt_simulator.loaddistmodel(noise_model_path)
    with tempfile.TemporaryDirectory() as dst_dir:
        im_dst_path = os.path.join(dst_dir, "noisy_depth.png")
        gt_simulator.simulate(im_src_path,
                              im_dst_path,
                              depth_scale=1000.0,
                              deterministic=True)

        im_dst_gt = o3d.t.io.read_image(im_dst_path).as_tensor().numpy()
        im_dst_gt = im_dst_gt.astype(np.float32) / depth_scale

    # Our C++ implementation of the simulator.
    simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)
    simulator.enable_deterministic_debug_mode()
    np.testing.assert_allclose(simulator.noise_model.numpy(),
                               gt_simulator.model)

    # With uint16 input.
    im_src = o3d.t.geometry.Image(im_src_uint16)
    im_dst = simulator.simulate(im_src, depth_scale=depth_scale)
    im_dst = im_dst.as_tensor().numpy().astype(np.float32) / depth_scale
    np.testing.assert_allclose(im_dst, im_dst_gt)

    # With float32 input.
    im_src = o3d.t.geometry.Image(im_src_float32)
    im_dst = simulator.simulate(im_src, depth_scale=1.0)
    im_dst = im_dst.as_tensor().numpy()
    im_dst = (im_dst * depth_scale).astype(np.uint16).astype(
        np.float32) / depth_scale  # Simulate rounding integers.
    np.testing.assert_allclose(im_dst, im_dst_gt)
