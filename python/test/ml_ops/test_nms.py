# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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
import mltest

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


@mltest.parametrize.ml
def test_nms(ml):
    boxes = np.array([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                      [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                      [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                      [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                      [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                      [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                     dtype=np.float32)
    scores = np.array([3, 1.1, 5, 2, 1, 0], dtype=np.float32)
    nms_overlap_thresh = 0.7
    keep_indices_ref = np.array([2, 3, 5]).astype(np.int64)

    keep_indices = mltest.run_op(ml,
                                 ml.device,
                                 True,
                                 ml.ops.nms,
                                 boxes,
                                 scores,
                                 nms_overlap_thresh=nms_overlap_thresh)

    np.testing.assert_equal(keep_indices, keep_indices_ref)
    assert keep_indices.dtype == keep_indices_ref.dtype


@mltest.parametrize.ml
def test_nms_empty(ml):
    boxes = np.zeros((0, 5), dtype=np.float32)
    scores = np.array([], dtype=np.float32)
    nms_overlap_thresh = 0.7
    keep_indices_ref = np.array([]).astype(np.int64)

    keep_indices = mltest.run_op(ml,
                                 ml.device,
                                 True,
                                 ml.ops.nms,
                                 boxes,
                                 scores,
                                 nms_overlap_thresh=nms_overlap_thresh)

    np.testing.assert_equal(keep_indices, keep_indices_ref)
    assert keep_indices.dtype == keep_indices_ref.dtype
