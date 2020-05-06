# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2019 www.open3d.org
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
import pytest
import os


def test_load_tf_op_library():

    if not o3d._build_config['BUILD_TENSORFLOW_OPS']:
        return

    import open3d.ml.tf as ml3d
    assert hasattr(ml3d.python.ops.lib._lib, 'OP_LIST')


def test_execute_tf_op():

    if not o3d._build_config['BUILD_TENSORFLOW_OPS']:
        return

    import open3d.ml.tf as ml3d

    values = np.arange(0, 10)
    prefix_sum = np.array([0, 3, 4, 4])

    ans = ml3d.ops.reduce_subarrays_sum(values, prefix_sum)
    # test was a success if we reach this line but check correctness anyway
    assert np.all(ans.numpy() == [3, 3, 0, 39])
