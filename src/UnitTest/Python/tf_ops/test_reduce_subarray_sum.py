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
import pytest
import os

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
pytestmark = [
    pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                       reason='tf ops not built'),
    pytest.mark.filterwarnings(
        'ignore::DeprecationWarning:.*(tensorflow|protobuf).*'),
]

# check for GPUs and set memory growth to prevent tf from allocating all memory
gpu_devices = []
try:
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in gpu_devices:
        tf.config.experimental.set_memory_growth(dev, True)
except:
    pass

# define the list of devices for running the ops
device_names = ['CPU:0']
if gpu_devices:
    device_names.append('GPU:0')
devices = pytest.mark.parametrize('device_name', device_names)

# the supported input dtypes
value_dtypes = pytest.mark.parametrize(
    'dtype', [np.int32, np.int64, np.float32, np.float64])


@devices
@value_dtypes
@pytest.mark.parametrize('seed', range(3))
def test_reduce_subarray_sum_random(seed, dtype, device_name):
    import open3d.ml.tf as ml3d

    rng = np.random.RandomState(seed)

    values_shape = [rng.randint(100, 200)]
    values = rng.uniform(0, 10, size=values_shape).astype(dtype)

    prefix_sum = [0]
    for i in range(rng.randint(1, 10)):
        prefix_sum.append(
            rng.randint(0, values_shape[0] - prefix_sum[-1]) + prefix_sum[-1])

    expected_result = []
    for start, stop in zip(prefix_sum, prefix_sum[1:] + values_shape):
        # np.sum correctly handles zero length arrays and returns 0
        expected_result.append(np.sum(values[start:stop]))
    np.array(expected_result, dtype=dtype)

    prefix_sum = np.array(prefix_sum, dtype=np.int64)

    with tf.device(device_name):
        result = ml3d.ops.reduce_subarrays_sum(values, prefix_sum)
        assert device_name in result.device
    result = result.numpy()

    if np.issubdtype(dtype, np.integer):
        assert np.all(result == expected_result)
    else:  # floating point types
        assert np.allclose(result, expected_result)
