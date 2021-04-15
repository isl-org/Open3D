# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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
import time
import pytest


@pytest.mark.parametrize(
    "input_array, expect_exception",
    [
        # Empty case
        (np.ones((0, 3), dtype=np.float64), False),
        # Wrong shape
        (np.ones((2, 4), dtype=np.float64), True),
        # Non-numpy array
        ([[1, 2, 3], [4, 5, 6]], False),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], False),
        # Datatypes
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), False),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False),
        # Slice non-contiguous memory
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                  dtype=np.float64)[:, 0:6:2], False),
        # Transpose view
        (np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64).T, False),
        # Fortran layout
        (np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]],
                                    dtype=np.float64)), False),
    ])
def test_Vector3dVector(input_array, expect_exception):

    def run_test(input_array):
        open3d_array = o3d.utility.Vector3dVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)

    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)


@pytest.mark.parametrize(
    "input_array, expect_exception",
    [
        # Empty case
        (np.ones((0, 3), dtype=np.int32), False),
        # Wrong shape
        (np.ones((2, 4), dtype=np.int32), True),
        # Non-numpy array
        ([[1, 2, 3], [4, 5, 6]], False),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], False),
        # Datatypes
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), False),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), False),
        # Slice non-contiguous memory
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                  dtype=np.int32)[:, 0:6:2], False),
        # Transpose view
        (np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32).T, False),
        # Fortran layout
        (np.asfortranarray(np.array([[1, 2, 3], [4, 5, 6]],
                                    dtype=np.int32)), False),
    ])
def test_Vector3iVector(input_array, expect_exception):

    def run_test(input_array):
        open3d_array = o3d.utility.Vector3iVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)

    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)


@pytest.mark.parametrize(
    "input_array, expect_exception",
    [
        # Empty case
        (np.ones((0, 2), dtype=np.int32), False),
        # Wrong shape
        (np.ones((10, 3), dtype=np.int32), True),
        # Non-numpy array
        ([[1, 2], [4, 5]], False),
        ([[1.0, 2.0], [4.0, 5.0]], False),
        # Datatypes
        (np.array([[1, 2], [4, 5]], dtype=np.float64), False),
        (np.array([[1, 2], [4, 5]], dtype=np.int32), False),
        (np.array([[1, 2], [4, 5]], dtype=np.int32), False),
        # Slice non-contiguous memory
        (np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                  dtype=np.int32)[:, 0:6:3], False),
        # Transpose view
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).T, False),
        # Fortran layout
        (np.asfortranarray(np.array([[1, 2], [4, 5]], dtype=np.int32)), False),
    ])
def test_Vector2iVector(input_array, expect_exception):

    def run_test(input_array):
        open3d_array = o3d.utility.Vector2iVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)

    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)


@pytest.mark.parametrize(
    "input_array, expect_exception",
    [
        # Empty case
        (np.ones((0, 4, 4), dtype=np.float64), False),
        # Wrong shape
        (np.ones((10, 3), dtype=np.float64), True),
        (np.ones((10, 3, 3), dtype=np.float64), True),
        # Non-numpy array
        ([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]], False
        ),
        # Datatypes
        (np.random.randint(10, size=(10, 4, 4)).astype(np.float64), False),
        (np.random.randint(10, size=(10, 4, 4)).astype(np.int32), False),
        # Slice non-contiguous memory
        (np.random.random(
            (10, 8, 8)).astype(np.float64)[:, 0:8:2, 0:8:2], False),
        # Fortran layout
        (np.asfortranarray(
            np.array(np.random.random((10, 4, 4)), dtype=np.float64)), False),
    ])
def test_Matrix4dVector(input_array, expect_exception):

    def run_test(input_array):
        open3d_array = o3d.utility.Matrix4dVector(input_array)
        output_array = np.asarray(open3d_array)
        np.testing.assert_allclose(input_array, output_array)

    if expect_exception:
        with pytest.raises(Exception):
            run_test(input_array)
    else:
        run_test(input_array)


# Run with pytest -s to show output
def test_benchmark():
    vector_size = int(2e6)

    x = np.random.randint(10, size=(vector_size, 3)).astype(np.float64)
    print("\no3d.utility.Vector3dVector:", x.shape)
    start_time = time.time()
    y = o3d.utility.Vector3dVector(x)
    print("open3d -> numpy: %.6fs" % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print("numpy -> open3d: %.6fs" % (time.time() - start_time))
    np.testing.assert_allclose(x, z)

    print("\no3d.utility.Vector3iVector:", x.shape)
    x = np.random.randint(10, size=(vector_size, 3)).astype(np.int32)
    start_time = time.time()
    y = o3d.utility.Vector3iVector(x)
    print("open3d -> numpy: %.6fs" % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print("numpy -> open3d: %.6fs" % (time.time() - start_time))
    np.testing.assert_allclose(x, z)

    print("\no3d.utility.Vector2iVector:", x.shape)
    x = np.random.randint(10, size=(vector_size, 2)).astype(np.int32)
    start_time = time.time()
    y = o3d.utility.Vector2iVector(x)
    print("open3d -> numpy: %.6fs" % (time.time() - start_time))
    start_time = time.time()
    z = np.asarray(y)
    print("numpy -> open3d: %.6fs" % (time.time() - start_time))
    np.testing.assert_allclose(x, z)
