# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
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


@pytest.mark.parametrize("vector_name, width, dtype", [
    ("Vector2dVector", 2, np.float64),
    ("Vector3dVector", 3, np.float64),
    ("Vector2iVector", 2, np.int32),
    ("Vector3iVector", 3, np.int32),
    ("Vector4iVector", 4, np.int32),
])
@pytest.mark.parametrize("writeable", [True, False],
                         ids=["writable", "readonly"])
@pytest.mark.parametrize(
    "layout",
    ["c", "empty", "fortran", "column_slice", "reversed", "converted"])
def test_eigen_vector_numpy_input_is_copied(vector_name, width, dtype,
                                            writeable, layout):
    input_array = np.arange(4 * width, dtype=dtype).reshape(4, width)
    if layout == "empty":
        input_array = input_array[:0]
    elif layout == "fortran":
        input_array = np.asfortranarray(input_array)
    elif layout == "column_slice":
        input_array = np.arange(8 * width, dtype=dtype).reshape(4, 2 * width)
        input_array = input_array[:, ::2]
    elif layout == "reversed":
        input_array = input_array[::-1, ::-1]
    elif layout == "converted":
        converted_dtype = np.float32 if dtype == np.float64 else np.int64
        input_array = input_array.astype(converted_dtype)
    input_array.setflags(write=writeable)
    expected = input_array.copy()

    vector = getattr(o3d.utility, vector_name)(input_array)
    output_array = np.asarray(vector)

    assert output_array.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(output_array, expected)
    assert not np.shares_memory(output_array, input_array)
    assert input_array.flags.writeable == writeable
    np.testing.assert_array_equal(input_array, expected)

    # Open3D owns the copy even when the NumPy input is read-only.
    output_array[...] = -1
    np.testing.assert_array_equal(np.asarray(vector),
                                  np.full_like(expected, -1))
    np.testing.assert_array_equal(input_array, expected)
    assert input_array.flags.writeable == writeable


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
