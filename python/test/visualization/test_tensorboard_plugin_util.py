# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib.util
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


_UTIL_PATH = (Path(__file__).resolve().parents[2] / "open3d" / "visualization" /
              "tensorboard_plugin" / "util.py")


def _load_source_util():
    module_name = "open3d.visualization.tensorboard_plugin.util_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _UTIL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_uint8_open3d_tensor_returns_input():
    util = _load_source_util()
    tensor = o3d.core.Tensor([[1, 2, 3]], dtype=o3d.core.uint8)

    normalized, min_val, max_val = util._normalize(tensor)

    assert normalized is tensor
    assert min_val == 0
    assert max_val == 1


def test_normalize_float_inputs_support_open3d_and_numpy():
    util = _load_source_util()

    open3d_tensor = o3d.core.Tensor([[0.0, 2.0]], dtype=o3d.core.float32)
    normalized_open3d, min_open3d, max_open3d = util._normalize(open3d_tensor)
    np.testing.assert_allclose(
        normalized_open3d.numpy(),
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    assert min_open3d == 0.0
    assert max_open3d == 2.0

    numpy_tensor = np.array([[1.0, 3.0]], dtype=np.float32)
    normalized_numpy, min_numpy, max_numpy = util._normalize(numpy_tensor)
    np.testing.assert_allclose(
        normalized_numpy,
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    assert min_numpy == 1.0
    assert max_numpy == 3.0
