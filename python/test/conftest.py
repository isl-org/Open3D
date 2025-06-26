# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import pytest
import importlib.util
import open3d as o3d


def pytest_collection_modifyitems(config, items):
    """
    Skip tests in ml_ops if Open3D is not built with ML support, or PyTorch or
    TensorFlow are not installed.
    """
    skip_ml = None
    if not o3d._build_config["BUILD_TENSORFLOW_OPS"] and not o3d._build_config[
            "BUILD_PYTORCH_OPS"]:
        skip_ml = pytest.mark.skip(reason="Open3D is not built with ML support")

    if skip_ml is not None:
        tf_installed = importlib.util.find_spec("tensorflow") is not None
        torch_installed = importlib.util.find_spec("torch") is not None
        if not tf_installed and not torch_installed:
            skip_ml = pytest.mark.skip(
                reason=
                "Requires at least one of TensorFlow and PyTorch to be installed"
            )

    if skip_ml is None:
        return

    for item in items:
        if "ml_ops" in str(item.path):
            item.add_marker(skip_ml)
