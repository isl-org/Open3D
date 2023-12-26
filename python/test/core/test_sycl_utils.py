# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import pytest
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


@pytest.mark.skipif(not o3d._build_config["BUILD_SYCL_MODULE"],
                    reason="Skip if SYCL not enabled.")
def test_run_sycl_demo():
    assert o3d.core.sycl_demo() == 0
