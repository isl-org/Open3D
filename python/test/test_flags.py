# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d


def test_global_flags():
    assert o3d.pybind._GLIBCXX_USE_CXX11_ABI in (True, False)
    assert o3d.pybind._GLIBCXX_USE_CXX11_ABI == o3d._build_config[
        'GLIBCXX_USE_CXX11_ABI']
    assert o3d._build_config['GLIBCXX_USE_CXX11_ABI'] in (True, False)
    assert o3d._build_config['ENABLE_HEADLESS_RENDERING'] in (True, False)
    assert o3d._build_config['BUILD_CUDA_MODULE'] in (True, False)
