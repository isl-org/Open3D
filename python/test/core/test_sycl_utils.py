# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
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
@pytest.mark.xfail(raises=RuntimeError,
                   reason="Github Actions Windows: "
                   "No device of requested type available.")
def test_run_sycl_demo():
    assert o3d.core.sycl_demo() == 0


@pytest.mark.skipif(not o3d._build_config["BUILD_SYCL_MODULE"],
                    reason="Skip if SYCL not enabled.")
def test_sycl_device_properties():
    devices = o3d.core.sycl.get_available_devices()
    if not devices:
        pytest.skip("No SYCL device available.")
    for device in devices:
        props = o3d.core.sycl.get_device_properties(device)
        assert props.compute_units > 0
        assert props.local_mem_size > 0
        assert props.max_work_group_size > 0
        assert isinstance(props.sub_group_sizes, list)
        for size in props.sub_group_sizes:
            assert props.supports_subgroup_size(size)
        assert not props.supports_subgroup_size(0)
