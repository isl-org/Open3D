# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d


def test_compute_fpfh_feature_optional_indices_signature():
    doc = o3d.pipelines.registration.compute_fpfh_feature.__doc__ or ""
    signature = doc.splitlines()[0]

    assert "indices:" in signature
    assert "| None = None" in signature
