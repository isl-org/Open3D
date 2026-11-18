# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import pytest


@pytest.mark.parametrize("name", [
    "read_point_cloud", "write_point_cloud", "read_triangle_mesh",
    "write_triangle_mesh"
])
def test_file_path_annotation_preserves_union(name):
    signature = getattr(o3d.io, name).__doc__.splitlines()[0]
    assert "filename: os.PathLike | str | bytes" in signature
