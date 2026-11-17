# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import base64
import json
import struct

import numpy as np
import open3d as o3d
import pytest


def _read_gltf(path):
    """Read exported metadata and bytes without another mesh importer."""
    data = path.read_bytes()
    chunks = {}
    if path.suffix == ".gltf":
        model = json.loads(data)
    else:
        assert struct.unpack_from("<4sII", data) == (b"glTF", 2, len(data))
        offset = 12
        while offset < len(data):
            length, kind = struct.unpack_from("<I4s", data, offset)
            offset += 8
            chunks[kind] = data[offset:offset + length]
            offset += length
        assert offset == len(data)
        model = json.loads(chunks[b"JSON"])

    uri = model["buffers"][0].get("uri")
    if uri is not None:
        header, payload = uri.split(",", 1)
        assert header.startswith("data:") and header.endswith(";base64")
        return model, base64.b64decode(payload, validate=True)
    return model, chunks[b"BIN\x00"]


@pytest.mark.parametrize("suffix", [".gltf", ".glb"])
@pytest.mark.parametrize("vertex_count, triangles, component_type", [
    pytest.param(6, [[0, 1, 2], [3, 4, 5]], 5123, id="contiguous-control"),
    pytest.param(4, [[0, 2, 1], [3, 1, 2]], 5123, id="shared-vertices"),
    pytest.param(6, [[2, 3, 4], [3, 5, 4]], 5123, id="nonzero-minimum"),
    pytest.param(20, [[2, 7, 11], [2, 11, 17]], 5123, id="sparse-uint16"),
    pytest.param(
        65539, [[1, 2, 65536], [2, 3, 65536]], 5125, id="sparse-uint32"),
])
def test_gltf_index_accessor_bounds(tmp_path, suffix, vertex_count, triangles,
                                    component_type):
    """Accessor bounds must describe the serialized indices, not their count."""
    # Two triangles keep the index buffer aligned independently of glTF padding.
    indices = np.asarray(triangles, dtype=np.int32)
    vertex_ids = np.arange(vertex_count)
    vertices = np.column_stack(
        (vertex_ids % 2, (vertex_ids // 2) % 2, vertex_ids // 4))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(indices)
    output_path = tmp_path / ("mesh" + suffix)
    assert o3d.io.write_triangle_mesh(str(output_path), mesh)

    model, buffer = _read_gltf(output_path)
    primitive = model["meshes"][0]["primitives"][0]
    accessor = model["accessors"][primitive["indices"]]
    view = model["bufferViews"][accessor["bufferView"]]
    assert accessor["type"] == "SCALAR"
    assert accessor["componentType"] == component_type
    assert accessor["count"] == indices.size
    assert view["buffer"] == 0
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    component_format = {5123: "H", 5125: "I"}[component_type]
    exported_indices = struct.unpack_from(
        f"<{accessor['count']}{component_format}", buffer, offset)

    np.testing.assert_array_equal(exported_indices, indices.ravel())
    assert accessor["min"] == [min(exported_indices)]
    assert accessor["max"] == [max(exported_indices)]
