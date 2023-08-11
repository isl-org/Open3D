// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/MaterialRecord.h"

namespace open3d {
namespace visualization {
namespace rendering {

struct TriangleMeshModel {
    struct MeshInfo {
        std::shared_ptr<geometry::TriangleMesh> mesh;
        std::string mesh_name;
        unsigned int material_idx;
    };

    std::vector<MeshInfo> meshes_;
    std::vector<visualization::rendering::MaterialRecord> materials_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
