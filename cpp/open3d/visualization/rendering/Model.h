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

    /**
     * Breaks a mesh into separate #MeshInfo structures and add them to the
     * TriangleMeshModel based on the material assigned to each triangle using
     * the mesh's geometry::TriangleMesh#triangle_matrial_ids_. A
     * geometry::TriangleMesh where
     * geometry::TriangleMesh#triangle_material_ids_ is empty will just be
     * copied
     * @param mesh The mesh to be broken apart and added to the model
     * @param name The base name of the mesh to be used for the generated
     * #MeshInfo objects
     */
    void AddMesh(const geometry::TriangleMesh& mesh, const std::string& name);

    /**
     * Helper function to create a TriangleMeshModel from a TriangleMesh
     * @see AddMesh()
     * @param mesh The mesh to break apart from
     * @param name The name of the mesh to be used in the #MeshInfo struct
     * @return The constructed TriangleMeshModel struct
     */
    static TriangleMeshModel FromTriangleMesh(
            const geometry::TriangleMesh& mesh,
            const std::string& name = "Mesh");
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
