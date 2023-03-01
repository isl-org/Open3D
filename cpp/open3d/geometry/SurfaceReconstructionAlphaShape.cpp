// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <iostream>
#include <list>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/geometry/VtkUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> TriangleMesh::CreateFromPointCloudAlphaShape(
        const PointCloud& pcd,
        double alpha,
        std::shared_ptr<TetraMesh> tetra_mesh,
        std::vector<size_t>* pt_map) {
    std::vector<size_t> pt_map_computed;
    if (tetra_mesh == nullptr) {
        utility::LogDebug(
                "[CreateFromPointCloudAlphaShape] "
                "ComputeDelaunayTetrahedralization");
        std::tie(tetra_mesh, pt_map_computed) =
                Qhull::ComputeDelaunayTetrahedralization(pcd.points_);
        pt_map = &pt_map_computed;
        utility::LogDebug(
                "[CreateFromPointCloudAlphaShape] done "
                "ComputeDelaunayTetrahedralization");
    }

    utility::LogDebug("[CreateFromPointCloudAlphaShape] init triangle mesh");
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_ = tetra_mesh->vertices_;
    if (pcd.HasNormals()) {
        mesh->vertex_normals_.resize(mesh->vertices_.size());
        for (size_t idx = 0; idx < (*pt_map).size(); ++idx) {
            mesh->vertex_normals_[idx] = pcd.normals_[(*pt_map)[idx]];
        }
    }
    if (pcd.HasColors()) {
        mesh->vertex_colors_.resize(mesh->vertices_.size());
        for (size_t idx = 0; idx < (*pt_map).size(); ++idx) {
            mesh->vertex_colors_[idx] = pcd.colors_[(*pt_map)[idx]];
        }
    }
    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done init triangle mesh");

    std::vector<double> vsqn(tetra_mesh->vertices_.size());
    for (size_t vidx = 0; vidx < vsqn.size(); ++vidx) {
        vsqn[vidx] = tetra_mesh->vertices_[vidx].squaredNorm();
    }

    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] add triangles from tetras that "
            "satisfy constraint");
    const auto& verts = tetra_mesh->vertices_;
    for (size_t tidx = 0; tidx < tetra_mesh->tetras_.size(); ++tidx) {
        const auto& tetra = tetra_mesh->tetras_[tidx];
        // clang-format off
        Eigen::Matrix4d tmp;
        tmp << verts[tetra(0)](0), verts[tetra(0)](1), verts[tetra(0)](2), 1,
                verts[tetra(1)](0), verts[tetra(1)](1), verts[tetra(1)](2), 1,
                verts[tetra(2)](0), verts[tetra(2)](1), verts[tetra(2)](2), 1,
                verts[tetra(3)](0), verts[tetra(3)](1), verts[tetra(3)](2), 1;
        double a = tmp.determinant();
        tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](1), verts[tetra(0)](2),
                vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](1), verts[tetra(1)](2),
                vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](1), verts[tetra(2)](2),
                vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](1), verts[tetra(3)](2);
        double c = tmp.determinant();
        tmp << vsqn[tetra(0)], verts[tetra(0)](1), verts[tetra(0)](2), 1,
                vsqn[tetra(1)], verts[tetra(1)](1), verts[tetra(1)](2), 1,
                vsqn[tetra(2)], verts[tetra(2)](1), verts[tetra(2)](2), 1,
                vsqn[tetra(3)], verts[tetra(3)](1), verts[tetra(3)](2), 1;
        double dx = tmp.determinant();
        tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](2), 1,
                vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](2), 1,
                vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](2), 1,
                vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](2), 1;
        double dy = tmp.determinant();
        tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](1), 1,
                vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](1), 1,
                vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](1), 1,
                vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](1), 1;
        double dz = tmp.determinant();
        // clang-format on
        if (a == 0) {
            utility::LogWarning(
                    "[CreateFromPointCloudAlphaShape] invalid tetra in "
                    "TetraMesh");
        } else {
            double r = std::sqrt(dx * dx + dy * dy + dz * dz - 4 * a * c) /
                       (2 * std::abs(a));

            if (r <= alpha) {
                mesh->triangles_.push_back(TriangleMesh::GetOrderedTriangle(
                        tetra(0), tetra(1), tetra(2)));
                mesh->triangles_.push_back(TriangleMesh::GetOrderedTriangle(
                        tetra(0), tetra(1), tetra(3)));
                mesh->triangles_.push_back(TriangleMesh::GetOrderedTriangle(
                        tetra(0), tetra(2), tetra(3)));
                mesh->triangles_.push_back(TriangleMesh::GetOrderedTriangle(
                        tetra(1), tetra(2), tetra(3)));
            }
        }
    }
    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done add triangles from tetras "
            "that satisfy constraint");

    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] remove triangles within "
            "the mesh");
    std::unordered_map<Eigen::Vector3i, int,
                       utility::hash_eigen<Eigen::Vector3i>>
            triangle_count;
    for (size_t tidx = 0; tidx < mesh->triangles_.size(); ++tidx) {
        Eigen::Vector3i triangle = mesh->triangles_[tidx];
        if (triangle_count.count(triangle) == 0) {
            triangle_count[triangle] = 1;
        } else {
            triangle_count[triangle] += 1;
        }
    }

    size_t to_idx = 0;
    for (size_t tidx = 0; tidx < mesh->triangles_.size(); ++tidx) {
        Eigen::Vector3i triangle = mesh->triangles_[tidx];
        if (triangle_count[triangle] == 1) {
            mesh->triangles_[to_idx] = triangle;
            to_idx++;
        }
    }
    mesh->triangles_.resize(to_idx);
    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done remove triangles within "
            "the mesh");

    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] remove duplicate triangles and "
            "unreferenced vertices");
    mesh->RemoveDuplicatedTriangles();
    mesh->RemoveUnreferencedVertices();
    utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done remove duplicate triangles "
            "and unreferenced vertices");

    auto tmesh = t::geometry::TriangleMesh::FromLegacy(*mesh);

    // use new object tmesh2 here even if some arrays share memory with tmesh.
    // We don't want to replace the blobs in tmesh.
    auto tmesh2 = t::geometry::vtkutils::ComputeNormals(
            tmesh, /*vertex_normals=*/true, /*face_normals=*/false,
            /*consistency=*/true, /*auto_orient_normals=*/true,
            /*splitting=*/false);

    mesh->vertices_ = core::eigen_converter::TensorToEigenVector3dVector(
            tmesh2.GetVertexPositions());
    mesh->triangles_ = core::eigen_converter::TensorToEigenVector3iVector(
            tmesh2.GetTriangleIndices());
    if (mesh->HasVertexColors()) {
        mesh->vertex_colors_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        tmesh2.GetVertexColors());
    }
    if (mesh->HasVertexNormals()) {
        mesh->vertex_normals_ =
                core::eigen_converter::TensorToEigenVector3dVector(
                        tmesh2.GetVertexNormals());
    }

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
