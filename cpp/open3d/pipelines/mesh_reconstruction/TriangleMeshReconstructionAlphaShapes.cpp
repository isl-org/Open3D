// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/mesh_reconstruction/TriangleMeshReconstruction.h"

#include "open3d/geometry/Qhull.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace pipelines {
namespace mesh_reconstruction {

std::shared_ptr<geometry::TriangleMesh> ReconstructAlphaShape(
        const geometry::PointCloud& pcd,
        double alpha,
        std::shared_ptr<geometry::TetraMesh> tetra_mesh,
        std::vector<size_t>* pt_map) {
    std::vector<size_t> pt_map_computed;
    if (tetra_mesh == nullptr) {
        utility::LogDebug(
                "[ReconstructAlphaShape] "
                "ComputeDelaunayTetrahedralization");
        std::tie(tetra_mesh, pt_map_computed) =
                geometry::Qhull::ComputeDelaunayTetrahedralization(pcd.points_);
        pt_map = &pt_map_computed;
        utility::LogDebug(
                "[ReconstructAlphaShape] done "
                "ComputeDelaunayTetrahedralization");
    }

    utility::LogDebug("[ReconstructAlphaShape] init triangle mesh");
    auto mesh = std::make_shared<geometry::TriangleMesh>();
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
    utility::LogDebug("[ReconstructAlphaShape] done init triangle mesh");

    std::vector<double> vsqn(tetra_mesh->vertices_.size());
    for (size_t vidx = 0; vidx < vsqn.size(); ++vidx) {
        vsqn[vidx] = tetra_mesh->vertices_[vidx].squaredNorm();
    }

    utility::LogDebug(
            "[ReconstructAlphaShape] add triangles from tetras that "
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
            utility::LogError(
                    "[ReconstructAlphaShape] invalid tetra in "
                    "TetraMesh");
        }
        double r = std::sqrt(dx * dx + dy * dy + dz * dz - 4 * a * c) /
                   (2 * std::abs(a));

        if (r <= alpha) {
            mesh->triangles_.push_back(
                    GetOrderedTriangle(tetra(0), tetra(1), tetra(2)));
            mesh->triangles_.push_back(
                    GetOrderedTriangle(tetra(0), tetra(1), tetra(3)));
            mesh->triangles_.push_back(
                    GetOrderedTriangle(tetra(0), tetra(2), tetra(3)));
            mesh->triangles_.push_back(
                    GetOrderedTriangle(tetra(1), tetra(2), tetra(3)));
        }
    }
    utility::LogDebug(
            "[ReconstructAlphaShape] done add triangles from tetras "
            "that satisfy constraint");

    utility::LogDebug(
            "[ReconstructAlphaShape] remove triangles within "
            "the mesh");
    std::unordered_map<Eigen::Vector3i, int,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
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
            "[ReconstructAlphaShape] done remove triangles within "
            "the mesh");

    utility::LogDebug(
            "[ReconstructAlphaShape] remove duplicate triangles and "
            "unreferenced vertices");
    mesh->RemoveDuplicatedTriangles();
    mesh->RemoveUnreferencedVertices();
    utility::LogDebug(
            "[ReconstructAlphaShape] done remove duplicate triangles "
            "and unreferenced vertices");

    return mesh;
}

}  // namespace mesh_reconstruction
}  // namespace pipelines
}  // namespace open3d
