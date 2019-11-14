// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/Qhull.h"
#include "Open3D/Geometry/TetraMesh.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

#include <Eigen/Dense>

#include <iostream>
#include <list>

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> TriangleMesh::CreateFromPointCloudAlphaShape(
        const PointCloud& pcd, double alpha) {
    std::shared_ptr<TetraMesh> tetra_mesh;
    std::vector<size_t> pt_map;
    std::tie(tetra_mesh, pt_map) =
            Qhull::ComputeDelaunayTetrahedralization(pcd.points_);

    auto mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_ = tetra_mesh->vertices_;
    utility::LogInfo("{}, {}, {}", pcd.points_.size(),
                     tetra_mesh->vertices_.size(), pt_map.size());
    // if (pcd.HasNormals()) {
    // for(size_t idx = 0; idx < pt_map.size(); ++idx) {
    //     mesh->vertex_normals_.push_back(pcd.normals_[]);
    // }
    // }
    // if (pcd.HasColors()) {
    //     mesh->vertex_colors_ = pcd.colors_;
    // }

    std::vector<double> vsqn(tetra_mesh->vertices_.size());
    for (size_t vidx = 0; vidx < vsqn.size(); ++vidx) {
        vsqn[vidx] = tetra_mesh->vertices_[vidx].squaredNorm();
    }

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
        double r = std::sqrt(dx * dx + dy * dy + dz * dz - 4 * a * c) /
                   (2 * std::abs(a));

        if (r <= alpha) {
            mesh->triangles_.push_back(
                    Eigen::Vector3i(tetra(0), tetra(1), tetra(2)));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(tetra(0), tetra(1), tetra(3)));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(tetra(0), tetra(2), tetra(3)));
            mesh->triangles_.push_back(
                    Eigen::Vector3i(tetra(1), tetra(2), tetra(3)));
        }
    }

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

    mesh->RemoveDuplicatedTriangles();
    mesh->RemoveUnreferencedVertices();

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
