// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <Eigen/Dense>

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TetraMesh.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
namespace geometry {

std::shared_ptr<LineSet> LineSet::CreateFromPointCloudCorrespondences(
        const PointCloud &cloud0,
        const PointCloud &cloud1,
        const std::vector<std::pair<int, int>> &correspondences) {
    auto lineset_ptr = std::make_shared<LineSet>();
    size_t point0_size = cloud0.points_.size();
    size_t point1_size = cloud1.points_.size();
    lineset_ptr->points_.resize(point0_size + point1_size);
    for (size_t i = 0; i < point0_size; i++)
        lineset_ptr->points_[i] = cloud0.points_[i];
    for (size_t i = 0; i < point1_size; i++)
        lineset_ptr->points_[point0_size + i] = cloud1.points_[i];

    size_t corr_size = correspondences.size();
    lineset_ptr->lines_.resize(corr_size);
    for (size_t i = 0; i < corr_size; i++)
        lineset_ptr->lines_[i] =
                Eigen::Vector2i(correspondences[i].first,
                                point0_size + correspondences[i].second);
    return lineset_ptr;
}

std::shared_ptr<LineSet> LineSet::CreateFromTriangleMesh(
        const TriangleMesh &mesh) {
    auto line_set = std::make_shared<LineSet>();
    line_set->points_ = mesh.vertices_;

    std::unordered_set<Eigen::Vector2i,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            inserted_edges;
    auto InsertEdge = [&](int vidx0, int vidx1) {
        Eigen::Vector2i edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            line_set->lines_.push_back(Eigen::Vector2i(vidx0, vidx1));
        }
    };
    for (const auto &triangle : mesh.triangles_) {
        InsertEdge(triangle(0), triangle(1));
        InsertEdge(triangle(1), triangle(2));
        InsertEdge(triangle(2), triangle(0));
    }

    return line_set;
}

std::shared_ptr<LineSet> LineSet::CreateFromOrientedBoundingBox(
        const OrientedBoundingBox &box) {
    auto line_set = std::make_shared<LineSet>();
    line_set->points_ = box.GetBoxPoints();
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->PaintUniformColor(box.color_);
    return line_set;
}

std::shared_ptr<LineSet> LineSet::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox &box) {
    auto line_set = std::make_shared<LineSet>();
    line_set->points_ = box.GetBoxPoints();
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->PaintUniformColor(box.color_);
    return line_set;
}

std::shared_ptr<LineSet> LineSet::CreateFromTetraMesh(const TetraMesh &mesh) {
    auto line_set = std::make_shared<LineSet>();
    line_set->points_ = mesh.vertices_;

    std::unordered_set<Eigen::Vector2i,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            inserted_edges;
    auto InsertEdge = [&](int vidx0, int vidx1) {
        Eigen::Vector2i edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            line_set->lines_.push_back(Eigen::Vector2i(vidx0, vidx1));
        }
    };
    for (const auto &tetra : mesh.tetras_) {
        InsertEdge(tetra(0), tetra(1));
        InsertEdge(tetra(1), tetra(2));
        InsertEdge(tetra(2), tetra(0));
        InsertEdge(tetra(3), tetra(0));
        InsertEdge(tetra(3), tetra(1));
        InsertEdge(tetra(3), tetra(2));
    }

    return line_set;
}

}  // namespace geometry
}  // namespace open3d
