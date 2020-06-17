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

#include "Open3D/Geometry/TetraMesh.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"

#include <Eigen/Dense>
#include <array>
#include <numeric>
#include <tuple>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

TetraMesh &TetraMesh::Clear() {
    MeshBase::Clear();
    tetras_.clear();
    return *this;
}

TetraMesh &TetraMesh::operator+=(const TetraMesh &mesh) {
    typedef decltype(tetras_)::value_type Vector4i;
    if (mesh.IsEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    size_t old_tetra_num = tetras_.size();
    size_t add_tetra_num = mesh.tetras_.size();
    MeshBase::operator+=(mesh);
    tetras_.resize(tetras_.size() + mesh.tetras_.size());
    Vector4i index_shift = Vector4i::Constant(static_cast<int>(old_vert_num));
    for (size_t i = 0; i < add_tetra_num; i++) {
        tetras_[old_tetra_num + i] = mesh.tetras_[i] + index_shift;
    }
    return (*this);
}

TetraMesh TetraMesh::operator+(const TetraMesh &mesh) const {
    return (TetraMesh(*this) += mesh);
}

TetraMesh &TetraMesh::RemoveDuplicatedVertices() {
    typedef decltype(tetras_)::value_type::Scalar Index;
    typedef std::tuple<double, double, double> Coordinate3;
    std::unordered_map<Coordinate3, size_t,
                       utility::hash_tuple::hash<Coordinate3>>
            point_to_old_index;
    std::vector<Index> index_old_to_new(vertices_.size());
    size_t old_vertex_num = vertices_.size();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        Coordinate3 coord = std::make_tuple(vertices_[i](0), vertices_[i](1),
                                            vertices_[i](2));
        if (point_to_old_index.find(coord) == point_to_old_index.end()) {
            point_to_old_index[coord] = i;
            vertices_[k] = vertices_[i];
            index_old_to_new[i] = (Index)k;
            k++;
        } else {
            index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
        }
    }
    vertices_.resize(k);
    if (k < old_vertex_num) {
        for (auto &tetra : tetras_) {
            tetra(0) = index_old_to_new[tetra(0)];
            tetra(1) = index_old_to_new[tetra(1)];
            tetra(2) = index_old_to_new[tetra(2)];
            tetra(3) = index_old_to_new[tetra(3)];
        }
    }
    utility::LogDebug(
            "[RemoveDuplicatedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TetraMesh &TetraMesh::RemoveDuplicatedTetras() {
    typedef decltype(tetras_)::value_type::Scalar Index;
    typedef std::tuple<Index, Index, Index, Index> Index4;
    std::unordered_map<Index4, size_t, utility::hash_tuple::hash<Index4>>
            tetra_to_old_index;
    size_t old_tetra_num = tetras_.size();
    size_t k = 0;
    for (size_t i = 0; i < old_tetra_num; i++) {
        Index4 index;
        std::array<Index, 4> t{tetras_[i](0), tetras_[i](1), tetras_[i](2),
                               tetras_[i](3)};

        // We sort the indices to find duplicates, because tetra (0-1-2-3)
        // and tetra (2-0-3-1) are the same.
        std::sort(t.begin(), t.end());
        index = std::make_tuple(t[0], t[1], t[2], t[3]);

        if (tetra_to_old_index.find(index) == tetra_to_old_index.end()) {
            tetra_to_old_index[index] = i;
            tetras_[k] = tetras_[i];
            k++;
        }
    }
    tetras_.resize(k);
    utility::LogDebug("[RemoveDuplicatedTetras] {:d} tetras have been removed.",
                      (int)(old_tetra_num - k));

    return *this;
}

TetraMesh &TetraMesh::RemoveUnreferencedVertices() {
    typedef decltype(tetras_)::value_type::Scalar Index;
    std::vector<bool> vertex_has_reference(vertices_.size(), false);
    for (const auto &tetra : tetras_) {
        vertex_has_reference[tetra(0)] = true;
        vertex_has_reference[tetra(1)] = true;
        vertex_has_reference[tetra(2)] = true;
        vertex_has_reference[tetra(3)] = true;
    }
    std::vector<Index> index_old_to_new(vertices_.size());
    size_t old_vertex_num = vertices_.size();
    size_t k = 0;                                  // new index
    for (size_t i = 0; i < old_vertex_num; i++) {  // old index
        if (vertex_has_reference[i]) {
            vertices_[k] = vertices_[i];
            index_old_to_new[i] = (Index)k;
            k++;
        } else {
            index_old_to_new[i] = -1;
        }
    }
    vertices_.resize(k);
    if (k < old_vertex_num) {
        for (auto &tetra : tetras_) {
            tetra(0) = index_old_to_new[tetra(0)];
            tetra(1) = index_old_to_new[tetra(1)];
            tetra(2) = index_old_to_new[tetra(2)];
            tetra(3) = index_old_to_new[tetra(3)];
        }
    }
    utility::LogDebug(
            "[RemoveUnreferencedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TetraMesh &TetraMesh::RemoveDegenerateTetras() {
    size_t old_tetra_num = tetras_.size();
    size_t k = 0;
    for (size_t i = 0; i < old_tetra_num; i++) {
        const auto &tetra = tetras_[i];
        if (tetra(0) != tetra(1) && tetra(0) != tetra(2) &&
            tetra(0) != tetra(3) && tetra(1) != tetra(2) &&
            tetra(1) != tetra(3) && tetra(2) != tetra(3)) {
            tetras_[k] = tetras_[i];
            k++;
        }
    }
    tetras_.resize(k);
    utility::LogDebug(
            "[RemoveDegenerateTetras] {:d} tetras have been "
            "removed.",
            (int)(old_tetra_num - k));
    return *this;
}

std::shared_ptr<TriangleMesh> TetraMesh::ExtractTriangleMesh(
        const std::vector<double> &values, double level) {
    typedef decltype(tetras_)::value_type::Scalar Index;
    static_assert(std::is_signed<Index>(), "Index type must be signed");
    typedef std::tuple<Index, Index> Index2;

    auto triangle_mesh = std::make_shared<TriangleMesh>();

    if (values.size() != vertices_.size()) {
        utility::LogError(
                "[ExtractTriangleMesh] number of values does not match the "
                "number of vertices.");
    }

    auto SurfaceIntersectionTest = [](double v0, double v1, double level) {
        return (v0 < level && v1 >= level) || (v0 >= level && v1 < level);
    };

    auto ComputeEdgeVertex = [&values, level, this](Index idx1, Index idx2) {
        double v1 = values[idx1];
        double v2 = values[idx2];

        double t = (level - v2) / (v1 - v2);
        if (std::isnan(t) || t < 0 || t > 1) {
            t = 0.5;
        }
        Eigen::Vector3d intersection =
                t * vertices_[idx1] + (1 - t) * vertices_[idx2];

        return intersection;
    };

    auto ComputeTriangleNormal = [](const Eigen::Vector3d &a,
                                    const Eigen::Vector3d &b,
                                    const Eigen::Vector3d &c) {
        Eigen::Vector3d ab = b - a;
        Eigen::Vector3d ac = c - a;
        return ab.cross(ac);
    };

    auto MakeSortedTuple2 = [](Index a, Index b) {
        if (a < b)
            return std::make_tuple(a, b);
        else
            return std::make_tuple(b, a);
    };

    auto HasCommonVertexIndex = [](Index2 a, Index2 b) {
        return std::get<0>(b) == std::get<0>(a) ||
               std::get<1>(b) == std::get<0>(a) ||
               std::get<0>(b) == std::get<1>(a) ||
               std::get<1>(b) == std::get<1>(a);
    };

    std::unordered_map<Index2, size_t, utility::hash_tuple::hash<Index2>>
            intersecting_edges;

    const int tetra_edges[][2] = {{0, 1}, {0, 2}, {0, 3},
                                  {1, 2}, {1, 3}, {2, 3}};

    for (size_t tetra_i = 0; tetra_i < tetras_.size(); ++tetra_i) {
        const auto &tetra = tetras_[tetra_i];

        std::array<Eigen::Vector3d, 4> verts;
        std::array<Index2, 4> keys;  // keys for the edges
        std::array<Index, 4> verts_indices;
        std::array<Eigen::Vector3d, 4> edge_dirs;
        int num_verts = 0;

        for (int tet_edge_i = 0; tet_edge_i < 6; ++tet_edge_i) {
            Index edge_vert1 = tetra[tetra_edges[tet_edge_i][0]];
            Index edge_vert2 = tetra[tetra_edges[tet_edge_i][1]];
            double vert_value1 = values[edge_vert1];
            double vert_value2 = values[edge_vert2];
            if (SurfaceIntersectionTest(vert_value1, vert_value2, level)) {
                Index2 index = MakeSortedTuple2(edge_vert1, edge_vert2);
                verts[num_verts] = ComputeEdgeVertex(edge_vert1, edge_vert2);
                keys[num_verts] = index;

                // make edge_vert1 be the vertex that is smaller than level
                // (inside)
                if (values[edge_vert1] > values[edge_vert2])
                    std::swap(edge_vert1, edge_vert2);
                edge_dirs[num_verts] =
                        vertices_[edge_vert2] - vertices_[edge_vert1];
                ++num_verts;
            }
        }

        // add vertices and get the vertex indices
        for (int i = 0; i < num_verts; ++i) {
            if (intersecting_edges.count(keys[i]) == 0) {
                Index idx = static_cast<Index>(intersecting_edges.size());
                verts_indices[i] = idx;
                intersecting_edges[keys[i]] = idx;
                triangle_mesh->vertices_.push_back(verts[i]);
            } else {
                verts_indices[i] = Index(intersecting_edges[keys[i]]);
            }
        }

        // create triangles for this tetrahedron
        if (3 == num_verts) {
            Eigen::Vector3i tri(verts_indices[0], verts_indices[1],
                                verts_indices[2]);

            Eigen::Vector3d tri_normal =
                    ComputeTriangleNormal(verts[0], verts[1], verts[2]);

            // accumulate to improve robustness of the triangle orientation test
            double dot = 0;
            for (int i = 0; i < 3; ++i) dot += tri_normal.dot(edge_dirs[i]);
            if (dot < 0) std::swap(tri.x(), tri.y());

            triangle_mesh->triangles_.push_back(tri);
        } else if (4 == num_verts) {
            std::array<int, 4> order = {-1, 0, 0, 0};
            if (HasCommonVertexIndex(keys[0], keys[1]) &&
                HasCommonVertexIndex(keys[0], keys[2])) {
                order = {1, 0, 2, 3};
            } else if (HasCommonVertexIndex(keys[0], keys[1]) &&
                       HasCommonVertexIndex(keys[0], keys[3])) {
                order = {1, 0, 3, 2};
            } else if (HasCommonVertexIndex(keys[0], keys[2]) &&
                       HasCommonVertexIndex(keys[0], keys[3])) {
                order = {2, 0, 3, 1};
            }

            if (order[0] != -1) {
                // accumulate to improve robustness of the triangle orientation
                // test
                double dot = 0;
                for (int i = 0; i < 4; ++i) {
                    Eigen::Vector3d tri_normal = ComputeTriangleNormal(
                            verts[order[(4 + i - 1) % 4]], verts[order[i]],
                            verts[order[(i + 1) % 4]]);
                    dot += tri_normal.dot(edge_dirs[order[i]]);
                }
                if (dot < 0) std::reverse(order.begin(), order.end());

                std::array<Eigen::Vector3i, 2> tris;
                if ((verts[order[0]] - verts[order[2]]).squaredNorm() <
                    (verts[order[1]] - verts[order[3]]).squaredNorm()) {
                    tris[0] << verts_indices[order[0]], verts_indices[order[1]],
                            verts_indices[order[2]];
                    tris[1] << verts_indices[order[2]], verts_indices[order[3]],
                            verts_indices[order[0]];
                } else {
                    tris[0] << verts_indices[order[0]], verts_indices[order[1]],
                            verts_indices[order[3]];
                    tris[1] << verts_indices[order[1]], verts_indices[order[2]],
                            verts_indices[order[3]];
                }

                triangle_mesh->triangles_.insert(
                        triangle_mesh->triangles_.end(), {tris[0], tris[1]});
            } else {
                utility::LogWarning(
                        "[ExtractTriangleMesh] failed to create triangles for "
                        "tetrahedron {:d}: invalid edge configuration for "
                        "tetrahedron",
                        int(tetra_i));
            }
        } else if (0 != num_verts) {
            utility::LogWarning(
                    "[ExtractTriangleMesh] failed to create triangles for "
                    "tetrahedron {:d}: unexpected number of vertices {:d}",
                    int(tetra_i), num_verts);
        }
    }

    return triangle_mesh;
}

}  // namespace geometry
}  // namespace open3d
