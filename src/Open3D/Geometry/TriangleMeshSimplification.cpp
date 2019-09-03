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

#include "Open3D/Geometry/TriangleMesh.h"

#include <Eigen/Dense>
#include <queue>
#include <tuple>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

/// Error quadric that is used to minimize the squared distance of a point to
/// its neigbhouring triangle planes.
/// Cf. "Simplifying Surfaces with Color and Texture using Quadric Error
/// Metrics" by Garland and Heckbert.
class Quadric {
public:
    Quadric() {
        A_.fill(0);
        b_.fill(0);
        c_ = 0;
    }

    Quadric(const Eigen::Vector4d& plane, double weight = 1) {
        Eigen::Vector3d n = plane.head<3>();
        A_ = weight * n * n.transpose();
        b_ = weight * plane(3) * n;
        c_ = weight * plane(3) * plane(3);
    }

    Quadric& operator+=(const Quadric& other) {
        A_ += other.A_;
        b_ += other.b_;
        c_ += other.c_;
        return *this;
    }

    Quadric operator+(const Quadric& other) const {
        Quadric res;
        res.A_ = A_ + other.A_;
        res.b_ = b_ + other.b_;
        res.c_ = c_ + other.c_;
        return res;
    }

    double Eval(const Eigen::Vector3d& v) const {
        Eigen::Vector3d Av = A_ * v;
        double q = v.dot(Av) + 2 * b_.dot(v) + c_;
        return q;
    }

    bool IsInvertible() const { return std::fabs(A_.determinant()) > 1e-4; }

    Eigen::Vector3d Minimum() const { return -A_.ldlt().solve(b_); }

public:
    /// A_ = n . n^T, where n is the plane normal
    Eigen::Matrix3d A_;
    /// b_ = d . n, where n is the plane normal and d the non-normal component
    /// of the plane parameters
    Eigen::Vector3d b_;
    /// c_ = d . d, where d the non-normal component pf the plane parameters
    double c_;
};

std::shared_ptr<TriangleMesh> TriangleMesh::SimplifyVertexClustering(
        double voxel_size,
        TriangleMesh::SimplificationContraction
                contraction /* = SimplificationContraction::Average */) const {
    auto mesh = std::make_shared<TriangleMesh>();
    if (voxel_size <= 0.0) {
        utility::LogWarning("[VoxelGridFromPointCloud] voxel_size <= 0.\n");
        return mesh;
    }

    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogWarning(
                "[VoxelGridFromPointCloud] voxel_size is too small.\n");
        return mesh;
    }

    auto GetVoxelIdx = [&](const Eigen::Vector3d& vert) {
        Eigen::Vector3d ref_coord = (vert - voxel_min_bound) / voxel_size;
        Eigen::Vector3i idx(int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                            int(floor(ref_coord(2))));
        return idx;
    };

    std::unordered_map<Eigen::Vector3i, std::unordered_set<int>,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxel_vertices;
    std::unordered_map<Eigen::Vector3i, int,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxel_vert_ind;
    int new_vidx = 0;
    for (size_t vidx = 0; vidx < vertices_.size(); ++vidx) {
        const Eigen::Vector3i vox_idx = GetVoxelIdx(vertices_[vidx]);
        voxel_vertices[vox_idx].insert(int(vidx));

        if (voxel_vert_ind.count(vox_idx) == 0) {
            voxel_vert_ind[vox_idx] = new_vidx;
            new_vidx++;
        }
    }

    // aggregate vertex info
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    mesh->vertices_.resize(voxel_vertices.size());
    if (has_vert_normal) {
        mesh->vertex_normals_.resize(voxel_vertices.size());
    }
    if (has_vert_color) {
        mesh->vertex_colors_.resize(voxel_vertices.size());
    }

    auto AvgVertex = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertices_[vidx];
        }
        aggr /= double(ind.size());
        return aggr;
    };
    auto AvgNormal = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertex_normals_[vidx];
        }
        aggr /= double(ind.size());
        return aggr;
    };
    auto AvgColor = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertex_colors_[vidx];
        }
        aggr /= double(ind.size());
        return aggr;
    };

    if (contraction == TriangleMesh::SimplificationContraction::Average) {
        for (const auto& voxel : voxel_vertices) {
            int vox_vidx = voxel_vert_ind[voxel.first];
            mesh->vertices_[vox_vidx] = AvgVertex(voxel.second);
            if (has_vert_normal) {
                mesh->vertex_normals_[vox_vidx] = AvgNormal(voxel.second);
            }
            if (has_vert_color) {
                mesh->vertex_colors_[vox_vidx] = AvgColor(voxel.second);
            }
        }
    } else if (contraction ==
               TriangleMesh::SimplificationContraction::Quadric) {
        // Map triangles
        std::unordered_map<int, std::unordered_set<int>> vert_to_triangles;
        for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
            vert_to_triangles[triangles_[tidx](0)].emplace(int(tidx));
            vert_to_triangles[triangles_[tidx](1)].emplace(int(tidx));
            vert_to_triangles[triangles_[tidx](2)].emplace(int(tidx));
        }

        for (const auto& voxel : voxel_vertices) {
            int vox_vidx = voxel_vert_ind[voxel.first];

            Quadric q;
            for (int vidx : voxel.second) {
                for (int tidx : vert_to_triangles[vidx]) {
                    Eigen::Vector4d p = GetTrianglePlane(tidx);
                    double area = GetTriangleArea(tidx);
                    q += Quadric(p, area);
                }
            }
            if (q.IsInvertible()) {
                Eigen::Vector3d v = q.Minimum();
                mesh->vertices_[vox_vidx] = v;
            } else {
                mesh->vertices_[vox_vidx] = AvgVertex(voxel.second);
            }

            if (has_vert_normal) {
                mesh->vertex_normals_[vox_vidx] = AvgNormal(voxel.second);
            }
            if (has_vert_color) {
                mesh->vertex_colors_[vox_vidx] = AvgColor(voxel.second);
            }
        }
    }

    //  connect vertices
    std::unordered_set<Eigen::Vector3i,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            triangles;
    for (const auto& triangle : triangles_) {
        int vidx0 = voxel_vert_ind[GetVoxelIdx(vertices_[triangle(0)])];
        int vidx1 = voxel_vert_ind[GetVoxelIdx(vertices_[triangle(1)])];
        int vidx2 = voxel_vert_ind[GetVoxelIdx(vertices_[triangle(2)])];

        // only connect if in different voxels
        if (vidx0 == vidx1 || vidx0 == vidx2 || vidx1 == vidx2) {
            continue;
        }

        // Note: there can be still double faces with different orientation
        // The user has to clean up manually
        if (vidx1 < vidx0 && vidx1 < vidx2) {
            int tmp = vidx0;
            vidx0 = vidx1;
            vidx1 = vidx2;
            vidx2 = tmp;
        } else if (vidx2 < vidx0 && vidx2 < vidx1) {
            int tmp = vidx1;
            vidx1 = vidx0;
            vidx0 = vidx2;
            vidx2 = tmp;
        }

        triangles.emplace(Eigen::Vector3i(vidx0, vidx1, vidx2));
    }

    mesh->triangles_.resize(triangles.size());
    int tidx = 0;
    for (const Eigen::Vector3i& triangle : triangles) {
        mesh->triangles_[tidx] = triangle;
        tidx++;
    }

    if (HasTriangleNormals()) {
        mesh->ComputeTriangleNormals();
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::SimplifyQuadricDecimation(
        int target_number_of_triangles) const {
    typedef std::tuple<double, int, int> CostEdge;

    auto mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_ = vertices_;
    mesh->vertex_normals_ = vertex_normals_;
    mesh->vertex_colors_ = vertex_colors_;
    mesh->triangles_ = triangles_;

    std::vector<bool> vertices_deleted(vertices_.size(), false);
    std::vector<bool> triangles_deleted(triangles_.size(), false);

    // Map vertices to triangles and compute triangle planes and areas
    std::vector<std::unordered_set<int>> vert_to_triangles(vertices_.size());
    std::vector<Eigen::Vector4d> triangle_planes(triangles_.size());
    std::vector<double> triangle_areas(triangles_.size());
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        vert_to_triangles[triangles_[tidx](0)].emplace(tidx);
        vert_to_triangles[triangles_[tidx](1)].emplace(tidx);
        vert_to_triangles[triangles_[tidx](2)].emplace(tidx);

        triangle_planes[tidx] = GetTrianglePlane(tidx);
        triangle_areas[tidx] = GetTriangleArea(tidx);
    }

    // Compute the error metric per vertex
    std::vector<Quadric> Qs(vertices_.size());
    for (size_t vidx = 0; vidx < vertices_.size(); ++vidx) {
        for (int tidx : vert_to_triangles[vidx]) {
            Qs[vidx] += Quadric(triangle_planes[tidx], triangle_areas[tidx]);
        }
    }

    // For boundary edges add perpendicular plane quadric
    auto edge_triangle_count = GetEdgeToTrianglesMap();
    auto AddPerpPlaneQuadric = [&](int vidx0, int vidx1, int vidx2,
                                   double area) {
        int min = std::min(vidx0, vidx1);
        int max = std::max(vidx0, vidx1);
        Eigen::Vector2i edge(min, max);
        if (edge_triangle_count[edge].size() != 1) {
            return;
        }
        const auto& vert0 = mesh->vertices_[vidx0];
        const auto& vert1 = mesh->vertices_[vidx1];
        const auto& vert2 = mesh->vertices_[vidx2];
        Eigen::Vector3d vert2p = (vert2 - vert0).cross(vert2 - vert1);
        Eigen::Vector4d plane = ComputeTrianglePlane(vert0, vert1, vert2p);
        Quadric quad(plane, area);
        Qs[vidx0] += quad;
        Qs[vidx1] += quad;
    };
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        const auto& tria = triangles_[tidx];
        double area = triangle_areas[tidx];
        AddPerpPlaneQuadric(tria(0), tria(1), tria(2), area);
        AddPerpPlaneQuadric(tria(1), tria(2), tria(0), area);
        AddPerpPlaneQuadric(tria(2), tria(0), tria(1), area);
    }

    // Get valid edges and compute cost
    // Note: We could also select all vertex pairs as edges with dist < eps
    std::unordered_map<Eigen::Vector2i, Eigen::Vector3d,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            vbars;
    std::unordered_map<Eigen::Vector2i, double,
                       utility::hash_eigen::hash<Eigen::Vector2i>>
            costs;
    auto CostEdgeComp = [](const CostEdge& a, const CostEdge& b) {
        return std::get<0>(a) > std::get<0>(b);
    };
    std::priority_queue<CostEdge, std::vector<CostEdge>, decltype(CostEdgeComp)>
            queue(CostEdgeComp);

    auto AddEdge = [&](int vidx0, int vidx1, bool update) {
        int min = std::min(vidx0, vidx1);
        int max = std::max(vidx0, vidx1);
        Eigen::Vector2i edge(min, max);
        if (update || vbars.count(edge) == 0) {
            const Quadric& Q0 = Qs[min];
            const Quadric& Q1 = Qs[max];
            Quadric Qbar = Q0 + Q1;
            double cost;
            Eigen::Vector3d vbar;
            if (Qbar.IsInvertible()) {
                vbar = Qbar.Minimum();
                cost = Qbar.Eval(vbar);
            } else {
                const Eigen::Vector3d& v0 = mesh->vertices_[vidx0];
                const Eigen::Vector3d& v1 = mesh->vertices_[vidx0];
                Eigen::Vector3d vmid = (v0 + v1) / 2;
                double cost0 = Qbar.Eval(v0);
                double cost1 = Qbar.Eval(v1);
                double costmid = Qbar.Eval(vbar);
                cost = std::min(cost0, std::min(cost1, costmid));
                if (cost == costmid) {
                    vbar = vmid;
                } else if (cost == cost0) {
                    vbar = v0;
                } else {
                    vbar = v1;
                }
            }
            vbars[edge] = vbar;
            costs[edge] = cost;
            queue.push(CostEdge(cost, min, max));
        }
    };

    // add all edges to priority queue
    for (const auto& triangle : triangles_) {
        AddEdge(triangle(0), triangle(1), false);
        AddEdge(triangle(1), triangle(2), false);
        AddEdge(triangle(2), triangle(0), false);
    }

    // perform incremental edge collapse
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    int n_triangles = int(triangles_.size());
    while (n_triangles > target_number_of_triangles && !queue.empty()) {
        // retrieve edge from queue
        double cost;
        int vidx0, vidx1;
        std::tie(cost, vidx0, vidx1) = queue.top();
        queue.pop();

        // test if the edge has been updated (reinserted into queue)
        Eigen::Vector2i edge(vidx0, vidx1);
        bool valid = !vertices_deleted[vidx0] && !vertices_deleted[vidx1] &&
                     cost == costs[edge];
        if (!valid) {
            continue;
        }

        // avoid flip of triangle normal
        bool flipped = false;
        for (int tidx : vert_to_triangles[vidx1]) {
            if (triangles_deleted[tidx]) {
                continue;
            }

            const Eigen::Vector3i& tria = mesh->triangles_[tidx];
            bool has_vidx0 =
                    vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
            bool has_vidx1 =
                    vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);
            if (has_vidx0 && has_vidx1) {
                continue;
            }

            Eigen::Vector3d vert0 = mesh->vertices_[tria(0)];
            Eigen::Vector3d vert1 = mesh->vertices_[tria(1)];
            Eigen::Vector3d vert2 = mesh->vertices_[tria(2)];
            Eigen::Vector3d norm_before = (vert1 - vert0).cross(vert2 - vert0);
            norm_before /= norm_before.norm();

            if (vidx1 == tria(0)) {
                vert0 = vbars[edge];
            } else if (vidx1 == tria(1)) {
                vert1 = vbars[edge];
            } else if (vidx1 == tria(2)) {
                vert2 = vbars[edge];
            }

            Eigen::Vector3d norm_after = (vert1 - vert0).cross(vert2 - vert0);
            norm_after /= norm_after.norm();
            if (norm_before.dot(norm_after) < 0) {
                flipped = true;
                break;
            }
        }
        if (flipped) {
            continue;
        }

        // Connect triangles from vidx1 to vidx0, or mark deleted
        for (int tidx : vert_to_triangles[vidx1]) {
            if (triangles_deleted[tidx]) {
                continue;
            }

            Eigen::Vector3i& tria = mesh->triangles_[tidx];
            bool has_vidx0 =
                    vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
            bool has_vidx1 =
                    vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);

            if (has_vidx0 && has_vidx1) {
                triangles_deleted[tidx] = true;
                n_triangles--;
                continue;
            }

            if (vidx1 == tria(0)) {
                tria(0) = vidx0;
            } else if (vidx1 == tria(1)) {
                tria(1) = vidx0;
            } else if (vidx1 == tria(2)) {
                tria(2) = vidx0;
            }
            vert_to_triangles[vidx0].insert(tidx);
        }

        // update vertex vidx0 to vbar
        mesh->vertices_[vidx0] = vbars[edge];
        Qs[vidx0] += Qs[vidx1];
        if (has_vert_normal) {
            mesh->vertex_normals_[vidx0] = 0.5 * (mesh->vertex_normals_[vidx0] +
                                                  mesh->vertex_normals_[vidx1]);
        }
        if (has_vert_color) {
            mesh->vertex_colors_[vidx0] = 0.5 * (mesh->vertex_colors_[vidx0] +
                                                 mesh->vertex_colors_[vidx1]);
        }
        vertices_deleted[vidx1] = true;

        // Update edge costs for all triangles connecting to vidx0
        for (const auto& tidx : vert_to_triangles[vidx0]) {
            if (triangles_deleted[tidx]) {
                continue;
            }
            const Eigen::Vector3i& tria = mesh->triangles_[tidx];
            if (tria(0) == vidx0 || tria(1) == vidx0) {
                AddEdge(tria(0), tria(1), true);
            }
            if (tria(1) == vidx0 || tria(2) == vidx0) {
                AddEdge(tria(1), tria(2), true);
            }
            if (tria(2) == vidx0 || tria(0) == vidx0) {
                AddEdge(tria(2), tria(0), true);
            }
        }
    }

    // Apply changes to the triangle mesh
    int next_free = 0;
    std::unordered_map<int, int> vert_remapping;
    for (size_t idx = 0; idx < mesh->vertices_.size(); ++idx) {
        if (!vertices_deleted[idx]) {
            vert_remapping[int(idx)] = next_free;
            mesh->vertices_[next_free] = mesh->vertices_[idx];
            if (has_vert_normal) {
                mesh->vertex_normals_[next_free] = mesh->vertex_normals_[idx];
            }
            if (has_vert_color) {
                mesh->vertex_colors_[next_free] = mesh->vertex_colors_[idx];
            }
            next_free++;
        }
    }
    mesh->vertices_.resize(next_free);
    if (has_vert_normal) {
        mesh->vertex_normals_.resize(next_free);
    }
    if (has_vert_color) {
        mesh->vertex_colors_.resize(next_free);
    }

    next_free = 0;
    for (size_t idx = 0; idx < mesh->triangles_.size(); ++idx) {
        if (!triangles_deleted[idx]) {
            Eigen::Vector3i tria = mesh->triangles_[idx];
            mesh->triangles_[next_free](0) = vert_remapping[tria(0)];
            mesh->triangles_[next_free](1) = vert_remapping[tria(1)];
            mesh->triangles_[next_free](2) = vert_remapping[tria(2)];
            next_free++;
        }
    }
    mesh->triangles_.resize(next_free);

    if (HasTriangleNormals()) {
        mesh->ComputeTriangleNormals();
    }

    return mesh;
}

}  // namespace geometry
}  // namespace open3d
