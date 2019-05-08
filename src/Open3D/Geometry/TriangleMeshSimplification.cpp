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

struct Quadric {
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    double c;

    Quadric() {
        A.fill(0);
        b.fill(0);
        c = 0;
    }

    Quadric(const Eigen::Vector4d& plane, double weight = 1) {
        Eigen::Vector3d n = plane.head<3>();
        A = weight * n * n.transpose();
        b = weight * plane(3) * n;
        c = weight * plane(3) * plane(3);
    }

    Quadric& operator+=(const Quadric& other) {
        A += other.A;
        b += other.b;
        c += other.c;
        return *this;
    }

    Quadric operator+(const Quadric& other) const {
        Quadric res;
        res.A = A + other.A;
        res.b = b + other.b;
        res.c = c + other.c;
        return res;
    }

    double Eval(const Eigen::Vector3d& v) const {
        Eigen::Vector3d Av = A * v;
        double q = v.dot(Av) + 2 * b.dot(v) + c;
        return q;
    }

    bool IsInvertible() const { return abs(A.determinant()) > 1e-4; }

    Eigen::Vector3d Minimum() const { return -A.ldlt().solve(b); }
};

void TriangleMesh::SimplifyVertexClustering(
        double voxel_size,
        SimplificationContraction
                contraction /* = SimplificationContraction::Average */) {
    if (voxel_size <= 0.0) {
        utility::PrintWarning("[VoxelGridFromPointCloud] voxel_size <= 0.\n");
        return;
    }

    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::PrintWarning(
                "[VoxelGridFromPointCloud] voxel_size is too small.\n");
        return;
    }

    auto get_voxel_index = [&](const Eigen::Vector3d& vert) {
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
        const Eigen::Vector3i vox_idx = get_voxel_index(vertices_[vidx]);
        voxel_vertices[vox_idx].insert(vidx);

        if (voxel_vert_ind.count(vox_idx) == 0) {
            voxel_vert_ind[vox_idx] = new_vidx;
            new_vidx++;
        }
    }

    // aggregate vertex info
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    std::vector<Eigen::Vector3d> vertices(voxel_vertices.size());
    std::vector<Eigen::Vector3d> vertex_normals;
    if (has_vert_normal) {
        vertex_normals.resize(voxel_vertices.size());
    }
    std::vector<Eigen::Vector3d> vertex_colors;
    if (has_vert_color) {
        vertex_colors.resize(voxel_vertices.size());
    }

    auto avg_vertex_fcn = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertices_[vidx];
        }
        aggr /= ind.size();
        return aggr;
    };
    auto avg_normal_fcn = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertex_normals_[vidx];
        }
        aggr /= ind.size();
        return aggr;
    };
    auto avg_color_fcn = [&](const std::unordered_set<int> ind) {
        Eigen::Vector3d aggr(0, 0, 0);
        for (int vidx : ind) {
            aggr += vertex_colors_[vidx];
        }
        aggr /= ind.size();
        return aggr;
    };

    if (contraction == SimplificationContraction::Average) {
        for (const auto& voxel : voxel_vertices) {
            int vox_vidx = voxel_vert_ind[voxel.first];
            vertices[vox_vidx] = avg_vertex_fcn(voxel.second);
            if (has_vert_normal) {
                vertex_normals[vox_vidx] = avg_normal_fcn(voxel.second);
            }
            if (has_vert_color) {
                vertex_colors[vox_vidx] = avg_color_fcn(voxel.second);
            }
        }
    } else if (contraction == SimplificationContraction::Quadric) {
        // Map triangles
        std::unordered_map<int, std::unordered_set<int>> vert_to_triangles;
        int next_tidx = 0;
        for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
            vert_to_triangles[triangles_[tidx](0)].emplace(tidx);
            vert_to_triangles[triangles_[tidx](1)].emplace(tidx);
            vert_to_triangles[triangles_[tidx](2)].emplace(tidx);
        }

        for (const auto& voxel : voxel_vertices) {
            int vox_vidx = voxel_vert_ind[voxel.first];

            Quadric q;
            for (int vidx : voxel.second) {
                for (int tidx : vert_to_triangles[vidx]) {
                    Eigen::Vector4d p = TrianglePlane(tidx);
                    double area = TriangleArea(tidx);
                    q += Quadric(p, area);
                }
            }
            if (q.IsInvertible()) {
                Eigen::Vector3d v = q.Minimum();
                vertices[vox_vidx] = v;
            } else {
                vertices[vox_vidx] = avg_vertex_fcn(voxel.second);
            }

            if (has_vert_normal) {
                vertex_normals[vox_vidx] = avg_normal_fcn(voxel.second);
            }
            if (has_vert_color) {
                vertex_colors[vox_vidx] = avg_color_fcn(voxel.second);
            }
        }
    }

    //  connect vertices
    std::unordered_set<Eigen::Vector3i,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            triangles;
    for (const auto& triangle : triangles_) {
        int vidx0 = voxel_vert_ind[get_voxel_index(vertices_[triangle(0)])];
        int vidx1 = voxel_vert_ind[get_voxel_index(vertices_[triangle(1)])];
        int vidx2 = voxel_vert_ind[get_voxel_index(vertices_[triangle(2)])];

        // only connect if in different voxels
        if (vidx0 == vidx1 || vidx0 == vidx2 || vidx1 == vidx2) {
            continue;
        }

        // Note: there can be still double faces with different orientation
        // The use has to clean up manually
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

    triangles_.resize(triangles.size());
    int tidx = 0;
    for (Eigen::Vector3i triangle : triangles) {
        triangles_[tidx] = triangle;
        tidx++;
    }

    // set simplified properties to this
    vertices_ = vertices;
    vertex_normals_ = vertex_normals;
    vertex_colors_ = vertex_colors;
}

void TriangleMesh::SimplifyQuadricDecimation(int target_number_of_triangles) {
    typedef std::tuple<double, int, int> CostEdge;

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

        triangle_planes[tidx] = TrianglePlane(tidx);
        triangle_areas[tidx] = TriangleArea(tidx);
    }

    // Compute the error metric per vertex
    std::vector<Quadric> Qs(vertices_.size());
    for (size_t vidx = 0; vidx < vertices_.size(); ++vidx) {
        for (int tidx : vert_to_triangles[vidx]) {
            Qs[vidx] += Quadric(triangle_planes[tidx], triangle_areas[tidx]);
        }
    }

    // For boundary edges add perpendicular plane quadric
    auto edge_triangle_count = EdgeTriangleCount();
    auto add_perp_plan_quadric = [&](int vidx0, int vidx1, int vidx2,
                                     double area) {
        int min = std::min(vidx0, vidx1);
        int max = std::max(vidx0, vidx1);
        Edge edge(min, max);
        if (edge_triangle_count[edge] != 1) {
            return;
        }
        const auto& vert0 = vertices_[vidx0];
        const auto& vert1 = vertices_[vidx1];
        const auto& vert2 = vertices_[vidx2];
        Eigen::Vector3d vert2p = (vert2 - vert0).cross(vert2 - vert1);
        Eigen::Vector4d plane = TrianglePlane(vert0, vert1, vert2p);
        Quadric quad(plane, area);
        Qs[vidx0] += quad;
        Qs[vidx1] += quad;
    };
    for (size_t tidx = 0; tidx < triangles_.size(); ++tidx) {
        const auto& tria = triangles_[tidx];
        double area = triangle_areas[tidx];
        add_perp_plan_quadric(tria(0), tria(1), tria(2), area);
        add_perp_plan_quadric(tria(1), tria(2), tria(0), area);
        add_perp_plan_quadric(tria(2), tria(0), tria(1), area);
    }

    // Get valid edges and compute cost
    // Note: We could also select all vertex pairs as edges with dist < eps
    std::unordered_map<Edge, Eigen::Vector3d, utility::hash_tuple::hash<Edge>>
            vbars;
    std::unordered_map<Edge, double, utility::hash_tuple::hash<Edge>> costs;
    auto cost_edge_comp = [](const CostEdge& a, const CostEdge& b) {
        return std::get<0>(a) > std::get<0>(b);
    };
    std::priority_queue<CostEdge, std::vector<CostEdge>,
                        decltype(cost_edge_comp)>
            queue(cost_edge_comp);

    auto add_edge = [&](int vidx0, int vidx1, bool update) {
        int min = std::min(vidx0, vidx1);
        int max = std::max(vidx0, vidx1);
        Edge edge(min, max);
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
                const Eigen::Vector3d& v0 = vertices_[vidx0];
                const Eigen::Vector3d& v1 = vertices_[vidx0];
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
        add_edge(triangle(0), triangle(1), false);
        add_edge(triangle(1), triangle(2), false);
        add_edge(triangle(2), triangle(0), false);
    }

    // perform incremental edge collapse
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    int n_triangles = triangles_.size();
    while (n_triangles > target_number_of_triangles && !queue.empty()) {
        // retrieve edge from queue
        double cost;
        int vidx0, vidx1;
        std::tie(cost, vidx0, vidx1) = queue.top();
        queue.pop();

        // test if the edge has been updated (reinserted into queue)
        Edge edge(vidx0, vidx1);
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

            Eigen::Vector3i tria = triangles_[tidx];
            bool has_vidx0 =
                    vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
            bool has_vidx1 =
                    vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);
            if (has_vidx0 && has_vidx1) {
                continue;
            }

            Eigen::Vector3d vert0 = vertices_[tria(0)];
            Eigen::Vector3d vert1 = vertices_[tria(1)];
            Eigen::Vector3d vert2 = vertices_[tria(2)];
            Eigen::Vector3d norm_before = (vert2 - vert0).cross(vert2 - vert1);
            norm_before /= norm_before.norm();

            if (vidx1 == tria(0)) {
                vert0 = vbars[edge];
            } else if (vidx1 == tria(1)) {
                vert1 = vbars[edge];
            } else if (vidx1 == tria(2)) {
                vert2 = vbars[edge];
            }

            Eigen::Vector3d norm_after = (vert2 - vert0).cross(vert2 - vert1);
            norm_after /= norm_after.norm();
            if (norm_before.dot(norm_before) < 0) {
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

            Eigen::Vector3i& tria = triangles_[tidx];
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
        vertices_[vidx0] = vbars[edge];
        Qs[vidx0] += Qs[vidx1];
        if (has_vert_normal) {
            vertex_normals_[vidx0] =
                    0.5 * (vertex_normals_[vidx0] + vertex_normals_[vidx1]);
        }
        if (has_vert_color) {
            vertex_colors_[vidx0] =
                    0.5 * (vertex_colors_[vidx0] + vertex_colors_[vidx1]);
        }
        vertices_deleted[vidx1] = true;

        // Update edge costs for all triangles connecting to vidx0
        for (const auto& tidx : vert_to_triangles[vidx0]) {
            if (triangles_deleted[tidx]) {
                continue;
            }
            const Eigen::Vector3i& tria = triangles_[tidx];
            if (tria(0) == vidx0 || tria(1) == vidx0) {
                add_edge(tria(0), tria(1), true);
            }
            if (tria(1) == vidx0 || tria(2) == vidx0) {
                add_edge(tria(1), tria(2), true);
            }
            if (tria(2) == vidx0 || tria(0) == vidx0) {
                add_edge(tria(2), tria(0), true);
            }
        }
    }

    // Apply changes to the triangle mesh
    int next_free = 0;
    std::unordered_map<int, int> vert_remapping;
    for (size_t idx = 0; idx < vertices_.size(); ++idx) {
        if (!vertices_deleted[idx]) {
            vert_remapping[idx] = next_free;
            vertices_[next_free] = vertices_[idx];
            if (has_vert_normal) {
                vertex_normals_[next_free] = vertex_normals_[idx];
            }
            if (has_vert_color) {
                vertex_colors_[next_free] = vertex_colors_[idx];
            }
            next_free++;
        }
    }
    vertices_.resize(next_free);
    if (has_vert_normal) {
        vertex_normals_.resize(next_free);
    }
    if (has_vert_color) {
        vertex_colors_.resize(next_free);
    }

    next_free = 0;
    for (size_t idx = 0; idx < triangles_.size(); ++idx) {
        if (!triangles_deleted[idx]) {
            Eigen::Vector3i tria = triangles_[idx];
            triangles_[next_free](0) = vert_remapping[tria(0)];
            triangles_[next_free](1) = vert_remapping[tria(1)];
            triangles_[next_free](2) = vert_remapping[tria(2)];
            next_free++;
        }
    }
    triangles_.resize(next_free);
}

}  // namespace geometry
}  // namespace open3d
