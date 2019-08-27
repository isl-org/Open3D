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

#include "Open3D/Geometry/Qhull.h"
#include "Open3D/Geometry/TetraMesh.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

#include "libqhullcpp/PointCoordinates.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"

namespace open3d {
namespace geometry {

std::shared_ptr<TriangleMesh> Qhull::ComputeConvexHull(
        const std::vector<Eigen::Vector3d>& points, std::vector<int> &pt_map) {
    auto convex_hull = std::make_shared<TriangleMesh>();

    std::vector<double> qhull_points_data(points.size() * 3);
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
        const auto& pt = points[pidx];
        qhull_points_data[pidx * 3 + 0] = pt(0);
        qhull_points_data[pidx * 3 + 1] = pt(1);
        qhull_points_data[pidx * 3 + 2] = pt(2);
    }

    orgQhull::PointCoordinates qhull_points(3, "");
    qhull_points.append(qhull_points_data);

    orgQhull::Qhull qhull;
    qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                   qhull_points.count(), qhull_points.coordinates(), "Qt");

    orgQhull::QhullFacetList facets = qhull.facetList();
    convex_hull->triangles_.resize(facets.count());
    std::unordered_map<int, int> vert_map;
    std::unordered_set<int> inserted_vertices;
    int tidx = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullFacet f = *it;
        orgQhull::QhullVertexSet vSet = f.vertices();
        int triangle_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt) {
            orgQhull::QhullVertex v = *vIt;
            orgQhull::QhullPoint p = v.point();

            int vidx = p.id();
            convex_hull->triangles_[tidx](triangle_subscript) = vidx;
            triangle_subscript++;

            if (inserted_vertices.count(vidx) == 0) {
                inserted_vertices.insert(vidx);
                vert_map[vidx] = int(convex_hull->vertices_.size());
                double* coords = p.coordinates();
                convex_hull->vertices_.push_back(
                        Eigen::Vector3d(coords[0], coords[1], coords[2]));
                pt_map.push_back(vidx);
            }
        }

        tidx++;
    }

    for (Eigen::Vector3i& triangle : convex_hull->triangles_) {
        triangle(0) = vert_map[triangle(0)];
        triangle(1) = vert_map[triangle(1)];
        triangle(2) = vert_map[triangle(2)];
    }

    return convex_hull;
}

std::shared_ptr<TetraMesh> Qhull::ComputeDelaunayTetrahedralization(
        const std::vector<Eigen::Vector3d>& points, std::vector<int> &pt_map) {
    typedef decltype(TetraMesh::tetras_)::value_type Vector4i;
    auto delaunay_triangulation = std::make_shared<TetraMesh>();

    if (points.size() < 4) {
        utility::LogWarning(
                "[ComputeDelaunayTriangulation3D] not enough points to create "
                "a tetrahedral mesh.\n");
        return delaunay_triangulation;
    }

    // qhull cannot deal with this case
    if (points.size() == 4) {
        delaunay_triangulation->vertices_ = points;
        delaunay_triangulation->tetras_.push_back(Vector4i(0, 1, 2, 3));
        return delaunay_triangulation;
    }

    std::vector<double> qhull_points_data(points.size() * 3);
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
        const auto& pt = points[pidx];
        qhull_points_data[pidx * 3 + 0] = pt(0);
        qhull_points_data[pidx * 3 + 1] = pt(1);
        qhull_points_data[pidx * 3 + 2] = pt(2);
    }

    orgQhull::PointCoordinates qhull_points(3, "");
    qhull_points.append(qhull_points_data);

    orgQhull::Qhull qhull;
    qhull.runQhull(qhull_points.comment().c_str(), qhull_points.dimension(),
                   qhull_points.count(), qhull_points.coordinates(),
                   "d Qbb Qt");

    orgQhull::QhullFacetList facets = qhull.facetList();
    delaunay_triangulation->tetras_.resize(facets.count());
    std::unordered_map<int, int> vert_map;
    std::unordered_set<int> inserted_vertices;
    int tidx = 0;
    for (orgQhull::QhullFacetList::iterator it = facets.begin();
         it != facets.end(); ++it) {
        if (!(*it).isGood()) continue;

        orgQhull::QhullFacet f = *it;
        orgQhull::QhullVertexSet vSet = f.vertices();
        int tetra_subscript = 0;
        for (orgQhull::QhullVertexSet::iterator vIt = vSet.begin();
             vIt != vSet.end(); ++vIt) {
            orgQhull::QhullVertex v = *vIt;
            orgQhull::QhullPoint p = v.point();

            int vidx = p.id();
            delaunay_triangulation->tetras_[tidx](tetra_subscript) = vidx;
            tetra_subscript++;

            if (inserted_vertices.count(vidx) == 0) {
                inserted_vertices.insert(vidx);
                vert_map[vidx] = int(delaunay_triangulation->vertices_.size());
                double* coords = p.coordinates();
                delaunay_triangulation->vertices_.push_back(
                        Eigen::Vector3d(coords[0], coords[1], coords[2]));
                pt_map.push_back(vidx);
            }
        }

        tidx++;
    }

    for (auto& tetra : delaunay_triangulation->tetras_) {
        tetra(0) = vert_map[tetra(0)];
        tetra(1) = vert_map[tetra(1)];
        tetra(2) = vert_map[tetra(2)];
        tetra(3) = vert_map[tetra(3)];
    }

    return delaunay_triangulation;
}

/// This is the implementation of the Hidden Point Removal operator described
/// in Katz et. al. 'Direct Visibility of Point Sets', 2007. Some additional
/// information about the choice of `radius` for noisy point clouds can be
/// found in Mehra et. al. 'Visibility of Noisy Point Cloud Data', 2010.
std::shared_ptr<TriangleMesh> Qhull::HiddenPointRemoval(
        const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d camera,
        double radius, std::vector<int> &pt_map) {

    if(radius <= 0)
    {
        utility::LogWarning(
            "[HiddenPointRemoval] radius must be larger than zero.\n");
        return std::make_shared<TriangleMesh>();
    }

    // perform spherical projection
    std::vector<Eigen::Vector3d> spherical_projection;
    for (size_t pidx = 0; pidx < points.size(); ++pidx) {
      Eigen::Vector3d projected_point = points[pidx] - camera;
        double norm = projected_point.norm();
        spherical_projection.push_back(projected_point + 2 *
            (radius - norm) * projected_point / norm);
    }

    // add origin
    size_t origin_pidx = spherical_projection.size();
    spherical_projection.push_back(Eigen::Vector3d(0, 0, 0));

    // calculate convex hull of spherical projection
    auto visible_mesh = Qhull::ComputeConvexHull(spherical_projection,
                                                 pt_map);

    // reassign original points to mesh
    int origin_vidx = pt_map.size();
    for (size_t vidx = 0; vidx < pt_map.size(); vidx++) {
        size_t pidx = pt_map[vidx];
        visible_mesh->vertices_[vidx] = points[pidx];
        if(pidx == origin_pidx) {
            origin_vidx = vidx;
            visible_mesh->vertices_[vidx] = camera;
        }
    }

    // erase origin if part of mesh
    if(origin_vidx < (int)(visible_mesh->vertices_.size())) {
        visible_mesh->vertices_.erase(visible_mesh->vertices_.begin() +
                                      origin_vidx);
        pt_map.erase(pt_map.begin() + origin_vidx);
        for(size_t tidx = visible_mesh->triangles_.size(); tidx-- > 0;){
            if(visible_mesh->triangles_[tidx](0) == origin_vidx ||
                    visible_mesh->triangles_[tidx](1) == origin_vidx ||
                    visible_mesh->triangles_[tidx](2) == origin_vidx) {
                visible_mesh->triangles_.erase(
                    visible_mesh->triangles_.begin() + tidx);
            }
            else {
                if(visible_mesh->triangles_[tidx](0) > origin_vidx)
                    visible_mesh->triangles_[tidx](0) -= 1;
                if(visible_mesh->triangles_[tidx](1) > origin_vidx)
                    visible_mesh->triangles_[tidx](1) -= 1;
                if(visible_mesh->triangles_[tidx](2) > origin_vidx)
                    visible_mesh->triangles_[tidx](2) -= 1;
            }
        }
    }
    return visible_mesh;
}

}  // namespace geometry
}  // namespace open3d
