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
        const std::vector<Eigen::Vector3d>& points) {
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
        const std::vector<Eigen::Vector3d>& points) {
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

}  // namespace geometry
}  // namespace open3d
