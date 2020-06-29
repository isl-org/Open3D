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

#pragma once

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace mesh_reconstruction {

inline Eigen::Vector3i GetOrderedTriangle(int vidx0, int vidx1, int vidx2) {
    if (vidx0 > vidx2) {
        std::swap(vidx0, vidx2);
    }
    if (vidx0 > vidx1) {
        std::swap(vidx0, vidx1);
    }
    if (vidx1 > vidx2) {
        std::swap(vidx1, vidx2);
    }
    return Eigen::Vector3i(vidx0, vidx1, vidx2);
}

/// \brief Alpha shapes are a generalization of the convex hull. With
/// decreasing alpha value the shape schrinks and creates cavities.
/// See Edelsbrunner and Muecke, "Three-Dimensional Alpha Shapes", 1994.
/// \param pcd PointCloud for what the alpha shape should be computed.
/// \param alpha parameter to control the shape. A very big value will
/// give a shape close to the convex hull.
/// \param tetra_mesh If not a nullptr, then uses this to construct the
/// alpha shape. Otherwise, ComputeDelaunayTetrahedralization is called.
/// \param pt_map Optional map from tetra_mesh vertex indices to pcd
/// points.
/// \return TriangleMesh of the alpha shape.
std::shared_ptr<geometry::TriangleMesh> ReconstructAlphaShape(
        const geometry::PointCloud &pcd,
        double alpha,
        std::shared_ptr<geometry::TetraMesh> tetra_mesh = nullptr,
        std::vector<size_t> *pt_map = nullptr);

/// Function that computes a triangle mesh from an oriented PointCloud \p
/// pcd. This implements the Ball Pivoting algorithm proposed in F.
/// Bernardini et al., "The ball-pivoting algorithm for surface
/// reconstruction", 1999. The implementation is also based on the
/// algorithms outlined in Digne, "An Analysis and Implementation of a
/// Parallel Ball Pivoting Algorithm", 2014. The surface reconstruction is
/// done by rolling a ball with a given radius (cf. \p radii) over the
/// point cloud, whenever the ball touches three points a triangle is
/// created.
/// \param pcd defines the PointCloud from which the TriangleMesh surface is
/// reconstructed. Has to contain normals.
/// \param radii defines the radii of
/// the ball that are used for the surface reconstruction.
std::shared_ptr<geometry::TriangleMesh> ReconstructBallPivoting(
        const geometry::PointCloud &pcd, const std::vector<double> &radii);

/// \brief Function that computes a triangle mesh from an oriented
/// PointCloud pcd. This implements the Screened Poisson Reconstruction
/// proposed in Kazhdan and Hoppe, "Screened Poisson Surface
/// Reconstruction", 2013. This function uses the original implementation by
/// Kazhdan. See https://github.com/mkazhdan/PoissonRecon
///
/// \param pcd PointCloud with normals and optionally colors.
/// \param depth Maximum depth of the tree that will be used for surface
/// reconstruction. Running at depth d corresponds to solving on a grid
/// whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
/// reconstructor adapts the octree to the sampling density, the specified
/// reconstruction depth is only an upper bound.
/// \param width Specifies the
/// target width of the finest level octree cells. This parameter is ignored
/// if depth is specified.
/// \param scale Specifies the ratio between the
/// diameter of the cube used for reconstruction and the diameter of the
/// samples' bounding cube. \param linear_fit If true, the reconstructor use
/// linear interpolation to estimate the positions of iso-vertices.
/// \return The estimated TriangleMesh, and per vertex densitie values that
/// can be used to to trim the mesh.
std::tuple<std::shared_ptr<geometry::TriangleMesh>, std::vector<double>>
ReconstructPoisson(const geometry::PointCloud &pcd,
                   size_t depth = 8,
                   size_t width = 0,
                   float scale = 1.1f,
                   bool linear_fit = false);

}  // namespace mesh_reconstruction
}  // namespace pipelines
}  // namespace open3d
