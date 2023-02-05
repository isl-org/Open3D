// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <memory>

#include "open3d/Macro.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class RaycastingScene
/// \brief A scene class with basic ray casting and closest point queries.
///
/// The RaycastingScene allows to compute ray intersections with triangle meshes
/// or compute the closest point on the surface of a mesh with respect to one
/// or more query points.
/// It builds an internal acceleration structure to speed up those queries.
///
/// This class supports only the CPU device.
class RaycastingScene {
public:
    /// \brief Default Constructor.
    RaycastingScene(int64_t nthreads = 0);

    ~RaycastingScene();

    /// \brief Add a triangle mesh to the scene.
    /// \param vertex_positions Vertices as Tensor of dim {N,3} and dtype float.
    /// \param triangle_indices Triangles as Tensor of dim {M,3} and dtype
    /// uint32_t. \return The geometry ID of the added mesh.
    uint32_t AddTriangles(const core::Tensor &vertex_positions,
                          const core::Tensor &triangle_indices);

    /// \brief Add a triangle mesh to the scene.
    /// \param mesh A triangle mesh.
    /// \return The geometry ID of the added mesh.
    uint32_t AddTriangles(const TriangleMesh &mesh);

    /// \brief Computes the first intersection of the rays with the scene.
    /// \param rays A tensor with >=2 dims, shape {.., 6}, and Dtype Float32
    /// describing the rays.
    /// {..} can be any number of dimensions, e.g., to organize rays for
    /// creating an image the shape can be {height, width, 6}.
    /// The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
    /// with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
    /// necessary to normalize the direction but the returned hit distance uses
    /// the length of the direction vector as unit.
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \return The returned dictionary contains:
    ///         - \b t_hit A tensor with the distance to the first hit. The
    ///           shape is {..}. If there is no intersection the hit distance
    ///           is \a inf .
    ///         - \b geometry_ids A tensor with the geometry IDs. The shape is
    ///           {..}. If there is no intersection the ID is \a INVALID_ID .
    ///         - \b primitive_ids A tensor with the primitive IDs, which
    ///           corresponds to the triangle index. The shape is {..}.
    ///           If there is no intersection the ID is \a INVALID_ID .
    ///         - \b primitive_uvs A tensor with the barycentric coordinates of
    ///           the hit points within the hit triangles. The shape is {.., 2}.
    ///         - \b primitive_normals A tensor with the normals of the hit
    ///           triangles. The shape is {.., 3}.
    std::unordered_map<std::string, core::Tensor> CastRays(
            const core::Tensor &rays, const int nthreads = 0);

    /// \brief Checks if the rays have any intersection with the scene.
    /// \param rays A tensor with >=2 dims, shape {.., 6}, and Dtype Float32
    /// describing the rays.
    /// {..} can be any number of dimensions, e.g., to organize rays for
    /// creating an image the shape can be {height, width, 6}.
    /// The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
    /// with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
    /// necessary to normalize the direction.
    /// \param tnear The tnear offset for the rays. The default is 0.
    /// \param tfar The tfar value for the ray. The default is infinity.
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \return A boolean tensor which indicates if the ray is occluded by the
    /// scene (true) or not (false).
    core::Tensor TestOcclusions(
            const core::Tensor &rays,
            const float tnear = 0.f,
            const float tfar = std::numeric_limits<float>::infinity(),
            const int nthreads = 0);

    /// \brief Computes the number of intersection of the rays with the scene.
    /// \param rays A tensor with >=2 dims, shape {.., 6}, and Dtype Float32
    /// describing the rays.
    /// {..} can be any number of dimensions, e.g., to organize rays for
    /// creating an image the shape can be {height, width, 6}.
    /// The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
    /// with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
    /// necessary to normalize the direction.
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \return A tensor with the number of intersections. The shape is {..}.
    core::Tensor CountIntersections(const core::Tensor &rays,
                                    const int nthreads = 0);

    /// \brief Computes the closest points on the surfaces of the scene.
    /// \param query_points A tensor with >=2 dims, shape {.., 3} and Dtype
    /// Float32 describing the query points. {..} can be any number of
    /// dimensions, e.g., to organize the query_point to create a 3D grid the
    /// shape can be {depth, height, width, 3}. The last dimension must be 3 and
    /// has the format [x, y, z].
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \return The returned dictionary contains:
    ///         - \b points A tensor with the closest surface points. The shape
    ///           is {..}.
    ///         - \b geometry_ids A tensor with the geometry IDs. The shape is
    ///           {..}.
    ///         - \b primitive_ids A tensor with the primitive IDs, which
    ///           corresponds to the triangle index. The shape is {..}.
    ///         - \b primitive_uvs A tensor with the barycentric coordinates of
    ///           the closest points within the triangles. The shape is {.., 2}.
    ///         - \b primitive_normals A tensor with the normals of the
    ///           closest triangle . The shape is {.., 3}.
    std::unordered_map<std::string, core::Tensor> ComputeClosestPoints(
            const core::Tensor &query_points, const int nthreads = 0);

    /// \brief Computes the distance to the surface of the scene.
    /// \param query_points A tensor with >=2 dims, shape {.., 3} and Dtype
    /// Float32 describing the query points. {..} can be any number of
    /// dimensions, e.g., to organize the query_point to create a 3D grid the
    /// shape can be {depth, height, width, 3}. The last dimension must be 3 and
    /// has the format [x, y, z].
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \return A tensor with the distances to the
    /// surface. The shape is {..}.
    core::Tensor ComputeDistance(const core::Tensor &query_points,
                                 const int nthreads = 0);

    /// \brief Computes the signed distance to the surface of the scene.
    ///
    /// This function computes the signed distance to the meshes in the scene.
    /// The function assumes that all meshes are watertight and that there are
    /// no intersections between meshes, i.e., inside and outside must be well
    /// defined. The function determines the sign of the distance by counting
    /// the intersections of a rays starting at the query points.
    ///
    /// \param query_points A tensor with >=2 dims, shape {.., 3} and Dtype
    /// Float32 describing the query points. {..} can be any number of
    /// dimensions, e.g., to organize the query_point to create a 3D grid the
    /// shape can be {depth, height, width, 3}. The last dimension must be 3 and
    /// has the format [x, y, z].
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \param nsamples The number of rays used for determining the inside.
    /// This must be an odd number. The default is 1. Use a higher value if you
    /// notice sign flipping, which can occur when rays hit exactly an edge or
    /// vertex in the scene.
    ///
    /// \return A tensor with the signed distances to
    /// the surface. The shape is
    /// {..}. Negative distances mean a point is inside a closed surface.
    core::Tensor ComputeSignedDistance(const core::Tensor &query_points,
                                       const int nthreads = 0,
                                       const int nsamples = 1);

    /// \brief Computes the occupancy at the query point positions.
    ///
    /// This function computes whether the query points are inside or outside.
    /// The function assumes that all meshes are watertight and that there are
    /// no intersections between meshes, i.e., inside and outside must be well
    /// defined. The function determines if a point is inside by counting the
    /// intersections of a rays starting at the query points.
    ///
    /// \param query_points A tensor with >=2 dims, shape {.., 3} and Dtype
    /// Float32 describing the query_points.
    /// {..} can be any number of dimensions, e.g., to
    /// organize the query_point to create a 3D grid the shape can be
    /// {depth, height, width, 3}.
    /// The last dimension must be 3 and has the format [x, y, z].
    /// \param nthreads The number of threads to use. Set to 0 for automatic.
    /// \param nsamples The number of rays used for determining the inside.
    /// This must be an odd number. The default is 1. Use a higher value if you
    /// notice errors in the occupancy values. Errors can occur when rays hit
    /// exactly an edge or vertex in the scene.
    ///
    /// \return A tensor with the occupancy values. The shape is {..}. Values
    /// are either 0 or 1. A point is occupied or inside if the value is 1.
    core::Tensor ComputeOccupancy(const core::Tensor &query_points,
                                  const int nthreads = 0,
                                  const int nsamples = 1);

    /// \brief Creates rays for the given camera parameters.
    ///
    /// \param intrinsic_matrix The upper triangular intrinsic matrix with
    /// shape {3,3}.
    /// \param extrinsic_matrix The 4x4 world to camera SE(3) transformation
    /// matrix. \param width_px The width of the image in pixels. \param
    /// height_px The height of the image in pixels. \return A tensor of shape
    /// {height_px, width_px, 6} with the rays.
    static core::Tensor CreateRaysPinhole(const core::Tensor &intrinsic_matrix,
                                          const core::Tensor &extrinsic_matrix,
                                          int width_px,
                                          int height_px);

    /// \brief Creates rays for the given camera parameters.
    ///
    /// \param fov_deg The horizontal field of view in degree.
    /// \param center The point the camera is looking at with shape {3}.
    /// \param eye The position of the camera with shape {3}.
    /// \param up The up-vector with shape {3}.
    /// \param width_px The width of the image in pixels.
    /// \param height_px The height of the image in pixels.
    /// \return A tensor of shape {height_px, width_px, 6} with the rays.
    static core::Tensor CreateRaysPinhole(double fov_deg,
                                          const core::Tensor &center,
                                          const core::Tensor &eye,
                                          const core::Tensor &up,
                                          int width_px,
                                          int height_px);

    /// \brief The value for invalid IDs.
    static uint32_t INVALID_ID();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
