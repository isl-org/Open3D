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

#include "open3d/t/geometry/RaycastingScene.h"
#include "pybind/core/tensor_type_caster.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_raycasting_scene(py::module& m) {
    py::class_<RaycastingScene> raycasting_scene(m, "RaycastingScene", R"doc(
A scene class with basic ray casting and closest point queries.

The RaycastingScene allows to compute ray intersections with triangle meshes
or compute the closest point on the surface of a mesh with respect to one
or more query points.
It builds an internal acceleration structure to speed up those queries.

This class supports only the CPU device.

The following shows how to create a scene and compute ray intersections::

    import open3d as o3d
    import matplotlib.pyplot as plt

    cube = o3d.t.geometry.TriangleMesh.from_legacy(
                                        o3d.geometry.TriangleMesh.create_box())

    # Create scene and add the cube mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)

    # Rays are 6D vectors with origin and ray direction.
    # Here we use a helper function to create rays for a pinhole camera.
    rays = scene.create_rays_pinhole(fov_deg=60,
                                     center=[0.5,0.5,0.5],
                                     eye=[-1,-1,-1],
                                     up=[0,0,1],
                                     width_px=320,
                                     height_px=240)

    # Compute the ray intersections.
    ans = scene.cast_rays(rays)

    # Visualize the hit distance (depth)
    plt.imshow(ans['t_hit'].numpy())

)doc");

    // Constructors.
    raycasting_scene.def(py::init<>());

    raycasting_scene.def(
            "add_triangles",
            py::overload_cast<const core::Tensor&, const core::Tensor&>(
                    &RaycastingScene::AddTriangles),
            "vertex_positions"_a, "triangle_indices"_a, R"doc(
Add a triangle mesh to the scene.

Args:
    vertices (open3d.core.Tensor): Vertices as Tensor of dim {N,3} and dtype
        Float32.
    triangles (open3d.core.Tensor): Triangles as Tensor of dim {M,3} and dtype
        UInt32.

Returns:
    The geometry ID of the added mesh.
)doc");

    raycasting_scene.def("add_triangles",
                         py::overload_cast<const TriangleMesh&>(
                                 &RaycastingScene::AddTriangles),
                         "mesh"_a, R"doc(
Add a triangle mesh to the scene.

Args:
    mesh (open3d.t.geometry.TriangleMesh): A triangle mesh.

Returns:
    The geometry ID of the added mesh.
)doc");

    raycasting_scene.def("cast_rays", &RaycastingScene::CastRays, "rays"_a,
                         "nthreads"_a = 0,
                         R"doc(
Computes the first intersection of the rays with the scene.

Args:
    rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype
        Float32 describing the rays.
        {..} can be any number of dimensions, e.g., to organize rays for
        creating an image the shape can be {height, width, 6}. The last
        dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
        with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is
        not necessary to normalize the direction but the returned hit distance
        uses the length of the direction vector as unit.

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A dictionary which contains the following keys

    t_hit
        A tensor with the distance to the first hit. The shape is {..}. If there
        is no intersection the hit distance is *inf*.

    geometry_ids
        A tensor with the geometry IDs. The shape is {..}. If there
        is no intersection the ID is *INVALID_ID*.

    primitive_ids
        A tensor with the primitive IDs, which corresponds to the triangle
        index. The shape is {..}.  If there is no intersection the ID is
        *INVALID_ID*.

    primitive_uvs
        A tensor with the barycentric coordinates of the hit points within the
        hit triangles. The shape is {.., 2}.

    primitive_normals
        A tensor with the normals of the hit triangles. The shape is {.., 3}.
)doc");

    raycasting_scene.def("test_occlusions", &RaycastingScene::TestOcclusions,
                         "rays"_a, "tnear"_a = 0.f,
                         "tfar"_a = std::numeric_limits<float>::infinity(),
                         "nthreads"_a = 0,
                         R"doc(
Checks if the rays have any intersection with the scene.

Args:
    rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype
        Float32 describing the rays.
        {..} can be any number of dimensions, e.g., to organize rays for
        creating an image the shape can be {height, width, 6}.
        The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
        with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
        necessary to normalize the direction.

    tnear (float): The tnear offset for the rays. The default is 0.

    tfar (float): The tfar value for the ray. The default is infinity.

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A boolean tensor which indicates if the ray is occluded by the scene (true)
    or not (false).
)doc");

    raycasting_scene.def("count_intersections",
                         &RaycastingScene::CountIntersections, "rays"_a,
                         "nthreads"_a = 0, R"doc(
Computes the number of intersection of the rays with the scene.

Args:
    rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype
        Float32 describing the rays.
        {..} can be any number of dimensions, e.g., to organize rays for
        creating an image the shape can be {height, width, 6}.
        The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
        with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
        necessary to normalize the direction.

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A tensor with the number of intersections. The shape is {..}.
)doc");

    raycasting_scene.def("compute_closest_points",
                         &RaycastingScene::ComputeClosestPoints,
                         "query_points"_a, "nthreads"_a = 0, R"doc(
Computes the closest points on the surfaces of the scene.

Args:
    query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3},
        and Dtype Float32 describing the query points.
        {..} can be any number of dimensions, e.g., to organize the query_point
        to create a 3D grid the shape can be {depth, height, width, 3}.
        The last dimension must be 3 and has the format [x, y, z].

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    The returned dictionary contains

    points
        A tensor with the closest surface points. The shape is {..}.

    geometry_ids
        A tensor with the geometry IDs. The shape is {..}.

    primitive_ids
        A tensor with the primitive IDs, which corresponds to the triangle
        index. The shape is {..}.

)doc");

    raycasting_scene.def("compute_distance", &RaycastingScene::ComputeDistance,
                         "query_points"_a, "nthreads"_a = 0, R"doc(
Computes the distance to the surface of the scene.

Args:
    query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3},
        and Dtype Float32 describing the query points.
        {..} can be any number of dimensions, e.g., to organize the
        query points to create a 3D grid the shape can be
        {depth, height, width, 3}.
        The last dimension must be 3 and has the format [x, y, z].

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A tensor with the distances to the surface. The shape is {..}.
)doc");

    raycasting_scene.def("compute_signed_distance",
                         &RaycastingScene::ComputeSignedDistance,
                         "query_points"_a, "nthreads"_a = 0, R"doc(
Computes the signed distance to the surface of the scene.

This function computes the signed distance to the meshes in the scene.
The function assumes that all meshes are watertight and that there are
no intersections between meshes, i.e., inside and outside must be well
defined. The function determines the sign of the distance by counting
the intersections of a rays starting at the query points.

Args:
    query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3},
        and Dtype Float32 describing the query_points.
        {..} can be any number of dimensions, e.g., to organize the
        query points to create a 3D grid the shape can be
        {depth, height, width, 3}.
        The last dimension must be 3 and has the format [x, y, z].

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A tensor with the signed distances to the surface. The shape is {..}.
    Negative distances mean a point is inside a closed surface.
)doc");

    raycasting_scene.def("compute_occupancy",
                         &RaycastingScene::ComputeOccupancy, "query_points"_a,
                         "nthreads"_a = 0,
                         R"doc(
Computes the occupancy at the query point positions.

This function computes whether the query points are inside or outside.
The function assumes that all meshes are watertight and that there are
no intersections between meshes, i.e., inside and outside must be well
defined. The function determines if a point is inside by counting the
intersections of a rays starting at the query points.

Args:
    query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3},
        and Dtype Float32 describing the query points.
        {..} can be any number of dimensions, e.g., to organize the
        query points to create a 3D grid the shape can be
        {depth, height, width, 3}.
        The last dimension must be 3 and has the format [x, y, z].

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    A tensor with the occupancy values. The shape is {..}. Values are either 0
    or 1. A point is occupied or inside if the value is 1.
)doc");

    raycasting_scene.def_static(
            "create_rays_pinhole",
            py::overload_cast<const core::Tensor&, const core::Tensor&, int,
                              int>(&RaycastingScene::CreateRaysPinhole),
            "intrinsic_matrix"_a, "extrinsic_matrix"_a, "width_px"_a,
            "height_px"_a, R"doc(
Creates rays for the given camera parameters.

Args:
    intrinsic_matrix (open3d.core.Tensor): The upper triangular intrinsic matrix
        with shape {3,3}.
    extrinsic_matrix (open3d.core.Tensor): The 4x4 world to camera SE(3)
        transformation matrix.
    width_px (int): The width of the image in pixels.
    height_px (int): The height of the image in pixels.

Returns:
    A tensor of shape {height_px, width_px, 6} with the rays.
)doc");

    raycasting_scene.def_static(
            "create_rays_pinhole",
            py::overload_cast<double, const core::Tensor&, const core::Tensor&,
                              const core::Tensor&, int, int>(
                    &RaycastingScene::CreateRaysPinhole),
            "fov_deg"_a, "center"_a, "eye"_a, "up"_a, "width_px"_a,
            "height_px"_a, R"doc(
Creates rays for the given camera parameters.

Args:
    fov_deg (float): The horizontal field of view in degree.
    center (open3d.core.Tensor): The point the camera is looking at with shape
        {3}.
    eye (open3d.core.Tensor): The position of the camera with shape {3}.
    up (open3d.core.Tensor): The up-vector with shape {3}.
    width_px (int): The width of the image in pixels.
    height_px (int): The height of the image in pixels.

Returns:
    A tensor of shape {height_px, width_px, 6} with the rays.
)doc");

    raycasting_scene.def_property_readonly_static(
            "INVALID_ID",
            [](py::object /* self */) -> uint32_t {
                return RaycastingScene::INVALID_ID();
            },
            R"doc(
The value for invalid IDs
)doc");
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
