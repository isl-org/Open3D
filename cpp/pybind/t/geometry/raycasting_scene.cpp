// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    raycasting_scene.def(py::init<int64_t>(), "nthreads"_a = 0, R"doc(
Create a RaycastingScene.

Args:
    nthreads (int): The number of threads to use for building the scene. Set to 0 for automatic.
)doc");

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

    raycasting_scene.def("list_intersections",
                         &RaycastingScene::ListIntersections, "rays"_a,
                         "nthreads"_a = 0, R"doc(
Lists the intersections of the rays with the scene::

    import open3d as o3d
    import numpy as np

    # Create scene and add the monkey model.
    scene = o3d.t.geometry.RaycastingScene()
    d = o3d.data.MonkeyModel()
    mesh = o3d.t.io.read_triangle_mesh(d.path)
    mesh_id = scene.add_triangles(mesh)

    # Create a grid of rays covering the bounding box
    bb_min = mesh.vertex['positions'].min(dim=0).numpy()
    bb_max = mesh.vertex['positions'].max(dim=0).numpy()
    x,y = np.linspace(bb_min, bb_max, num=10)[:,:2].T
    xv, yv = np.meshgrid(x,y)
    orig = np.stack([xv, yv, np.full_like(xv, bb_min[2]-1)], axis=-1).reshape(-1,3)
    dest = orig + np.full(orig.shape, (0,0,2+bb_max[2]-bb_min[2]),dtype=np.float32)
    rays = np.concatenate([orig, dest-orig], axis=-1).astype(np.float32)

    # Compute the ray intersections.
    lx = scene.list_intersections(rays)
    lx = {k:v.numpy() for k,v in lx.items()}

    # Calculate intersection coordinates using the primitive uvs and the mesh
    v = mesh.vertex['positions'].numpy()
    t = mesh.triangle['indices'].numpy()
    tidx = lx['primitive_ids']
    uv = lx['primitive_uvs']
    w = 1 - np.sum(uv, axis=1)
    c = \
    v[t[tidx, 1].flatten(), :] * uv[:, 0][:, None] + \
    v[t[tidx, 2].flatten(), :] * uv[:, 1][:, None] + \
    v[t[tidx, 0].flatten(), :] * w[:, None]

    # Calculate intersection coordinates using ray_ids
    c = rays[lx['ray_ids']][:,:3] + rays[lx['ray_ids']][:,3:]*lx['t_hit'][...,None]
                                    
    # Visualize the rays and intersections.
    lines = o3d.t.geometry.LineSet()
    lines.point.positions = np.hstack([orig,dest]).reshape(-1,3)
    lines.line.indices = np.arange(lines.point.positions.shape[0]).reshape(-1,2)
    lines.line.colors = np.full((lines.line.indices.shape[0],3), (1,0,0))
    x = o3d.t.geometry.PointCloud(positions=c)
    o3d.visualization.draw([mesh, lines, x], point_size=8)


Args:
    rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype
        Float32 describing the rays; {..} can be any number of dimensions.
        The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz]
        with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not
        necessary to normalize the direction although it should be normalised if 
        t_hit is to be calculated in coordinate units.

    nthreads (int): The number of threads to use. Set to 0 for automatic.

Returns:
    The returned dictionary contains
    
    ray_splits
        A tensor with ray intersection splits. Can be used to iterate over all intersections for each ray. The shape is {num_rays + 1}.
    
    ray_ids
        A tensor with ray IDs. The shape is {num_intersections}.
        
    t_hit
        A tensor with the distance to the hit. The shape is {num_intersections}. 
        
    geometry_ids
        A tensor with the geometry IDs. The shape is {num_intersections}.

    primitive_ids
        A tensor with the primitive IDs, which corresponds to the triangle
        index. The shape is {num_intersections}.
        
    primitive_uvs 
        A tensor with the barycentric coordinates of the intersection points within 
        the triangles. The shape is {num_intersections, 2}.
    
        
An example of using ray_splits::

    ray_splits: [0, 2, 3, 6, 6, 8] # note that the length of this is num_rays+1
    t_hit: [t1, t2, t3, t4, t5, t6, t7, t8]

    for ray_id, (start, end) in enumerate(zip(ray_splits[:-1], ray_splits[1:])):
        for i,t in enumerate(t_hit[start:end]):
            print(f'ray {ray_id}, intersection {i} at {t}')
            
        
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

    primitive_uvs 
        A tensor with the barycentric coordinates of the closest points within 
        the triangles. The shape is {.., 2}.

    primitive_normals 
        A tensor with the normals of the closest triangle . The shape is 
        {.., 3}.

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

    raycasting_scene.def(
            "compute_signed_distance", &RaycastingScene::ComputeSignedDistance,
            "query_points"_a, "nthreads"_a = 0, "nsamples"_a = 1, R"doc(
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

    nsamples (int): The number of rays used for determining the inside.
        This must be an odd number. The default is 1. Use a higher value if you
        notice sign flipping, which can occur when rays hit exactly an edge or 
        vertex in the scene.

Returns:
    A tensor with the signed distances to the surface. The shape is {..}.
    Negative distances mean a point is inside a closed surface.
)doc");

    raycasting_scene.def("compute_occupancy",
                         &RaycastingScene::ComputeOccupancy, "query_points"_a,
                         "nthreads"_a = 0, "nsamples"_a = 1,
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

    nsamples (int): The number of rays used for determining the inside.
        This must be an odd number. The default is 1. Use a higher value if you
        notice errors in the occupancy values. Errors can occur when rays hit
        exactly an edge or vertex in the scene.

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