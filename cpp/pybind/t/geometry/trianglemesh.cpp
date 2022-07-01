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

#include "open3d/t/geometry/TriangleMesh.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_trianglemesh(py::module& m) {
    py::class_<TriangleMesh, PyGeometry<TriangleMesh>,
               std::shared_ptr<TriangleMesh>, Geometry, DrawableGeometry>
            triangle_mesh(m, "TriangleMesh",
                          R"(
A triangle mesh contains vertices and triangles. The triangle mesh class stores
the attribute data in key-value maps. There are two maps: the vertex attributes
map, and the triangle attribute map.

The attributes of the triangle mesh have different levels::

    import open3d as o3d

    device = o3d.core.Device("CPU:0")
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32

    # Create an empty triangle mesh
    # Use mesh.vertex to access the vertices' attributes
    # Use mesh.triangle to access the triangles' attributes
    mesh = o3d.t.geometry.TriangleMesh(device)

    # Default attribute: vertex["positions"], triangle["indices"]
    # These attributes is created by default and is required by all triangle
    # meshes. The shape of both must be (N, 3). The device of "positions"
    # determines the device of the triangle mesh.
    mesh.vertex["positions"] = o3d.core.Tensor([[0, 0, 0],
                                                [0, 0, 1],
                                                [0, 1, 0],
                                                [0, 1, 1]], dtype_f, device)
    mesh.triangle["indices"] = o3d.core.Tensor([[0, 1, 2],
                                                [0, 2, 3]]], dtype_i, device)

    # Common attributes: vertex["colors"]  , vertex["normals"]
    #                    triangle["colors"], triangle["normals"]
    # Common attributes are used in built-in triangle mesh operations. The
    # spellings must be correct. For example, if "normal" is used instead of
    # "normals", some internal operations that expects "normals" will not work.
    # "normals" and "colors" must have shape (N, 3) and must be on the same
    # device as the triangle mesh.
    mesh.vertex["normals"] = o3d.core.Tensor([[0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0],
                                              [1, 1, 1]], dtype_f, device)
    mesh.vertex["colors"] = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                             [0.1, 0.1, 0.1],
                                             [0.2, 0.2, 0.2],
                                             [0.3, 0.3, 0.3]], dtype_f, device)
    mesh.triangle["normals"] = o3d.core.Tensor(...)
    mesh.triangle["colors"] = o3d.core.Tensor(...)

    # User-defined attributes
    # You can also attach custom attributes. The value tensor must be on the
    # same device as the triangle mesh. The are no restrictions on the shape and
    # dtype, e.g.,
    pcd.vertex["labels"] = o3d.core.Tensor(...)
    pcd.triangle["features"] = o3d.core.Tensor(...)
)");

    // Constructors.
    triangle_mesh
            .def(py::init<const core::Device&>(),
                 "Construct an empty trianglemesh on the provided ``device`` "
                 "(default: 'CPU:0').",
                 "device"_a = core::Device("CPU:0"))
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "vertex_positions"_a, "triangle_indices"_a)
            .def("__repr__", &TriangleMesh::ToString);

    // Triangle mesh's attributes: vertices, vertex_colors, vertex_normals, etc.
    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    triangle_mesh.def_property_readonly(
            "vertex",
            py::overload_cast<>(&TriangleMesh::GetVertexAttr, py::const_));
    triangle_mesh.def_property_readonly(
            "triangle",
            py::overload_cast<>(&TriangleMesh::GetTriangleAttr, py::const_));

    // Device transfers.
    triangle_mesh.def("to", &TriangleMesh::To,
                      "Transfer the triangle mesh to a specified device.",
                      "device"_a, "copy"_a = false);
    triangle_mesh.def("clone", &TriangleMesh::Clone,
                      "Returns copy of the triangle mesh on the same device.");

    triangle_mesh.def(
            "cpu",
            [](const TriangleMesh& triangle_mesh) {
                return triangle_mesh.To(core::Device("CPU:0"));
            },
            "Transfer the triangle mesh to CPU. If the triangle mesh "
            "is already on CPU, no copy will be performed.");
    triangle_mesh.def(
            "cuda",
            [](const TriangleMesh& triangle_mesh, int device_id) {
                return triangle_mesh.To(core::Device("CUDA", device_id));
            },
            "Transfer the triangle mesh to a CUDA device. If the triangle mesh "
            "is already on the specified CUDA device, no copy will be "
            "performed.",
            "device_id"_a = 0);

    // Triangle Mesh's specific functions.
    triangle_mesh.def("get_min_bound", &TriangleMesh::GetMinBound,
                      "Returns the min bound for point coordinates.");
    triangle_mesh.def("get_max_bound", &TriangleMesh::GetMaxBound,
                      "Returns the max bound for point coordinates.");
    triangle_mesh.def("get_center", &TriangleMesh::GetCenter,
                      "Returns the center for point coordinates.");
    triangle_mesh.def("transform", &TriangleMesh::Transform, "transformation"_a,
                      "Transforms the points and normals (if exist).");
    triangle_mesh.def("translate", &TriangleMesh::Translate, "translation"_a,
                      "relative"_a = true, "Translates points.");
    triangle_mesh.def("scale", &TriangleMesh::Scale, "scale"_a, "center"_a,
                      "Scale points.");
    triangle_mesh.def("rotate", &TriangleMesh::Rotate, "R"_a, "center"_a,
                      "Rotate points and normals (if exist).");

    triangle_mesh.def(
            "compute_convex_hull", &TriangleMesh::ComputeConvexHull,
            "joggle_inputs"_a = false,
            R"(Compute the convex hull of a point cloud using qhull. This runs on the CPU.

Args:
    joggle_inputs (default False). Handle precision problems by
    randomly perturbing the input data. Set to True if perturbing the input
    iis acceptable but you need convex simplicial output. If False,
    neighboring facets may be merged in case of precision problems. See
    `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle`__ for more
    details.

Returns:
    TriangleMesh representing the convexh hull. This contains an
    extra vertex property "point_indices" that contains the index of the
    corresponding vertex in the original mesh.

Example:
    We will load the Stanford Bunny dataset, compute and display it's convex hull::

        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        hull = mesh.compute_convex_hull()
        o3d.visualization.draw([{'name': 'bunny', 'geometry': mesh}, {'name': 'convex hull', 'geometry': hull}])
)");

    // creation
    triangle_mesh.def_static(
            "from_legacy", &TriangleMesh::FromLegacy, "mesh_legacy"_a,
            "vertex_dtype"_a = core::Float32, "triangle_dtype"_a = core::Int64,
            "device"_a = core::Device("CPU:0"),
            "Create a TriangleMesh from a legacy Open3D TriangleMesh.");
    // conversion
    triangle_mesh.def("to_legacy", &TriangleMesh::ToLegacy,
                      "Convert to a legacy Open3D TriangleMesh.");

    triangle_mesh.def("clip_plane", &TriangleMesh::ClipPlane, "point"_a,
                      "normal"_a,
                      R"(Returns a new triangle mesh clipped with the plane.

This method clips the triangle mesh with the specified plane.
Parts of the mesh on the positive side of the plane will be kept and triangles
intersected by the plane will be cut.

Args:
    point (open3d.core.Tensor): A point on the plane.

    normal (open3d.core.Tensor): The normal of the plane. The normal points to
        the positive side of the plane for which the geometry will be kept.

Returns:
    New triangle mesh clipped with the plane.


This example shows how to create a hemisphere from a sphere::

    import open3d as o3d

    sphere = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_sphere())
    hemisphere = sphere.clip_plane(point=[0,0,0], normal=[1,0,0])

    o3d.visualization.draw(hemisphere)
)");

    // Triangle Mesh's creation APIs.
    triangle_mesh.def_static(
            "create_box", &TriangleMesh::CreateBox,
            "Create a box triangle mesh. One vertex of the box"
            "will be placed at the origin and the box aligns"
            "with the positive x, y, and z axes."
            "width"_a = 1.0,
            "height"_a = 1.0, "depth"_a = 1.0, "float_dtype"_a = core::Float32,
            "int_dtype"_a = core::Int64, "device"_a = core::Device("CPU:0"));

    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_box",
            {{"width", "x-directional length."},
             {"height", "y-directional length."},
             {"depth", "z-directional length."},
             {"vertex_dtype", "Float_dtype, Float32 or Float64."},
             {"triangle_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create mesh."}});

    triangle_mesh.def(
            "simplify_quadric_decimation",
            &TriangleMesh::SimplifyQuadricDecimation, "target_reduction"_a,
            "preserve_volume"_a = true,
            R"(Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert.
    
This function always uses the CPU device.

Args:
    target_reduction (float): The factor of triangles to delete, i.e., setting
        this to 0.9 will return a mesh with about 10% of the original triangle
        count. It is not guaranteed that the target reduction factor will be
        reached.

    preserve_volume (bool): If set to True this enables volume preservation
        which reduces the error in triangle normal direction.

Returns:
    Simplified TriangleMesh.

Example:
    This shows how to simplifify the Stanford Bunny mesh::

        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        simplified = mesh.simplify_quadric_decimation(0.99)
        o3d.visualization.draw([{'name': 'bunny', 'geometry': simplified}])
)");

    triangle_mesh.def(
            "boolean_union", &TriangleMesh::BooleanUnion, "mesh"_a,
            "tolerance"_a = 1e-6,
            R"(Computes the mesh that encompasses the union of the volumes of two meshes.
Both meshes should be manifold.

This function always uses the CPU device.

Args:
    mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the 
        boolean operation.

    tolerance (float): Threshold which determines when point distances are
        considered to be 0.

Returns:
    The mesh describing the union volume.

Example:
    This copmutes the union of a sphere and a cube::

        box = o3d.geometry.TriangleMesh.create_box()
        box = o3d.t.geometry.TriangleMesh.from_legacy(box)
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
        sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

        ans = box.boolean_union(sphere)

        o3d.visualization.draw([{'name': 'union', 'geometry': ans}])
)");

    triangle_mesh.def(
            "boolean_intersection", &TriangleMesh::BooleanIntersection,
            "mesh"_a, "tolerance"_a = 1e-6,
            R"(Computes the mesh that encompasses the intersection of the volumes of two meshes.
Both meshes should be manifold.

This function always uses the CPU device.

Args:
    mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the 
        boolean operation.

    tolerance (float): Threshold which determines when point distances are
        considered to be 0.

Returns:
    The mesh describing the intersection volume.

Example:
    This copmutes the intersection of a sphere and a cube::

        box = o3d.geometry.TriangleMesh.create_box()
        box = o3d.t.geometry.TriangleMesh.from_legacy(box)
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
        sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

        ans = box.boolean_intersection(sphere)

        o3d.visualization.draw([{'name': 'intersection', 'geometry': ans}])
)");

    triangle_mesh.def(
            "boolean_difference", &TriangleMesh::BooleanDifference, "mesh"_a,
            "tolerance"_a = 1e-6,
            R"(Computes the mesh that encompasses the volume after subtracting the volume of the second operand.
Both meshes should be manifold.

This function always uses the CPU device.

Args:
    mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the 
        boolean operation.

    tolerance (float): Threshold which determines when point distances are
        considered to be 0.

Returns:
    The mesh describing the difference volume.

Example:
    This subtracts the sphere from the cube volume::

        box = o3d.geometry.TriangleMesh.create_box()
        box = o3d.t.geometry.TriangleMesh.from_legacy(box)
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
        sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

        ans = box.boolean_difference(sphere)

        o3d.visualization.draw([{'name': 'difference', 'geometry': ans}])
)");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
