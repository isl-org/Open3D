// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/TriangleMesh.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_trianglemesh_declarations(py::module& m) {
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

    # Default attribute: vertex.positions, triangle.indices
    # These attributes is created by default and is required by all triangle
    # meshes. The shape of both must be (N, 3). The device of "positions"
    # determines the device of the triangle mesh.
    mesh.vertex.positions = o3d.core.Tensor([[0, 0, 0],
                                                [0, 0, 1],
                                                [0, 1, 0],
                                                [0, 1, 1]], dtype_f, device)
    mesh.triangle.indices = o3d.core.Tensor([[0, 1, 2],
                                                [0, 2, 3]]], dtype_i, device)

    # Common attributes: vertex.colors  , vertex.normals
    #                    triangle.colors, triangle.normals
    # Common attributes are used in built-in triangle mesh operations. The
    # spellings must be correct. For example, if "normal" is used instead of
    # "normals", some internal operations that expects "normals" will not work.
    # "normals" and "colors" must have shape (N, 3) and must be on the same
    # device as the triangle mesh.
    mesh.vertex.normals = o3d.core.Tensor([[0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0],
                                              [1, 1, 1]], dtype_f, device)
    mesh.vertex.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                             [0.1, 0.1, 0.1],
                                             [0.2, 0.2, 0.2],
                                             [0.3, 0.3, 0.3]], dtype_f, device)
    mesh.triangle.normals = o3d.core.Tensor(...)
    mesh.triangle.colors = o3d.core.Tensor(...)

    # User-defined attributes
    # You can also attach custom attributes. The value tensor must be on the
    # same device as the triangle mesh. The are no restrictions on the shape and
    # dtype, e.g.,
    pcd.vertex.labels = o3d.core.Tensor(...)
    pcd.triangle.features = o3d.core.Tensor(...)
)");
}

void pybind_trianglemesh_definitions(py::module& m) {
    auto triangle_mesh =
            static_cast<py::class_<TriangleMesh, PyGeometry<TriangleMesh>,
                                   std::shared_ptr<TriangleMesh>, Geometry,
                                   DrawableGeometry>>(m.attr("TriangleMesh"));
    // Constructors.
    triangle_mesh
            .def(py::init<const core::Device&>(),
                 "Construct an empty trianglemesh on the provided ``device`` "
                 "(default: 'CPU:0').",
                 "device"_a = core::Device("CPU:0"))
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "vertex_positions"_a, "triangle_indices"_a)
            .def("__repr__", &TriangleMesh::ToString);

    py::detail::bind_copy_functions<TriangleMesh>(triangle_mesh);
    // Pickle support.
    triangle_mesh.def(py::pickle(
            [](const TriangleMesh& mesh) {
                // __getstate__
                return py::make_tuple(mesh.GetDevice(), mesh.GetVertexAttr(),
                                      mesh.GetTriangleAttr());
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 3) {
                    utility::LogError(
                            "Cannot unpickle TriangleMesh! Expecting a tuple "
                            "of size 3.");
                }

                const core::Device device = t[0].cast<core::Device>();
                TriangleMesh mesh(device);
                if (!device.IsAvailable()) {
                    utility::LogWarning(
                            "Device ({}) is not available. TriangleMesh will "
                            "be created on CPU.",
                            device.ToString());
                    mesh.To(core::Device("CPU:0"));
                }

                const TensorMap vertex_attr = t[1].cast<TensorMap>();
                const TensorMap triangle_attr = t[2].cast<TensorMap>();
                for (auto& kv : vertex_attr) {
                    mesh.SetVertexAttr(kv.first, kv.second);
                }
                for (auto& kv : triangle_attr) {
                    mesh.SetTriangleAttr(kv.first, kv.second);
                }

                return mesh;
            }));

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
            "normalize_normals", &TriangleMesh::NormalizeNormals,
            "Normalize both triangle normals and vertex normals to length 1.");
    triangle_mesh.def("compute_triangle_normals",
                      &TriangleMesh::ComputeTriangleNormals,
                      "Function to compute triangle normals, usually called "
                      "before rendering.",
                      "normalized"_a = true);
    triangle_mesh.def("compute_vertex_normals",
                      &TriangleMesh::ComputeVertexNormals,
                      "Function to compute vertex normals, usually called "
                      "before rendering.",
                      "normalized"_a = true);

    triangle_mesh.def(
            "get_surface_area", &TriangleMesh::GetSurfaceArea,
            R"(Computes the surface area of the mesh, i.e., the sum of the individual triangle surfaces.

Example:
    This computes the surface area of the Stanford Bunny::
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.io.read_triangle_mesh(bunny.path)
        print('The surface area is', mesh.get_surface_area())

Returns:
    A scalar describing the surface area of the mesh.
)");

    triangle_mesh.def(
            "compute_convex_hull", &TriangleMesh::ComputeConvexHull,
            "joggle_inputs"_a = false,
            R"(Compute the convex hull of a point cloud using qhull. This runs on the CPU.

Args:
    joggle_inputs (bool with default False): Handle precision problems by
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
    triangle_mesh.def_static(
            "from_triangle_mesh_model", &TriangleMesh::FromTriangleMeshModel,
            "model"_a, "vertex_dtype"_a = core::Float32,
            "triangle_dtype"_a = core::Int64,
            "device"_a = core::Device("CPU:0"),
            R"(Convert a TriangleMeshModel (e.g. as read from a file with
`open3d.io.read_triangle_mesh_model()`) to a dictionary of mesh names to
triangle meshes with the specified vertex and triangle dtypes and moved to the
specified device. Only a single material per mesh is supported. Materials common
to multiple meshes will be duplicated. Textures (as t.geometry.Image) will use
shared storage on the CPU (GPU resident images for textures is not yet supported).

Returns:
    Dictionary of names to triangle meshes.

Example:
    Converting the FlightHelmetModel to a dictionary of triangle meshes::

        flight_helmet = o3d.data.FlightHelmetModel()
        model = o3d.io.read_triangle_model(flight_helmet.path)
        mesh_dict = o3d.t.geometry.TriangleMesh.from_triangle_mesh_model(model)
        o3d.visualization.draw(list({"name": name, "geometry": tmesh} for
            (name, tmesh) in mesh_dict.items()))
)");
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

    triangle_mesh.def(
            "slice_plane",
            // Accept anything for contour_values that pybind can convert to
            // std::list. This also avoids o3d.utility.DoubleVector.
            [](const TriangleMesh& self, const core::Tensor& point,
               const core::Tensor& normal, std::list<double> contour_values) {
                std::vector<double> cv(contour_values.begin(),
                                       contour_values.end());
                return self.SlicePlane(point, normal, cv);
            },
            "point"_a, "normal"_a, "contour_values"_a = std::list<double>{0.0},
            R"(Returns a line set with the contour slices defined by the plane and values.

This method generates slices as LineSet from the mesh at specific contour
values with respect to a plane.

Args:
    point (open3d.core.Tensor): A point on the plane.
    normal (open3d.core.Tensor): The normal of the plane.
    contour_values (list): A list of contour values at which slices will be
        generated. The value describes the signed distance to the plane.

Returns:
    LineSet with the extracted contours.


This example shows how to create a hemisphere from a sphere::

    import open3d as o3d
    import numpy as np

    bunny = o3d.data.BunnyMesh()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
    contours = mesh.slice_plane([0,0,0], [0,1,0], np.linspace(0,0.2))
    o3d.visualization.draw([{'name': 'bunny', 'geometry': contours}])

)");

    // Triangle Mesh's creation APIs.
    triangle_mesh
            .def_static("create_box", &TriangleMesh::CreateBox,
                        "Create a box triangle mesh. One vertex of the box"
                        "will be placed at the origin and the box aligns"
                        "with the positive x, y, and z axes."
                        "width"_a = 1.0,
                        "height"_a = 1.0, "depth"_a = 1.0,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_sphere", &TriangleMesh::CreateSphere,
                        "Create a sphere mesh centered at (0, 0, 0).",
                        "radius"_a = 1.0, "resolution"_a = 20,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_tetrahedron", &TriangleMesh::CreateTetrahedron,
                        "Create a tetrahedron mesh centered at (0, 0, 0).",
                        "radius"_a = 1.0, "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_octahedron", &TriangleMesh::CreateOctahedron,
                        "Create a octahedron mesh centered at (0, 0, 0).",
                        "radius"_a = 1.0, "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_icosahedron", &TriangleMesh::CreateIcosahedron,
                        "Create a icosahedron mesh centered at (0, 0, 0).",
                        "radius"_a = 1.0, "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_cylinder", &TriangleMesh::CreateCylinder,
                        "Create a cylinder mesh.", "radius"_a = 1.0,
                        "height"_a = 2.0, "resolution"_a = 20, "split"_a = 4,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_cone", &TriangleMesh::CreateCone,
                        "Create a cone mesh.", "radius"_a = 1.0,
                        "height"_a = 2.0, "resolution"_a = 20, "split"_a = 1,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_torus", &TriangleMesh::CreateTorus,
                        "Create a torus mesh.", "torus_radius"_a = 1.0,
                        "tube_radius"_a = 0.5, "radial_resolution"_a = 30,
                        "tubular_resolution"_a = 20,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_arrow", &TriangleMesh::CreateArrow,
                        "Create a arrow mesh.", "cylinder_radius"_a = 1.0,
                        "cone_radius"_a = 1.5, "cylinder_height"_a = 5.0,
                        "cone_height"_a = 4.0, "resolution"_a = 20,
                        "cylinder_split"_a = 4, "cone_split"_a = 1,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_coordinate_frame",
                        &TriangleMesh::CreateCoordinateFrame,
                        "Create a coordinate frame mesh.", "size"_a = 1.0,
                        "origin"_a = Eigen::Vector3d(0.0, 0.0, 0.0),
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"))
            .def_static("create_mobius", &TriangleMesh::CreateMobius,
                        "Create a Mobius strip.", "length_split"_a = 70,
                        "width_split"_a = 15, "twists"_a = 1, "raidus"_a = 1,
                        "flatness"_a = 1, "width"_a = 1, "scale"_a = 1,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"));

    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_box",
            {{"width", "x-directional length."},
             {"height", "y-directional length."},
             {"depth", "z-directional length."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create mesh."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_sphere",
            {{"radius", "The radius of the sphere."},
             {"resolution", "The resolution of the sphere."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create sphere."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_tetrahedron",
            {{"radius", "Distance from centroid to mesh vetices."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create tetrahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_octahedron",
            {{"radius", "Distance from centroid to mesh vetices."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_icosahedron",
            {{"radius", "Distance from centroid to mesh vetices."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_cylinder",
            {{"radius", "The radius of the cylinder."},
             {"height",
              "The height of the cylinder.The axis of the cylinder will be "
              "from (0, 0, -height/2) to (0, 0, height/2)."},
             {"resolution",
              " The circle will be split into ``resolution`` segments"},
             {"split", "The ``height`` will be split into ``split`` segments."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_cone",
            {{"radius", "The radius of the cone."},
             {"height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, 0) to (0, 0, height)."},
             {"resolution",
              "The circle will be split into ``resolution`` segments"},
             {"split", "The ``height`` will be split into ``split`` segments."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_torus",
            {{"torus_radius",
              "The radius from the center of the torus to the center of the "
              "tube."},
             {"tube_radius", "The radius of the torus tube."},
             {"radial_resolution",
              "The number of segments along the radial direction."},
             {"tubular_resolution",
              "The number of segments along the tubular direction."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_arrow",
            {{"cylinder_radius", "The radius of the cylinder."},
             {"cone_radius", "The radius of the cone."},
             {"cylinder_height",
              "The height of the cylinder. The cylinder is from (0, 0, 0) to "
              "(0, 0, cylinder_height)"},
             {"cone_height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, cylinder_height) to (0, 0, cylinder_height + cone_height)"},
             {"resolution",
              "The cone will be split into ``resolution`` segments."},
             {"cylinder_split",
              "The ``cylinder_height`` will be split into ``cylinder_split`` "
              "segments."},
             {"cone_split",
              "The ``cone_height`` will be split into ``cone_split`` "
              "segments."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_coordinate_frame",
            {{"size", "The size of the coordinate frame."},
             {"origin", "The origin of the coordinate frame."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});
    docstring::ClassMethodDocInject(
            m, "TriangleMesh", "create_mobius",
            {{"length_split", "The number of segments along the Mobius strip."},
             {"width_split",
              "The number of segments along the width of the Mobius strip."},
             {"twists", "Number of twists of the Mobius strip."},
             {"radius", "The radius of the Mobius strip."},
             {"flatness", "Controls the flatness/height of the Mobius strip."},
             {"width", "Width of the Mobius strip."},
             {"scale", "Scale the complete Mobius strip."},
             {"float_dtype", "Float_dtype, Float32 or Float64."},
             {"int_dtype", "Int_dtype, Int32 or Int64."},
             {"device", "Device of the create octahedron."}});

    triangle_mesh.def_static("create_text", &TriangleMesh::CreateText, "text"_a,
                             "depth"_a = 0.0, "float_dtype"_a = core::Float32,
                             "int_dtype"_a = core::Int64,
                             "device"_a = core::Device("CPU:0"),
                             R"(Create a triangle mesh from a text string.

Args:
    text (str): The text for generating the mesh. ASCII characters 32-126 are
        supported (includes alphanumeric characters and punctuation). In
        addition the line feed '\n' is supported to start a new line.
    depth (float): The depth of the generated mesh. If depth is 0 then a flat mesh will be generated.
    float_dtype (o3d.core.Dtype): Float type for the vertices. Either Float32 or Float64.
    int_dtype (o3d.core.Dtype): Int type for the triangle indices. Either Int32 or Int64.
    device (o3d.core.Device): The device for the returned mesh.

Returns:
    Text as triangle mesh.

Example:
    This shows how to simplifify the Stanford Bunny mesh::

        import open3d as o3d

        mesh = o3d.t.geometry.TriangleMesh.create_text('Open3D', depth=1)
        o3d.visualization.draw([{'name': 'text', 'geometry': mesh}])
)");

    triangle_mesh.def_static(
            "create_isosurfaces",
            // Accept anything for contour_values that pybind can convert to
            // std::list. This also avoids o3d.utility.DoubleVector.
            [](const core::Tensor& volume, std::list<double> contour_values,
               const core::Device& device) {
                std::vector<double> cv(contour_values.begin(),
                                       contour_values.end());
                return TriangleMesh::CreateIsosurfaces(volume, cv, device);
            },
            "volume"_a, "contour_values"_a = std::list<double>{0.0},
            "device"_a = core::Device("CPU:0"),
            R"(Create a mesh from a 3D scalar field (volume) by computing the
isosurface.

This method uses the Flying Edges dual contouring method that computes the
isosurface similar to Marching Cubes. The center of the first voxel of the
volume is at the origin (0,0,0). The center of the voxel at index [z,y,x]
will be at (x,y,z).

Args:
    volume (open3d.core.Tensor): 3D tensor with the volume.
    contour_values (list): A list of contour values at which isosurfaces will
        be generated. The default value is 0.
    device (o3d.core.Device): The device for the returned mesh.

Returns:
    A TriangleMesh with the extracted isosurfaces.


This example shows how to create a sphere from a volume::

    import open3d as o3d
    import numpy as np

    grid_coords = np.stack(np.meshgrid(*3*[np.linspace(-1,1,num=64)], indexing='ij'), axis=-1)
    vol = 0.5 - np.linalg.norm(grid_coords, axis=-1)
    mesh = o3d.t.geometry.TriangleMesh.create_isosurfaces(vol)
    o3d.visualization.draw(mesh)


This example shows how to convert a mesh to a signed distance field (SDF) and back to a mesh::

    import open3d as o3d
    import numpy as np

    mesh1 = o3d.t.geometry.TriangleMesh.create_torus()
    grid_coords = np.stack(np.meshgrid(*3*[np.linspace(-2,2,num=64, dtype=np.float32)], indexing='ij'), axis=-1)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh1)
    sdf = scene.compute_signed_distance(grid_coords)
    mesh2 = o3d.t.geometry.TriangleMesh.create_isosurfaces(sdf)

    # Flip the triangle orientation for SDFs with negative values as "inside" and positive values as "outside"
    mesh2.triangle.indices = mesh2.triangle.indices[:,[2,1,0]]

    o3d.visualization.draw(mesh2)

)");

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
    triangle_mesh.def("compute_adjacency_list", &TriangleMesh::ComputeAdjacencyList,
    "Return Mesh Adjacency Matrix in CSR format using Triangle indices attribute.");
    
    triangle_mesh.def("get_axis_aligned_bounding_box",
                      &TriangleMesh::GetAxisAlignedBoundingBox,
                      "Create an axis-aligned bounding box from vertex "
                      "attribute 'positions'.");
                      
    triangle_mesh.def("get_oriented_bounding_box",
                      &TriangleMesh::GetOrientedBoundingBox,
                      "Create an oriented bounding box from vertex attribute "
                      "'positions'.");

    triangle_mesh.def("fill_holes", &TriangleMesh::FillHoles,
                      "hole_size"_a = 1e6,
                      R"(Fill holes by triangulating boundary edges.

This function always uses the CPU device.

Args:
    hole_size (float): This is the approximate threshold for filling holes.
        The value describes the maximum radius of holes to be filled.

Returns:
    New mesh after filling holes.

Example:
    Fill holes at the bottom of the Stanford Bunny mesh::

        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        filled = mesh.fill_holes()
        o3d.visualization.draw([{'name': 'filled', 'geometry': ans}])
)");

    triangle_mesh.def(
            "compute_uvatlas", &TriangleMesh::ComputeUVAtlas, "size"_a = 512,
            "gutter"_a = 1.f, "max_stretch"_a = 1.f / 6,
            "parallel_partitions"_a = 1, "nthreads"_a = 0,
            R"(Creates an UV atlas and adds it as triangle attr 'texture_uvs' to the mesh.

Input meshes must be manifold for this method to work.
The algorithm is based on:
Zhou et al, "Iso-charts: Stretch-driven Mesh Parameterization using Spectral Analysis", Eurographics Symposium on Geometry Processing (2004)
Sander et al. "Signal-Specialized Parametrization" Europgraphics 2002
This function always uses the CPU device.

Args:
    size (int): The target size of the texture (size x size). The uv coordinates
        will still be in the range [0..1] but parameters like gutter use pixels
        as units.
    gutter (float): This is the space around the uv islands in pixels.
    max_stretch (float): The maximum amount of stretching allowed. The parameter
        range is [0..1] with 0 meaning no stretch allowed.

    parallel_partitions (int): The approximate number of partitions created
        before computing the UV atlas for parallelizing the computation.
        Parallelization can be enabled with values > 1. Note that
        parallelization increases the number of UV islands and can lead to results
        with lower quality.

    nthreads (int): The number of threads used when parallel_partitions
        is > 1. Set to 0 for automatic number of thread detection.

Returns:
    This function creates a face attribute "texture_uvs" and returns a tuple
    with (max stretch, num_charts, num_partitions) storing the
    actual amount of stretch, the number of created charts, and the number of
    parallel partitions created.

Example:
    This code creates a uv map for the Stanford Bunny mesh::

        import open3d as o3d
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        mesh.compute_uvatlas()

        # Add a wood texture and visualize
        texture_data = o3d.data.WoodTexture()
        mesh.material.material_name = 'defaultLit'
        mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(texture_data.albedo_texture_path)
        o3d.visualization.draw(mesh)
)");

    triangle_mesh.def("bake_vertex_attr_textures",
                      &TriangleMesh::BakeVertexAttrTextures, "size"_a,
                      "vertex_attr"_a, "margin"_a = 2., "fill"_a = 0.,
                      "update_material"_a = true,
                      R"(Bake vertex attributes into textures.

This function assumes a triangle attribute with name 'texture_uvs'.
Only float type attributes can be baked to textures.

This function always uses the CPU device.

Args:
    size (int): The width and height of the texture in pixels. Only square
        textures are supported.

    vertex_attr (set): The vertex attributes for which textures should be
        generated.

    margin (float): The margin in pixels. The recommended value is 2. The margin
        are additional pixels around the UV islands to avoid discontinuities.

    fill (float): The value used for filling texels outside the UV islands.

    update_material (bool): If true updates the material of the mesh.
        Baking a vertex attribute with the name 'albedo' will become the albedo
        texture in the material. Existing textures in the material will be
        overwritten.

Returns:
    A dictionary of tensors that store the baked textures.

Example:
    We generate a texture storing the xyz coordinates for each texel::

        import open3d as o3d
        from matplotlib import pyplot as plt

        box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
        box = o3d.t.geometry.TriangleMesh.from_legacy(box)
        box.vertex['albedo'] = box.vertex.positions

        # Initialize material and bake the 'albedo' vertex attribute to a
        # texture. The texture will be automatically added to the material of
        # the object.
        box.material.set_default_properties()
        texture_tensors = box.bake_vertex_attr_textures(128, {'albedo'})

        # Shows the textured cube.
        o3d.visualization.draw([box])

        # Plot the tensor with the texture.
        plt.imshow(texture_tensors['albedo'].numpy())

)");

    triangle_mesh.def("bake_triangle_attr_textures",
                      &TriangleMesh::BakeTriangleAttrTextures, "size"_a,
                      "triangle_attr"_a, "margin"_a = 2., "fill"_a = 0.,
                      "update_material"_a = true,
                      R"(Bake triangle attributes into textures.

This function assumes a triangle attribute with name 'texture_uvs'.

This function always uses the CPU device.

Args:
    size (int): The width and height of the texture in pixels. Only square
        textures are supported.

    triangle_attr (set): The vertex attributes for which textures should be
        generated.

    margin (float): The margin in pixels. The recommended value is 2. The margin
        are additional pixels around the UV islands to avoid discontinuities.

    fill (float): The value used for filling texels outside the UV islands.

    update_material (bool): If true updates the material of the mesh.
        Baking a vertex attribute with the name 'albedo' will become the albedo
        texture in the material. Existing textures in the material will be
        overwritten.

Returns:
    A dictionary of tensors that store the baked textures.

Example:
    We generate a texture visualizing the index of the triangle to which the
    texel belongs to::

        import open3d as o3d
        from matplotlib import pyplot as plt

        box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
        box = o3d.t.geometry.TriangleMesh.from_legacy(box)
        # Creates a triangle attribute 'albedo' which is the triangle index
        # multiplied by (255//12).
        box.triangle['albedo'] = (255//12)*np.arange(box.triangle.indices.shape[0], dtype=np.uint8)

        # Initialize material and bake the 'albedo' triangle attribute to a
        # texture. The texture will be automatically added to the material of
        # the object.
        box.material.set_default_properties()
        texture_tensors = box.bake_triangle_attr_textures(128, {'albedo'})

        # Shows the textured cube.
        o3d.visualization.draw([box])

        # Plot the tensor with the texture.
        plt.imshow(texture_tensors['albedo'].numpy())
)");

    triangle_mesh.def("extrude_rotation", &TriangleMesh::ExtrudeRotation,
                      "angle"_a, "axis"_a, "resolution"_a = 16,
                      "translation"_a = 0.0, "capping"_a = true,
                      R"(Sweeps the triangle mesh rotationally about an axis.
Args:
    angle (float): The rotation angle in degree.
    axis (open3d.core.Tensor): The rotation axis.
    resolution (int): The resolution defines the number of intermediate sweeps
        about the rotation axis.
    translation (float): The translation along the rotation axis.

Returns:
    A triangle mesh with the result of the sweep operation.

Example:
    This code generates a spring with a triangle cross-section::

        import open3d as o3d

        mesh = o3d.t.geometry.TriangleMesh([[1,1,0], [0.7,1,0], [1,0.7,0]], [[0,1,2]])
        spring = mesh.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
        o3d.visualization.draw([{'name': 'spring', 'geometry': spring}])
)");

    triangle_mesh.def("extrude_linear", &TriangleMesh::ExtrudeLinear,
                      "vector"_a, "scale"_a = 1.0, "capping"_a = true,
                      R"(Sweeps the line set along a direction vector.
Args:
    vector (open3d.core.Tensor): The direction vector.
    scale (float): Scalar factor which essentially scales the direction vector.

Returns:
    A triangle mesh with the result of the sweep operation.

Example:
    This code generates a wedge from a triangle::

        import open3d as o3d
        triangle = o3d.t.geometry.TriangleMesh([[1.0,1.0,0.0], [0,1,0], [1,0,0]], [[0,1,2]])
        wedge = triangle.extrude_linear([0,0,1])
        o3d.visualization.draw([{'name': 'wedge', 'geometry': wedge}])
)");

    triangle_mesh.def("pca_partition", &TriangleMesh::PCAPartition,
                      "max_faces"_a,
                      R"(Partition the mesh by recursively doing PCA.

This function creates a new face attribute with the name "partition_ids" storing
the partition id for each face.

Args:
    max_faces (int): The maximum allowed number of faces in a partition.


Example:

    This code partitions a mesh such that each partition contains at most 20k
    faces::

        import open3d as o3d
        import numpy as np
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        num_partitions = mesh.pca_partition(max_faces=20000)

        # print the partition ids and the number of faces for each of them.
        print(np.unique(mesh.triangle.partition_ids.numpy(), return_counts=True))

)");

    triangle_mesh.def(
            "select_faces_by_mask", &TriangleMesh::SelectFacesByMask, "mask"_a,
            R"(Returns a new mesh with the faces selected by a boolean mask.

Args:
    mask (open3d.core.Tensor): A boolean mask with the shape (N) with N as the
        number of faces in the mesh.

Returns:
    A new mesh with the selected faces. If the original mesh is empty, return an empty mesh.

Example:

    This code partitions the mesh using PCA and then visualized the individual
    parts::

        import open3d as o3d
        import numpy as np
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
        num_partitions = mesh.pca_partition(max_faces=20000)

        parts = []
        for i in range(num_partitions):
            mask = mesh.triangle.partition_ids == i
            part = mesh.select_faces_by_mask(mask)
            part.vertex.colors = np.tile(np.random.rand(3), (part.vertex.positions.shape[0],1))
            parts.append(part)

        o3d.visualization.draw(parts)

)");

    triangle_mesh.def(
            "select_by_index", &TriangleMesh::SelectByIndex, "indices"_a,
            R"(Returns a new mesh with the vertices selected according to the indices list.
If an item from the indices list exceeds the max vertex number of the mesh
or has a negative value, it is ignored.

Args:
    indices (open3d.core.Tensor): An integer list of indices. Duplicates are
    allowed, but ignored. Signed and unsigned integral types are accepted.

Returns:
    A new mesh with the selected vertices and faces built from these vertices.
    If the original mesh is empty, return an empty mesh.

Example:

    This code selects the top face of a box, which has indices [2, 3, 6, 7]::

        import open3d as o3d
        import numpy as np
        box = o3d.t.geometry.TriangleMesh.create_box()
        top_face = box.select_by_index([2, 3, 6, 7])
)");

    triangle_mesh.def(
            "remove_unreferenced_vertices",
            &TriangleMesh::RemoveUnreferencedVertices,
            R"(Removes unreferenced vertices from the mesh in-place.)");

    triangle_mesh.def(
            "compute_triangle_areas", &TriangleMesh::ComputeTriangleAreas,
            R"(Compute triangle areas and save it as \"areas\" triangle attribute.

Returns:
    The mesh.

Example:

    This code computes the overall surface area of a box::

        import open3d as o3d
        box = o3d.t.geometry.TriangleMesh.create_box()
        surface_area = box.compute_triangle_areas().triangle.areas.sum()
)");

    triangle_mesh.def("remove_non_manifold_edges",
                      &TriangleMesh::RemoveNonManifoldEdges,
                      R"(Function that removes all non-manifold edges, by
successively deleting  triangles with the smallest surface
area adjacent to the non-manifold edge until the number of
adjacent triangles to the edge is `<= 2`.

Returns:
    The mesh.
)");

    triangle_mesh.def("get_non_manifold_edges",
                      &TriangleMesh::GetNonManifoldEdges,
                      "allow_boundary_edges"_a = true,
                      R"(Returns the list consisting of non-manifold edges.)");

    triangle_mesh.def(
            "sample_points_uniformly", &TriangleMesh::SamplePointsUniformly,
            "number_of_points"_a, "use_triangle_normal"_a = false,
            R"(Sample points uniformly from the triangle mesh surface and return as a PointCloud. Normals and colors are interpolated from the triangle mesh. If texture_uvs and albedo are present, these are used to estimate the sampled point color, otherwise vertex colors are used, if present. During sampling, triangle areas are computed and saved in the "areas" attribute.

Args:
    number_of_points (int): The number of points to sample.
    use_triangle_normal (bool): If true, use the triangle normal as the normal of the sampled point. By default, the vertex normals are interpolated instead.

Returns:
    Sampled point cloud, with colors and normals, if available.

Example::

    mesh = o3d.t.geometry.TriangleMesh.create_box()
    mesh.vertex.colors = mesh.vertex.positions.clone()
    pcd = mesh.sample_points_uniformly(100000)
    o3d.visualization.draw([mesh, pcd], point_size=5, show_ui=True, show_skybox=False)

    )");

    triangle_mesh.def("compute_metrics", &TriangleMesh::ComputeMetrics,
                      "mesh2"_a, "metrics"_a, "params"_a,
                      R"(Compute various metrics between two triangle meshes. 
            
This uses ray casting for distance computations between a sampled point cloud 
and a triangle mesh.  Currently, Chamfer distance, Hausdorff distance  and 
F-Score `[Knapitsch2017] <../tutorial/reference.html#Knapitsch2017>`_ are supported. 
The Chamfer distance is the sum of the mean distance to the nearest neighbor from
the sampled surface points of the first mesh to the second mesh and vice versa. 
The F-Score at the fixed threshold radius is the harmonic mean of the Precision 
and Recall. Recall is the percentage of surface points from the first mesh that 
have the second mesh within the threshold radius, while Precision is the 
percentage of sampled points from the second mesh that have the first mesh 
surface within the threhold radius.

.. math::
    :nowrap:

    \begin{align}
        \text{Chamfer Distance: } d_{CD}(X,Y) &= \frac{1}{|X|}\sum_{i \in X} || x_i - n(x_i, Y) || + \frac{1}{|Y|}\sum_{i \in Y} || y_i - n(y_i, X) ||\\
        \text{Hausdorff distance: } d_H(X,Y) &= \max \left\{ \max_{i \in X} || x_i - n(x_i, Y) ||, \max_{i \in Y} || y_i - n(y_i, X) || \right\}\\
        \text{Precision: } P(X,Y|d) &= \frac{100}{|X|} \sum_{i \in X} || x_i - n(x_i, Y) || < d \\
        \text{Recall: } R(X,Y|d) &= \frac{100}{|Y|} \sum_{i \in Y} || y_i - n(y_i, X) || < d \\
        \text{F-Score: } F(X,Y|d) &= \frac{2 P(X,Y|d) R(X,Y|d)}{P(X,Y|d) + R(X,Y|d)} \\
    \end{align}

As a side effect, the triangle areas are saved in the "areas" attribute.

Args:
    mesh2 (t.geometry.TriangleMesh): Other triangle mesh to compare with.
    metrics (Sequence[t.geometry.Metric]): List of Metrics to compute. Multiple metrics can be computed at once for efficiency.
    params (t.geometry.MetricParameters): This holds parameters required by different metrics.

Returns:
    Tensor containing the requested metrics.

Example::

    from open3d.t.geometry import TriangleMesh, Metric, MetricParameters
    # box is a cube with one vertex at the origin and a side length 1
    box1 = TriangleMesh.create_box()
    box2 = TriangleMesh.create_box()
    box2.vertex.positions *= 1.1

    # 3 faces of the cube are the same, and 3 are shifted up by 0.1
    metric_params = MetricParameters(fscore_radius=o3d.utility.FloatVector(
        (0.05, 0.15)), n_sampled_points=100000)
    metrics = box1.compute_metrics(
        box2, (Metric.ChamferDistance, Metric.HausdorffDistance, Metric.FScore),
        metric_params)

    print(metrics)
    np.testing.assert_allclose(metrics.cpu().numpy(), (0.1, 0.17, 50, 100),
                               rtol=0.05)
    )");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
