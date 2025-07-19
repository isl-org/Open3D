// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/LineSet.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_lineset_declarations(py::module& m) {
    py::class_<LineSet, PyGeometry<LineSet>, std::shared_ptr<LineSet>, Geometry,
               DrawableGeometry>
            line_set(m, "LineSet", R"(
A LineSet contains points and lines joining them and optionally attributes on
the points and lines.  The ``LineSet`` class stores the attribute data in
key-value maps, where the key is the attribute name and value is a Tensor
containing the attribute data.  There are two maps: one each for ``point``
and ``line``.

The attributes of the line set have different levels::

    import open3d as o3d

    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32

    # Create an empty line set
    # Use lineset.point to access the point attributes
    # Use lineset.line to access the line attributes
    lineset = o3d.t.geometry.LineSet()

    # Default attribute: point.positions, line.indices
    # These attributes is created by default and are required by all line
    # sets. The shape must be (N, 3) and (N, 2) respectively. The device of
    # "positions" determines the device of the line set.
    lineset.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                  [0, 0, 1],
                                                  [0, 1, 0],
                                                  [0, 1, 1]], dtype_f, device)
    lineset.line.indices = o3d.core.Tensor([[0, 1],
                                               [1, 2],
                                               [2, 3],
                                               [3, 0]], dtype_i, device)

    # Common attributes: line.colors
    # Common attributes are used in built-in line set operations. The
    # spellings must be correct. For example, if "color" is used instead of
    # "color", some internal operations that expects "colors" will not work.
    # "colors" must have shape (N, 3) and must be on the same device as the
    # line set.
    lineset.line.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                              [0.1, 0.1, 0.1],
                                              [0.2, 0.2, 0.2],
                                              [0.3, 0.3, 0.3]], dtype_f, device)

    # User-defined attributes
    # You can also attach custom attributes. The value tensor must be on the
    # same device as the line set. The are no restrictions on the shape or
    # dtype, e.g.,
    lineset.point.labels = o3d.core.Tensor(...)
    lineset.line.features = o3d.core.Tensor(...)
)");
}

void pybind_lineset_definitions(py::module& m) {
    auto line_set = static_cast<
            py::class_<LineSet, PyGeometry<LineSet>, std::shared_ptr<LineSet>,
                       Geometry, DrawableGeometry>>(m.attr("LineSet"));
    // Constructors.
    line_set.def(py::init<const core::Device&>(),
                 "device"_a = core::Device("CPU:0"),
                 "Construct an empty LineSet on the provided device.")
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "point_positions"_a, "line_indices"_a, R"(
Construct a LineSet from point_positions and line_indices.

The input tensors will be directly used as the underlying storage of the line
set (no memory copy).  The resulting ``LineSet`` will have the same ``dtype``
and ``device`` as the tensor. The device for ``point_positions`` must be consistent with
``line_indices``.)");
    docstring::ClassMethodDocInject(
            m, "LineSet", "__init__",
            {{"point_positions", "A tensor with element shape (3,)"},
             {"line_indices",
              "A tensor with element shape (2,) and Int dtype."}});

    py::detail::bind_copy_functions<LineSet>(line_set);
    // Pickling support.
    line_set.def(py::pickle(
            [](const LineSet& line_set) {
                // __getstate__
                return py::make_tuple(line_set.GetDevice(),
                                      line_set.GetPointAttr(),
                                      line_set.GetLineAttr());
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 3) {
                    utility::LogError(
                            "Cannot unpickle LineSet! Expecting a tuple of "
                            "size 3.");
                }

                const core::Device device = t[0].cast<core::Device>();
                LineSet line_set(device);
                if (!device.IsAvailable()) {
                    utility::LogWarning(
                            "Device ({}) is not available. LineSet will be "
                            "created on CPU.",
                            device.ToString());
                    line_set.To(core::Device("CPU:0"));
                }

                const TensorMap point_attr = t[1].cast<TensorMap>();
                const TensorMap line_attr = t[2].cast<TensorMap>();
                for (auto& kv : point_attr) {
                    line_set.SetPointAttr(kv.first, kv.second);
                }
                for (auto& kv : line_attr) {
                    line_set.SetLineAttr(kv.first, kv.second);
                }

                return line_set;
            }));

    // Line set's attributes: point_positions, line_indices, line_colors, etc.
    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    line_set.def_property_readonly(
            "point", py::overload_cast<>(&LineSet::GetPointAttr, py::const_),
            "Dictionary containing point attributes. The primary key "
            "``positions`` contains point positions.");
    line_set.def_property_readonly(
            "line", py::overload_cast<>(&LineSet::GetLineAttr, py::const_),
            "Dictionary containing line attributes. The primary key "
            "``indices`` contains indices of points defining the lines.");

    line_set.def("__repr__", &LineSet::ToString);

    // Device transfers.
    line_set.def("to", &LineSet::To,
                 "Transfer the line set to a specified device.", "device"_a,
                 "copy"_a = false);
    line_set.def("clone", &LineSet::Clone,
                 "Returns copy of the line set on the same device.");
    line_set.def(
            "cpu",
            [](const LineSet& line_set) {
                return line_set.To(core::Device("CPU:0"));
            },
            "Transfer the line set to CPU. If the line set "
            "is already on CPU, no copy will be performed.");
    line_set.def(
            "cuda",
            [](const LineSet& line_set, int device_id) {
                return line_set.To(core::Device("CUDA", device_id));
            },
            "Transfer the line set to a CUDA device. If the line set "
            "is already on the specified CUDA device, no copy will be "
            "performed.",
            "device_id"_a = 0);

    // Line Set specific functions.
    line_set.def("get_min_bound", &LineSet::GetMinBound,
                 "Returns the min bound for point coordinates.");
    line_set.def("get_max_bound", &LineSet::GetMaxBound,
                 "Returns the max bound for point coordinates.");
    line_set.def("get_center", &LineSet::GetCenter,
                 "Returns the center for point coordinates.");
    line_set.def("transform", &LineSet::Transform, "transformation"_a, R"(
Transforms the points and lines. Custom attributes (e.g. point normals) are not
transformed. Extracts R, t from the transformation as:

.. math::
    T_{(4,4)} = \begin{bmatrix} R_{(3,3)} & t_{(3,1)} \\
                            O_{(1,3)} & s_{(1,1)} \end{bmatrix}

It assumes :math:`s = 1` (no scaling) and :math:`O = [0,0,0]` and applies the
transformation as :math:`P = R(P) + t`)");
    docstring::ClassMethodDocInject(
            m, "LineSet", "transform",
            {{"transformation",
              "Transformation [Tensor of shape (4,4)].  Should be on the same "
              "device as the LineSet"}});
    line_set.def("translate", &LineSet::Translate, "translation"_a,
                 "relative"_a = true,
                 "Translates points and lines of the LineSet.");
    docstring::ClassMethodDocInject(
            m, "LineSet", "translate",
            {{"translation",
              "Translation tensor of dimension (3,). Should be on the same "
              "device as the LineSet"},
             {"relative",
              "If true (default) translates relative to center of LineSet."}});
    line_set.def("scale", &LineSet::Scale, "scale"_a, "center"_a,
                 "Scale points and lines. Custom attributes are not scaled.");
    docstring::ClassMethodDocInject(
            m, "LineSet", "scale",
            {{"scale", "Scale magnitude."},
             {"center",
              "Center [Tensor of shape (3,)] about which the LineSet is to be "
              "scaled. Should be on the same device as the LineSet."}});
    line_set.def("rotate", &LineSet::Rotate, "R"_a, "center"_a,
                 "Rotate points and lines. Custom attributes (e.g. point "
                 "normals) are not rotated.");
    docstring::ClassMethodDocInject(
            m, "LineSet", "rotate",
            {{"R", "Rotation [Tensor of shape (3,3)]."},
             {"center",
              "Center [Tensor of shape (3,)] about which the LineSet is to be "
              "rotated. Should be on the same device as the LineSet."}});
    line_set.def_static("from_legacy", &LineSet::FromLegacy, "lineset_legacy"_a,
                        "float_dtype"_a = core::Float32,
                        "int_dtype"_a = core::Int64,
                        "device"_a = core::Device("CPU:0"),
                        "Create a LineSet from a legacy Open3D LineSet.");
    docstring::ClassMethodDocInject(
            m, "LineSet", "from_legacy",
            {
                    {"lineset_legacy", "Legacy Open3D LineSet."},
                    {"float_dtype",
                     "Float32 or Float64, used to store floating point values, "
                     "e.g. points, normals, colors."},
                    {"int_dtype",
                     "Int32 or Int64, used to store index values, e.g. line "
                     "indices."},
                    {"device",
                     "The device where the resulting LineSet resides."},
            });
    line_set.def("to_legacy", &LineSet::ToLegacy,
                 "Convert to a legacy Open3D LineSet.");

    line_set.def("get_axis_aligned_bounding_box",
                 &LineSet::GetAxisAlignedBoundingBox,
                 "Create an axis-aligned bounding box from point attribute "
                 "'positions'.");
    line_set.def("get_oriented_bounding_box", &LineSet::GetOrientedBoundingBox,
                 "Create an oriented bounding box from point attribute "
                 "'positions'.");
    line_set.def("extrude_rotation", &LineSet::ExtrudeRotation, "angle"_a,
                 "axis"_a, "resolution"_a = 16, "translation"_a = 0.0,
                 "capping"_a = true,
                 R"(Sweeps the line set rotationally about an axis.

Args:
    angle (float): The rotation angle in degree.
    axis (open3d.core.Tensor): The rotation axis.
    resolution (int): The resolution defines the number of intermediate sweeps
        about the rotation axis.
    translation (float): The translation along the rotation axis.

Returns:
    A triangle mesh with the result of the sweep operation.


Example:
    This code generates a spring from a single line::

        import open3d as o3d

        line = o3d.t.geometry.LineSet([[0.7,0,0],[1,0,0]], [[0,1]])
        spring = line.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
        o3d.visualization.draw([{'name': 'spring', 'geometry': spring}])

)");

    line_set.def("extrude_linear", &LineSet::ExtrudeLinear, "vector"_a,
                 "scale"_a = 1.0, "capping"_a = true,
                 R"(Sweeps the line set along a direction vector.

Args:
    vector (open3d.core.Tensor): The direction vector.
    scale (float): Scalar factor which essentially scales the direction vector.

Returns:
    A triangle mesh with the result of the sweep operation.


Example:
    This code generates an L-shaped mesh::

        import open3d as o3d

        lines = o3d.t.geometry.LineSet([[1.0,0.0,0.0],[0,0,0],[0,0,1]], [[0,1],[1,2]])
        mesh = lines.extrude_linear([0,1,0])
        o3d.visualization.draw([{'name': 'L', 'geometry': mesh}])

)");
    line_set.def("paint_uniform_color", &LineSet::PaintUniformColor, "color"_a,
                 "Assigns unifom color to all the lines of the LineSet. "
                 "Floating color values are clipped between 00 and 1.0. Input "
                 "`color` should be a (3,) shape tensor.");
    line_set.def_static(
            "create_camera_visualization", &LineSet::CreateCameraVisualization,
            "view_width_px"_a, "view_height_px"_a, "intrinsic"_a, "extrinsic"_a,
            "scale"_a = 1.f,
            py::arg_v(
                    "color", core::Tensor({}, core::Float32),
                    "open3d.core.Tensor([], dtype=open3d.core.Dtype.Float32)"),
            R"(Factory function to create a LineSet from intrinsic and extrinsic
matrices. Camera reference frame is shown with XYZ axes in RGB.

Args:
    view_width_px (int): The width of the view, in pixels.
    view_height_px (int): The height of the view, in pixels.
    intrinsic (open3d.core.Tensor): The intrinsic matrix {3,3} shape.
    extrinsic (open3d.core.Tensor): The extrinsic matrix {4,4} shape.
    scale (float): camera scale
    color (open3d.core.Tensor): color with float32 and shape {3}. Default is blue.

Example:

    Draw a purple camera frame with XYZ axes in RGB::

        import open3d.core as o3c
        from open3d.t.geometry import LineSet
        from open3d.visualization import draw
        K = o3c.Tensor([[512, 0, 512], [0, 512, 512], [0, 0, 1]], dtype=o3c.float32)
        T = o3c.Tensor.eye(4, dtype=o3c.float32)
        ls = LineSet.create_camera_visualization(1024, 1024, K, T, 1, [0.8, 0.2, 0.8])
        draw([ls])
)");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
