// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/BoundingVolume.h"

#include <string>

#include "open3d/core/CUDAUtils.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_boundingvolume_declarations(py::module& m) {
    py::class_<AxisAlignedBoundingBox, PyGeometry<AxisAlignedBoundingBox>,
               std::shared_ptr<AxisAlignedBoundingBox>, Geometry,
               DrawableGeometry>
            aabb(m, "AxisAlignedBoundingBox",
                 R"(A bounding box that is aligned along the coordinate axes and
has the properties:

- (``min_bound``, ``max_bound``): Lower and upper bounds of the bounding box for all axes. These are tensors with shape (3,) and a common data type and device. The data type can only be ``open3d.core.float32`` (default) or ``open3d.core.float64``. The device of the tensor determines the device of the box.
- ``color``: Color of the bounding box is a tensor with shape (3,) and a data type ``open3d.core.float32`` (default) or ``open3d.core.float64``. Values can only be in the range [0.0, 1.0].)");
    py::class_<OrientedBoundingBox, PyGeometry<OrientedBoundingBox>,
               std::shared_ptr<OrientedBoundingBox>, Geometry, DrawableGeometry>
            obb(m, "OrientedBoundingBox",
                R"(A bounding box oriented along an arbitrary frame of reference
with the properties:

- (``center``, ``rotation``, ``extent``): The oriented bounding box is defined by its center position (shape (3,)), rotation maxtrix (shape (3,3)) and extent (shape (3,)).  Each of these tensors must have the same data type and device. The data type can only be ``open3d.core.float32`` (default) or ``open3d.core.float64``. The device of the tensor determines the device of the box.
- ``color``: Color of the bounding box is a tensor with shape (3,) and a data type ``open3d.core.float32`` (default) or ``open3d.core.float64``. Values can only be in the range [0.0, 1.0].)");
}
void pybind_boundingvolume_definitions(py::module& m) {
    auto aabb = static_cast<py::class_<AxisAlignedBoundingBox,
                                       PyGeometry<AxisAlignedBoundingBox>,
                                       std::shared_ptr<AxisAlignedBoundingBox>,
                                       Geometry, DrawableGeometry>>(
            m.attr("AxisAlignedBoundingBox"));
    aabb.def(py::init<const core::Device&>(),
             "device"_a = core::Device("CPU:0"),
             "Construct an empty axis-aligned box on the provided "
             "device.");
    aabb.def(py::init<const core::Tensor&, const core::Tensor&>(),
             "min_bound"_a, "max_bound"_a,
             R"(Construct an axis-aligned box from min/max bound.
The axis-aligned box will be created on the device of the given bound
tensor, which must be on the same device and have the same data type.)");
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "__init__",
            {{"min_bound",
              "Lower bounds of the bounding box for all axes. Tensor with {3,} "
              "shape, and type float32 or float64."},
             {"max_bound",
              "Upper bounds of the bounding box for all axes. Tensor with {3,} "
              "shape, and type float32 or float64."}});

    aabb.def_property_readonly(
            "dtype", &AxisAlignedBoundingBox::GetDtype,
            "Returns the data type attribute of this AxisAlignedBoundingBox.");
    py::detail::bind_copy_functions<AxisAlignedBoundingBox>(aabb);
    aabb.def("__repr__", &AxisAlignedBoundingBox::ToString);
    aabb.def(
            "__add__",
            [](const AxisAlignedBoundingBox& self,
               const AxisAlignedBoundingBox& other) {
                AxisAlignedBoundingBox result = self.Clone();
                return result += other;
            },
            R"(Add operation for axis-aligned bounding box.
The device of ohter box must be the same as the device of the current box.)");

    // Device transfers.
    aabb.def("to", &AxisAlignedBoundingBox::To,
             "Transfer the axis-aligned box to a specified device.", "device"_a,
             "copy"_a = false);
    aabb.def("clone", &AxisAlignedBoundingBox::Clone,
             "Returns copy of the axis-aligned box on the same device.");
    aabb.def(
            "cpu",
            [](const AxisAlignedBoundingBox& aabb) {
                return aabb.To(core::Device("CPU:0"));
            },
            "Transfer the axis-aligned box to CPU. If the axis-aligned box is "
            "already on CPU, no copy will be performed.");
    aabb.def(
            "cuda",
            [](const AxisAlignedBoundingBox& aabb, int device_id) {
                return aabb.To(core::Device("CUDA", device_id));
            },
            "Transfer the axis-aligned box to a CUDA device. If the "
            "axis-aligned box is already on the specified CUDA device, no copy "
            "will be performed.",
            "device_id"_a = 0);

    aabb.def("set_min_bound", &AxisAlignedBoundingBox::SetMinBound,
             "Set the lower bound of the axis-aligned box.", "min_bound"_a);
    aabb.def("set_max_bound", &AxisAlignedBoundingBox::SetMaxBound,
             "Set the upper bound of the axis-aligned box.", "max_bound"_a);
    aabb.def("set_color", &AxisAlignedBoundingBox::SetColor,
             "Set the color of the axis-aligned box.", "color"_a);

    aabb.def_property_readonly("min_bound",
                               &AxisAlignedBoundingBox::GetMinBound,
                               "Returns the min bound for box coordinates.");
    aabb.def_property_readonly("max_bound",
                               &AxisAlignedBoundingBox::GetMaxBound,
                               "Returns the max bound for box coordinates.");
    aabb.def_property_readonly("color", &AxisAlignedBoundingBox::GetColor,
                               "Returns the color for box.");
    aabb.def("get_center", &AxisAlignedBoundingBox::GetCenter,
             "Returns the center for box coordinates.");

    aabb.def("translate", &AxisAlignedBoundingBox::Translate, R"(Translate the
axis-aligned box by the given translation. If relative is true, the translation
is applied to the current min and max bound. If relative is false, the
translation is applied to make the box's center at the given translation.)",
             "translation"_a, "relative"_a = true);
    aabb.def("scale", &AxisAlignedBoundingBox::Scale, R"(Scale the axis-aligned
box.
If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of the axis aligned
bounding box, and \f$s\f$ and \f$c\f$ are the provided scaling factor and
center respectively, then the new min_bound and max_bound are given by
\f$mi = c + s (mi - c)\f$ and \f$ma = c + s (ma - c)\f$.
The scaling center will be the box center if it is not specified.)",
             "scale"_a, "center"_a = utility::nullopt);

    aabb.def("get_extent", &AxisAlignedBoundingBox::GetExtent,
             "Get the extent/length of the bounding box in x, y, and z "
             "dimension.");
    aabb.def("get_half_extent", &AxisAlignedBoundingBox::GetHalfExtent,
             "Returns the half extent of the bounding box.");
    aabb.def("get_max_extent", &AxisAlignedBoundingBox::GetMaxExtent,
             "Returns the maximum extent, i.e. the maximum of X, Y and Z "
             "axis's extents.");
    aabb.def("volume", &AxisAlignedBoundingBox::Volume,
             "Returns the volume of the bounding box.");
    aabb.def("get_box_points", &AxisAlignedBoundingBox::GetBoxPoints,
             "Returns the eight points that define the bounding box. The "
             "Return tensor has shape {8, 3} and data type of float32.");
    aabb.def("get_point_indices_within_bounding_box",
             &AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox,
             "Indices to points that are within the bounding box.", "points"_a);

    aabb.def("to_legacy", &AxisAlignedBoundingBox::ToLegacy,
             "Convert to a legacy Open3D axis-aligned box.");
    aabb.def("get_oriented_bounding_box",
             &AxisAlignedBoundingBox::GetOrientedBoundingBox,
             "Convert to an oriented box.");
    aabb.def_static("from_legacy", &AxisAlignedBoundingBox::FromLegacy, "box"_a,
                    "dtype"_a = core::Float32,
                    "device"_a = core::Device("CPU:0"),
                    "Create an AxisAlignedBoundingBox from a legacy Open3D "
                    "axis-aligned box.");
    aabb.def_static(
            "create_from_points", &AxisAlignedBoundingBox::CreateFromPoints,
            "Creates the axis-aligned box that encloses the set of points.",
            "points"_a);

    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "set_min_bound",
            {{"min_bound",
              "Tensor with {3,} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "set_max_bound",
            {{"max_bound",
              "Tensor with {3,} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "set_color",
            {{"color",
              "Tensor with {3,} shape, and type float32 or float64, with "
              "values in range [0.0, 1.0]."}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "translate",
            {{"translation",
              "Translation tensor of shape (3,), type float32 or float64, "
              "device same as the box."},
             {"relative", "Whether to perform relative translation."}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "scale",
            {{"scale", "The scale parameter."},
             {"center",
              "Center used for the scaling operation. Tensor with {3,} shape, "
              "and type float32 or float64"}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox",
            "get_point_indices_within_bounding_box",
            {{"points",
              "Tensor with {N, 3} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "create_from_points",
            {{"points",
              "A list of points with data type of float32 or float64 (N x 3 "
              "tensor)."}});
    auto obb = static_cast<py::class_<
            OrientedBoundingBox, PyGeometry<OrientedBoundingBox>,
            std::shared_ptr<OrientedBoundingBox>, Geometry, DrawableGeometry>>(
            m.attr("OrientedBoundingBox"));
    obb.def(py::init<const core::Device&>(), "device"_a = core::Device("CPU:0"),
            "Construct an empty OrientedBoundingBox on the provided device.");
    obb.def(py::init<const core::Tensor&, const core::Tensor&,
                     const core::Tensor&>(),
            "center"_a, "rotation"_a, "extent"_a,
            R"(Construct an OrientedBoundingBox from center, rotation and extent.
The OrientedBoundingBox will be created on the device of the given tensors, which
must be on the same device and have the same data type.)");
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "__init__",
            {{"center",
              "Center of the bounding box. Tensor of shape {3,}, and type "
              "float32 or float64."},
             {"rotation",
              "Rotation matrix of the bounding box. Tensor of shape {3, 3}, "
              "and type float32 or float64."},
             {"extent",
              "Extent of the bounding box. Tensor of shape {3,}, and type "
              "float32 or float64."}});
    obb.def_property_readonly(
            "dtype", &OrientedBoundingBox::GetDtype,
            "Returns the data type attribute of this OrientedBoundingBox.");
    py::detail::bind_copy_functions<OrientedBoundingBox>(obb);
    obb.def("__repr__", &OrientedBoundingBox::ToString);

    // Device transfers.
    obb.def("to", &OrientedBoundingBox::To,
            "Transfer the oriented box to a specified device.", "device"_a,
            "copy"_a = false);
    obb.def("clone", &OrientedBoundingBox::Clone,
            "Returns copy of the oriented box on the same device.");
    obb.def(
            "cpu",
            [](const OrientedBoundingBox& obb) {
                return obb.To(core::Device("CPU:0"));
            },
            "Transfer the oriented box to CPU. If the oriented box is "
            "already on CPU, no copy will be performed.");
    obb.def(
            "cuda",
            [](const OrientedBoundingBox& obb, int device_id) {
                return obb.To(core::Device("CUDA", device_id));
            },
            "Transfer the oriented box to a CUDA device. If the oriented box "
            "is already on the specified CUDA device, no copy will be "
            "performed.",
            "device_id"_a = 0);

    obb.def("set_center", &OrientedBoundingBox::SetCenter,
            "Set the center of the box.", "center"_a);
    obb.def("set_rotation", &OrientedBoundingBox::SetRotation,
            "Set the rotation matrix of the box.", "rotation"_a);
    obb.def("set_extent", &OrientedBoundingBox::SetExtent,
            "Set the extent of the box.", "extent"_a);
    obb.def("set_color", &OrientedBoundingBox::SetColor,
            "Set the color of the oriented box.", "color"_a);

    obb.def_property_readonly("center", &OrientedBoundingBox::GetCenter,
                              "Returns the center for box.");
    obb.def_property_readonly("extent", &OrientedBoundingBox::GetExtent,
                              "Returns the extent for box coordinates.");
    obb.def_property_readonly("rotation", &OrientedBoundingBox::GetRotation,
                              "Returns the rotation for box.");
    obb.def_property_readonly("color", &OrientedBoundingBox::GetColor,
                              "Returns the color for box.");
    obb.def("get_min_bound", &OrientedBoundingBox::GetMinBound,
            "Returns the min bound for box.");
    obb.def("get_max_bound", &OrientedBoundingBox::GetMaxBound,
            "Returns the max bound for box.");

    obb.def("translate", &OrientedBoundingBox::Translate, R"(Translate the
oriented box by the given translation. If relative is true, the translation is
added to the center of the box. If false, the center will be assigned to the
translation.)",
            "translation"_a, "relative"_a = true);
    obb.def("rotate", &OrientedBoundingBox::Rotate,
            R"(Rotate the oriented box by the given rotation matrix. If the
rotation matrix is not orthogonal, the rotation will no be applied.
The rotation center will be the box center if it is not specified.)",
            "rotation"_a, "center"_a = utility::nullopt);
    obb.def("transform", &OrientedBoundingBox::Transform,
            "Transform the oriented box by the given transformation matrix.",
            "transformation"_a);
    obb.def("scale", &OrientedBoundingBox::Scale, R"(Scale the axis-aligned
box.
If \f$mi\f$ is the min_bound and \f$ma\f$ is the max_bound of the axis aligned
bounding box, and \f$s\f$ and \f$c\f$ are the provided scaling factor and
center respectively, then the new min_bound and max_bound are given by
\f$mi = c + s (mi - c)\f$ and \f$ma = c + s (ma - c)\f$.
The scaling center will be the box center if it is not specified.)",
            "scale"_a, "center"_a = utility::nullopt);

    obb.def("volume", &OrientedBoundingBox::Volume,
            "Returns the volume of the bounding box.");
    obb.def("get_box_points", &OrientedBoundingBox::GetBoxPoints,
            "Returns the eight points that define the bounding box. The "
            "Return tensor has shape {8, 3} and data type same as the box.");
    obb.def("get_point_indices_within_bounding_box",
            &OrientedBoundingBox::GetPointIndicesWithinBoundingBox,
            "Indices to points that are within the bounding box.", "points"_a);
    obb.def("get_axis_aligned_bounding_box",
            &OrientedBoundingBox::GetAxisAlignedBoundingBox,
            " Returns an oriented bounding box from the "
            "AxisAlignedBoundingBox.");

    obb.def("to_legacy", &OrientedBoundingBox::ToLegacy,
            "Convert to a legacy Open3D oriented box.");
    obb.def_static(
            "from_legacy", &OrientedBoundingBox::FromLegacy, "box"_a,
            "dtype"_a = core::Float32, "device"_a = core::Device("CPU:0"),
            "Create an oriented bounding box from the AxisAlignedBoundingBox.");
    obb.def_static(
            "create_from_axis_aligned_bounding_box",
            &OrientedBoundingBox::CreateFromAxisAlignedBoundingBox, "aabb"_a,
            "Create an OrientedBoundingBox from a legacy Open3D oriented box.");

    obb.def_static("create_from_points", &OrientedBoundingBox::CreateFromPoints,
                   R"(Creates an oriented bounding box with various algorithms.

Args:
    points (open3d.core.Tensor): A list of points with data type of float32 or 
        float64 (N x 3 tensor, where N must be larger than 3).
    robust (bool): If set to true uses a more robust method which works in 
        degenerate cases but introduces noise to the points coordinates.
    method (open3d.t.geometry.OrientedBoundingBox.Method): This is one of 
        ``PCA``, ``MINIMAL_APPROX``, ``MINIMAL_JYLANKI``. 

        - ``PCA`` computes an oriented bounding box using the principal component
          analysis. The output is in most cases not a minimal oriented bounding box.
          This algorithm is the fastest.
        - ``MINIMAL_APPROX`` is a fast approximation algorithm. The algorithm 
          makes use of the fact that at least one edge of the convex hull must be
          collinear with an edge of the minimum bounding box: for each triangle in
          the convex hull, calculate the minimal axis aligned box in the frame of
          that triangle. At the end, return the box with the smallest volume.
          This algorithm is a good choice if speed and accuracy is important.
        - ``MINIMAL_JYLANKI`` computes an oriented bounding box using a more 
          accurate but slower algorithm. This algorithm is inspired by the article 
          "An Exact Algorithm for Finding Minimum Oriented Bounding Boxes" 
          written by Jukka Jylänki. The original implementation can be found at 
          the following address:
          https://github.com/juj/MathGeoLib/blob/55053da5e3e55a83043af7324944407b174c3724/src/Geometry/OBB.cpp#L987
          This algorithm is the best choice if accuracy is most important.

Returns:
    The oriented bounding box.
    
    For the ``PCA`` method the bounding box is oriented such that the axes are 
    ordered with respect to the principal components.

    For the ``MINIMAL_JYLANKI`` method the bounding box is oriented to be closest 
    to the identity rotation matrix.

Example:
    This example shows a comparison of all algorithms::

        import open3d as o3d

        bunny = o3d.data.BunnyMesh()
        mesh = o3d.t.io.read_triangle_mesh(bunny.path)

        methods = [
            o3d.t.geometry.PCA,
            o3d.t.geometry.MINIMAL_APPROX,
            o3d.t.geometry.MINIMAL_JYLANKI,
        ]

        obbs = []
        for method, color in zip(methods, [(1,0,0),(0,1,0),(0,0,1)]):
            obb = o3d.t.geometry.OrientedBoundingBox.create_from_points(
                mesh.vertex.positions, method=method)
            obb.set_color(color)
            obbs.append(obb)
            print(f"Volume is {obb.volume()} with {method.name}")
            
        o3d.visualization.draw([obb.to_legacy() for obb in obbs] + [mesh])
)",
                   "points"_a, "robust"_a = false,
                   "method"_a = MethodOBBCreate::MINIMAL_APPROX);

    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "set_center",
            {{"center",
              "Tensor with {3,} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "set_extent",
            {{"extent",
              "Tensor with {3,} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "set_rotation",
            {{"rotation",
              "Tensor with {3, 3} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "set_color",
            {{"color",
              "Tensor with {3,} shape, and type float32 or float64, with "
              "values in range [0.0, 1.0]."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "translate",
            {{"translation",
              "Translation tensor of shape {3,}, type float32 or float64, "
              "device same as the box."},
             {"relative", "Whether to perform relative translation."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "rotate",
            {{"rotation",
              "Rotation matrix of shape {3, 3}, type float32 or float64, "
              "device same as the box."},
             {"center",
              "Center of the rotation, default is null, which means use center "
              "of the box as rotation center."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "scale",
            {{"scale", "The scale parameter."},
             {"center",
              "Center used for the scaling operation. Tensor with {3,} shape, "
              "and type float32 or float64"}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "get_point_indices_within_bounding_box",
            {{"points",
              "Tensor with {N, 3} shape, and type float32 or float64."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "create_from_axis_aligned_bounding_box",
            {{"aabb",
              "AxisAlignedBoundingBox object from which OrientedBoundingBox is "
              "created."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "create_from_points",
            {{"points",
              "A list of points with data type of float32 or float64 (N x 3 "
              "tensor, where N must be larger than 3)."},
             {"robust",
              "If set to true uses a more robust method which works in "
              "degenerate cases but introduces noise to the points "
              "coordinates."}});
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
