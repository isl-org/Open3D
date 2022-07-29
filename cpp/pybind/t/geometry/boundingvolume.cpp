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

#include "open3d/t/geometry/BoundingVolume.h"

#include <string>

#include "open3d/core/CUDAUtils.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_boundingvolume(py::module& m) {
    py::class_<AxisAlignedBoundingBox, PyGeometry<AxisAlignedBoundingBox>,
               std::shared_ptr<AxisAlignedBoundingBox>, Geometry,
               DrawableGeometry>
            aabb(m, "AxisAlignedBoundingBox",
                 R"(A bounding box that is aligned along the coordinate axes
and defined by the min_bound and max_bound."
- (min_bound, max_bound): Lower and upper bounds of the bounding box for all
axes.
    - Usage
        - AxisAlignedBoundingBox::GetMinBound()
        - AxisAlignedBoundingBox::SetMinBound(const core::Tensor &min_bound)
        - AxisAlignedBoundingBox::GetMaxBound()
        - AxisAlignedBoundingBox::SetMaxBound(const core::Tensor &max_bound)
    - Value tensor must have shape {3,}.
    - Value tensor must have the same data type and device.
    - Value tensor can only be float32 (default) or float64.
    - The device of the tensor determines the device of the box.

- color: Color of the bounding box.
    - Usage
        - AxisAlignedBoundingBox::GetColor()
        - AxisAlignedBoundingBox::SetColor(const core::Tensor &color)
    - Value tensor must have shape {3,}.
    - Value tensor can only be float32 (default) or float64.
    - Value tensor can only be range [0.0, 1.0].)");
    aabb.def(py::init<const core::Device&>(),
             "device"_a = core::Device("CPU:0"),
             "Construct an empty axis-aligned box on the provided "
             "device.");
    aabb.def(py::init<const core::Tensor&, const core::Tensor&>(),
             "min_bound"_a, "max_bound"_a,
             R"(Construct an  axis-aligned box from min/max bound.
The axis-aligned box will be created on the device of the given bound 
tensor, which must be on the same device and have the same data type.)");
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "__init__",
            {{"min_bound",
              "Lower bounds of the bounding box for all axes. Tensor with {3,} "
              "shape, and type float32 or float64"},
             {"max_bound",
              "Upper bounds of the bounding box for all axes. Tensor with {3,} "
              "shape, and type float32 or float64"}});

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
    aabb.def("get_min_bound", &AxisAlignedBoundingBox::GetMinBound,
             "Returns the min bound for box coordinates.");
    aabb.def("get_max_bound", &AxisAlignedBoundingBox::GetMaxBound,
             "Returns the max bound for box coordinates.");
    aabb.def("get_color", &AxisAlignedBoundingBox::GetColor,
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
\f$mi = c + s (mi - c)\f$ and \f$ma = c + s (ma - c)\f$.)",
             "scale"_a, "center"_a);

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
              "tensor, where N must be larger than 3)."}});
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
