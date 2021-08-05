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

#include "open3d/t/geometry/LineSet.h"

#include <string>
#include <unordered_map>

#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_lineset(py::module& m) {
    py::class_<LineSet, PyGeometry<LineSet>, std::shared_ptr<LineSet>, Geometry>
            line_set(m, "LineSet", R"(
A LineSet contains points and lines joining them and optionally attributes on
the points and lines.

The LineSet class stores the attribute data in key-value pairs for flexibility,
where the key is a string representing the attribute name and value is a Tensor
containing the attribute data.

By default, there are two sets of dictionaries, i.e. ``points`` and
``lines``. In most cases, the length of an attribute should be
equal to the length of the the data corresponding to the master key. For
instance ``points["colors"]`` should have the same length as
``points["positions"]``.

Although the attributes are all stored in a key-value pair dictionary, the
attributes have different levels:

    - **Level 0**: Default attributes ``points["positions"]`` and ``lines["indices"]``.
        - Created by default, required for all ``LineSet`` s.
        - The "positions" tensor must be of shape (N,3) while the "indices" tensor
          must be of shape (N,2) and DtypeCode Int.
        - The device of "positions" and "indices" must be consistent and they
          determine the device of the LineSet.
    - **Level 1**: Commonly used attributes: line colors.
    - **Level 2**: Custom attributes, e.g. labels.
        - Both are not created by default. Created by users. Level 1 attributes
          have some additional convenience functions in the C++ API compared to
          Level 2 attributes.

For all attributes, the device must be consistent with the device of the
LineSet. Dtype can be different.)");

    // Constructors.
    line_set.def(py::init<const core::Device&>(),
                 "device"_a = core::Device("CPU:0"),
                 "Construct an empty LineSet on the provided device.")
            .def(py::init<const core::Tensor&, const core::Tensor&>(),
                 "points"_a, "lines"_a, R"(
Construct a LineSet from points and lines.

The input tensors will be directly used as the underlying storage of the line
set (no memory copy).  The resulting ``LineSet`` will have the same dtype and device
as the tensor. The device for ``points`` must be consistent with  ``lines``.)");
    docstring::ClassMethodDocInject(
            m, "LineSet", "__init__",
            {{"points", "A tensor with element shape (3,)"},
             {"lines", "A tensor with element shape (2,) and Int Dtype."}});

    // Line set's attributes: points_positions, line_indices, line_colors, etc.
    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    line_set.def_property_readonly(
            "points", py::overload_cast<>(&LineSet::GetPointAttr, py::const_),
            "Dictionary containing point attributes. The primary key "
            "``positions`` contains point positions.");
    line_set.def_property_readonly(
            "lines", py::overload_cast<>(&LineSet::GetLineAttr, py::const_),
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

T (4x4) = [[ R(3x3)  t(3x1) ],
           [ O(1x3)  s(1x1) ]]

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
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
