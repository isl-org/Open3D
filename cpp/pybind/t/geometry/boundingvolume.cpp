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
                 "Class that defines an axis_aligned box "
                 "that can be computed from 3D "
                 "geometries, The axis aligned bounding "
                 "box uses the coordinate axes for "
                 "bounding box generation. This means that the bounding box is "
                 "oriented along the coordinate axes.");
    aabb.def(py::init<const core::Device&>(),
             "device"_a = core::Device("CPU:0"),
             "Construct an empty AxisAlignedBoundingBox on the provided "
             "device.");
    aabb.def(py::init<const core::Tensor&, const core::Tensor&>(),
             "min_bound"_a, "max_bound"_a,
             R"(Construct an AxisAlignedBoundingBox from min/max bound.
The AxisAlignedBoundingBox will be created on the device of the given bound 
tensor, which must be on the same device.)");
    docstring::ClassMethodDocInject(
            m, "AxisAlignedBoundingBox", "__init__",
            {{"min_bound", "Lower bounds of the bounding box for all axes."},
             {"max_bound", "Upper bounds of the bounding box for all axes."}});

    aabb.def("__repr__", &AxisAlignedBoundingBox::ToString);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d