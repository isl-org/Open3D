// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/t/geometry/Image.h"

#include <string>
#include <unordered_map>

#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_image(py::module& m) {
    py::class_<Image, PyGeometry<Image>, std::unique_ptr<Image>, Geometry>
            image(m, "Image", "An image contains a (multi-channel) 2D image.");

    // Constructors.
    image.def(py::init<int64_t, int64_t, int64_t, core::Dtype,
                       const core::Device&>(),
              "rows"_a = 0, "cols"_a = 0, "channels"_a = 1,
              "dtype"_a = core::Dtype::Float32,
              "device"_a = core::Device("CPU:0"))
            .def(py::init<const core::Tensor&>(), "image"_a);

    // Conversions.
    image.def("as_tensor", &Image::AsTensor);
    image.def_static("from_legacy_image", &Image::FromLegacyImage,
                     "image_legacy"_a, "device"_a = core::Device("CPU:0"),
                     "Create a Image from a legacy Open3D Image.");
    image.def("to_legacy_image", &Image::ToLegacyImage,
              "Convert to a legacy Open3D Image.");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
