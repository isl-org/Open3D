// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/tgeometry/Image.h"

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace tgeometry {

Image::Image(int64_t rows,
             int64_t cols,
             int64_t channels,
             core::Dtype dtype,
             const core::Device &device)
    : Geometry(Geometry::GeometryType::Image, 3) {
    if (rows < 0) {
        utility::LogError("rows must be >= 0, but got {}.", rows);
    }
    if (cols < 0) {
        utility::LogError("cols must be >= 0, but got {}.", cols);
    }
    if (channels <= 0) {
        utility::LogError("channels must be > 0, but got {}.", channels);
    }
    if (dtype != core::Dtype::UInt8 && dtype != core::Dtype::Float32 &&
        dtype != core::Dtype::Float64) {
        utility::LogError("Unsupported type {} for image.",
                          core::DtypeUtil::ToString(dtype));
    }
    data_ = core::Tensor({rows, cols, channels}, dtype, device);
}

}  // namespace tgeometry
}  // namespace open3d
