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

#include "open3d/t/geometry/RGBDImage.h"

namespace open3d {
namespace t {
namespace geometry {

RGBDImage &RGBDImage::Clear() {
    color_.Clear();
    depth_.Clear();
    return *this;
}

bool RGBDImage::IsEmpty() const { return color_.IsEmpty() && depth_.IsEmpty(); }

std::string RGBDImage::ToString() const {
    return fmt::format(
            "RGBD Image pair [{}Aligned]\n"
            "Color [size=({},{}), channels={}, format={}, device={}]\n"
            "Depth [size=({},{}), channels={}, format={}, device={}]",
            AreAligned() ? "" : "Not ", color_.GetCols(), color_.GetRows(),
            color_.GetChannels(), color_.GetDtype().ToString(),
            color_.GetDevice().ToString(), depth_.GetCols(), depth_.GetRows(),
            depth_.GetChannels(), depth_.GetDtype().ToString(),
            depth_.GetDevice().ToString());
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
