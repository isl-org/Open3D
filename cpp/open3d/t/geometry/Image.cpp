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

#include "open3d/t/geometry/Image.h"

#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

Image::Image(int64_t rows,
             int64_t cols,
             int64_t channels,
             core::Dtype dtype,
             const core::Device &device)
    : Geometry2D(Geometry::GeometryType::Image) {
    if (rows < 0) {
        utility::LogError("rows must be >= 0, but got {}.", rows);
    }
    if (cols < 0) {
        utility::LogError("cols must be >= 0, but got {}.", cols);
    }
    if (channels <= 0) {
        utility::LogError("channels must be > 0, but got {}.", channels);
    }
    data_ = core::Tensor({rows, cols, channels}, dtype, device);
}

Image::Image(const core::Tensor &tensor)
    : Geometry2D(Geometry::GeometryType::Image) {
    if (!tensor.IsContiguous()) {
        utility::LogError("Input tensor must be contiguous.");
    }
    if (tensor.NumDims() == 2) {
        data_ = tensor.Reshape(
                core::shape_util::Concat(tensor.GetShape(), {1}));
    } else if (tensor.NumDims() == 3) {
        data_ = tensor;
    } else {
        utility::LogError("Input tensor must be 2-D or 3-D, but got shape {}.",
                          tensor.GetShape().ToString());
    }
}

open3d::geometry::Image Image::ToLegacyImage() const {
    open3d::geometry::Image image_legacy;
    size_t elem_sz = data_.GetDtype().ByteSize();
    image_legacy.Prepare(GetCols(), GetRows(), GetChannels(), elem_sz);
    if (data_.IsContiguous()) {
        memcpy(image_legacy.data_.data(), data_.GetDataPtr(),
               image_legacy.data_.size());
    } else {
        for (int64_t i = 0, i_leg = 0; i < GetRows(); ++i)
            for (int64_t j = 0; i < GetCols(); ++j)
                for (int64_t k = 0; i < GetChannels(); ++k, i_leg += elem_sz)
                    // image_legacy is contiguous
                    memcpy(&image_legacy.data_[i_leg],
                           data_[i][j][k].GetDataPtr(), elem_sz);
    }
    return image_legacy;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
