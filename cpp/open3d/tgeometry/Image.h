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

#pragma once

#include <memory>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/tgeometry/Geometry.h"

namespace open3d {
namespace tgeometry {

/// \class Image
///
/// \brief The Image class stores image with customizable rols, cols, channels,
/// dtype and device.
class Image : public Geometry {
public:
    /// \brief Constructor for image.
    ///
    /// Row-major storage is used, similar to OpenCV. Use (row, col, channel)
    /// indexing order for image creation and accessing. In general, (r, c, ch)
    /// are the preferred variable names for consistency, and avoid using width,
    /// height, u, v, x, y for coordinates.
    ///
    /// \param rows Number of rows of the image, i.e. image height. \p rows must
    /// be non-negative.
    /// \param cols Number of columns of the image, i.e. image width. \p cols
    /// must be non-negative.
    /// \param channels Number of channels of the image. E.g. for RGB image,
    /// channels == 3; for grayscale image, channels == 1. \p channels must be
    /// greater than 0.
    /// \param dtype Data type of the image.
    /// \param device Device where the image is stored.
    Image(int64_t rows = 0,
          int64_t cols = 0,
          int64_t channels = 1,
          core::Dtype dtype = core::Dtype::Float32,
          const core::Device &device = core::Device("CPU:0"));

    /// \brief Construct from a tensor. The tensor won't be copied and memory
    /// will be shared.
    ///
    /// \param tensor: Tensor of the image. The tensor must be contiguous. The
    /// tensor must be 2D (rows, cols) or 3D (rows, cols, channels).
    Image(const core::Tensor &tensor);

    virtual ~Image() override {}

public:
    /// Clear image contents by resetting the rows and cols to 0, while
    /// keeping channels, dtype and device unchanged.
    Image &Clear() override {
        data_ = core::Tensor({0, 0, GetChannels()}, GetDtype(), GetDevice());
        return *this;
    }

    /// Returns true if rows * cols * channels == 0.
    bool IsEmpty() const override {
        return GetRows() * GetCols() * GetChannels() == 0;
    }

public:
    /// Get the number of rows of the image.
    int64_t GetRows() const { return data_.GetShape()[0]; }

    /// Get the number of columns of the image.
    int64_t GetCols() const { return data_.GetShape()[1]; }

    /// Get the number of channels of the image.
    int64_t GetChannels() const { return data_.GetShape()[2]; }

    /// Get dtype of the image.
    core::Dtype GetDtype() const { return data_.GetDtype(); }

    /// Get device of the image.
    core::Device GetDevice() const { return data_.GetDevice(); }

    /// Get pixel(s) in the image. If channels == 1, returns a tensor with shape
    /// {}, otherwise returns a tensor with shape {channels,}. The returned
    /// tensor is a slice of the image's tensor, so when modifying the slice,
    /// the original tensor will also be modified.
    core::Tensor At(int64_t r, int64_t c) const {
        if (GetChannels() == 1) {
            return data_[r][c][0];
        } else {
            return data_[r][c];
        }
    }

    /// Get pixel(s) in the image. Returns a tensor with shape {}.
    core::Tensor At(int64_t r, int64_t c, int64_t ch) const {
        return data_[r][c][ch];
    }

    /// Get raw buffer of the Image data.
    void *GetDataPtr() { return data_.GetDataPtr(); }

    /// Get raw buffer of the Image data.
    const void *GetDataPtr() const { return data_.GetDataPtr(); }

    /// Retuns the underlying Tensor of the Image.
    core::Tensor AsTensor() const { return data_; }

protected:
    /// Internal data of the Image, represented as a 3D tensor of shape {rols,
    /// cols, channels}. Image properties can be obtained from the tensor.
    core::Tensor data_;
};

}  // namespace tgeometry
}  // namespace open3d
