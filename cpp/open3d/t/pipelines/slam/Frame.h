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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slam {

/// \brief Frame is a container class storing an intrinsic matrix and several 2D
/// tensors, from depth map, vertex map, to color map, in a customized way.
class Frame {
public:
    Frame(int height,
          int width,
          const core::Tensor& intrinsics,
          const core::Device& device)
        : height_(height),
          width_(width),
          intrinsics_(intrinsics),
          device_(device) {}

    int GetHeight() const { return height_; }
    int GetWidth() const { return width_; }

    void SetIntrinsics(const core::Tensor& intrinsics) {
        intrinsics_ = intrinsics;
    }
    core::Tensor GetIntrinsics() const { return intrinsics_; }

    void SetData(const std::string& name, const core::Tensor& data) {
        data_[name] = data.To(device_);
    }
    core::Tensor GetData(const std::string& name) const {
        if (data_.count(name) == 0) {
            utility::LogWarning(
                    "Property not found for {}, return an empty tensor!", name);
            return core::Tensor();
        }
        return data_.at(name);
    }

    // Convenient interface for images
    void SetDataFromImage(const std::string& name,
                          const t::geometry::Image& data) {
        SetData(name, data.AsTensor());
    }

    t::geometry::Image GetDataAsImage(const std::string& name) const {
        return t::geometry::Image(GetData(name));
    }

private:
    int height_;
    int width_;

    // (3, 3) intrinsic matrix for a pinhole camera
    core::Tensor intrinsics_;
    core::Device device_;

    // Maintained maps, including:
    // depth_map: (H, W, 1), Float32 AFTER preprocessing
    // vertex_map: (H, W, 3), Float32,
    // color_map: (H, W, 3), Float32
    // normal_map: (H, W, 3), Float32
    std::unordered_map<std::string, core::Tensor> data_;
};

}  // namespace slam
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
