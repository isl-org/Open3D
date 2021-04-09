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

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/geometry/TSDFVoxelGrid.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace voxelhashing {
class Frame {
public:
    Frame(const core::Tensor& intrinsics) : intrinsics_(intrinsics) {}

    void SetRGBD(const t::geometry::RGBDImage& rgbd) {
        SetData("color", rgbd.color_.AsTensor());
        SetData("depth", rgbd.depth_.AsTensor());
    }

    void SetIntrinsics(const core::Tensor& intrinsics) {
        intrinsics_ = intrinsics;
    }
    core::Tensor GetIntrinsics() const { return intrinsics_; }

    void SetData(const std::string& name, const core::Tensor& data) {
        if (data_.count(name) != 0) {
            data_.at(name) = data;
        } else {
            data_.emplace(name, data);
        }
    }

private:
    // (3, 3) intrinsic matrix for a pinhole camera
    core::Tensor intrinsics_;
    // Possibly maintained maps, including:
    // vertex, color, depth
    std::unordered_map<std::string, core::Tensor> data_;
};
}  // namespace voxelhashing
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
