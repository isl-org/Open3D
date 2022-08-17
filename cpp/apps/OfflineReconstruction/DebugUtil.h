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

#include "open3d/Open3D.h"

namespace open3d {
namespace apps {
namespace offline_reconstruction {

static const Eigen::Matrix4d flip_transformation = Eigen::Matrix4d({
        {1, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1},
});

void DrawRegistrationResult(const geometry::PointCloud& src,
                            const geometry::PointCloud& dst,
                            const Eigen::Matrix4d& transformation,
                            bool keep_color = false) {
    auto transformed_src = std::make_shared<geometry::PointCloud>(src);
    auto transformed_dst = std::make_shared<geometry::PointCloud>(dst);
    if (!keep_color) {
        transformed_src->PaintUniformColor(Eigen::Vector3d(1, 0.706, 0));
        transformed_dst->PaintUniformColor(Eigen::Vector3d(0, 0.651, 0.929));
    }
    transformed_src->Transform(flip_transformation * transformation);
    transformed_dst->Transform(flip_transformation);
    visualization::DrawGeometries({transformed_src, transformed_dst});
}

}  // namespace offline_reconstruction
}  // namespace apps
}  // namespace open3d