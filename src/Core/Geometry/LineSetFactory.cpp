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

#include "LineSet.h"

#include <Eigen/Dense>
#include <Core/Geometry/PointCloud.h>

namespace open3d{

std::shared_ptr<LineSet> CreateLineSetFromPointCloudCorrespondences(
        const PointCloud &cloud0, const PointCloud &cloud1,
        const std::vector<std::pair<int, int>> &correspondences)
{
    auto lineset_ptr = std::make_shared<LineSet>();
    size_t point0_size = cloud0.points_.size();
    size_t point1_size = cloud1.points_.size();
    lineset_ptr->points_.resize(point0_size + point1_size);
    for (size_t i = 0; i < point0_size; i++)
        lineset_ptr->points_[i] = cloud0.points_[i];
    for (size_t i = 0; i < point1_size; i++)
        lineset_ptr->points_[point0_size + i] = cloud1.points_[i];
    
    size_t corr_size = correspondences.size();
    lineset_ptr->lines_.resize(corr_size);
    for (size_t i = 0; i < corr_size; i++)
        lineset_ptr->lines_[i] = Eigen::Vector2i(
            correspondences[i].first, point0_size + correspondences[i].second);
    return lineset_ptr;
}

}    // namespace open3d
