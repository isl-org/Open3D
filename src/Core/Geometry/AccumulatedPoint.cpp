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

#include "AccumulatedPoint.h"
#include "PointCloud.h"

using namespace open3d;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
AccumulatedPoint::AccumulatedPoint() :
        num_of_points_(0),
        point_(0.0, 0.0, 0.0),
        normal_(0.0, 0.0, 0.0),
        color_(0.0, 0.0, 0.0)
{
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void AccumulatedPoint::AddPoint(const PointCloud &cloud, int index)
{
    point_ += cloud.points_[index];
    if (cloud.HasNormals()) {
        if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
            normal_ += cloud.normals_[index];
        }
    }
    if (cloud.HasColors()) {
        color_ += cloud.colors_[index];
    }
    num_of_points_++;
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
Eigen::Vector3d AccumulatedPoint::GetAveragePoint() const
{
    return point_ / double(num_of_points_);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
Eigen::Vector3d AccumulatedPoint::GetAverageNormal() const
{
    return normal_.normalized();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
Eigen::Vector3d AccumulatedPoint::GetAverageColor() const
{
    return color_ / double(num_of_points_);
}
