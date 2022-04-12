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

#include "open3d/pipelines/registration/ColoredICP.h"

#include "open3d/io/PointCloudIO.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(ColoredICP, RegistrationColoredICP) {
    data::DemoICPPointClouds dataset;
    std::shared_ptr<geometry::PointCloud> src =
            io::CreatePointCloudFromFile(dataset.GetPaths()[0]);
    std::shared_ptr<geometry::PointCloud> dst =
            io::CreatePointCloudFromFile(dataset.GetPaths()[1]);

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    auto result = pipelines::registration::RegistrationColoredICP(
            *src, *dst, 0.07, transformation,
            pipelines::registration::TransformationEstimationForColoredICP(),
            pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, 50));
    transformation = result.transformation_;

    Eigen::Matrix4d ref_transformation;
    ref_transformation << 0.748899, 0.00425476, 0.662671, -0.275425, 0.115777,
            0.98376, -0.137159, 0.108227, -0.652492, 0.17944, 0.736244, 1.21057,
            0, 0, 0, 1;
    ExpectEQ(ref_transformation, transformation, /*threshold=*/1e-4);
}

TEST(ColoredICP, DISABLED_ICPConvergenceCriteria) { NotImplemented(); }

}  // namespace tests
}  // namespace open3d
