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

#include "open3d/t/pipelines/kernel/TransformationConverter.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TransformationConverterPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TransformationConverter,
                         TransformationConverterPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TransformationConverterPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TransformationConverter,
        TransformationConverterPermuteDevicePairs,
        testing::ValuesIn(
                TransformationConverterPermuteDevicePairs::TestCases()));

TEST_P(TransformationConverterPermuteDevices, RtToTransformation) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> rotation_vec{1, 0, 0, 0, 1, 0, 0, 0, 1};
    core::Tensor rotation(rotation_vec, {3, 3}, dtype, device);

    std::vector<float> translation_vec{0, 0, 0};
    core::Tensor translation(translation_vec, {3}, dtype, device);

    std::vector<float> transformation_vec{1, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 1};
    core::Tensor transformation(transformation_vec, {4, 4}, dtype, device);

    core::Tensor transformation_ =
            t::pipelines::kernel::RtToTransformation(rotation, translation);

    EXPECT_EQ(transformation.ToFlatVector<float>(),
              transformation_.ToFlatVector<float>());
}

TEST_P(TransformationConverterPermuteDevices, PoseToTransformation) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> pose_vec{0, 0, 0, 0, 0, 0};
    core::Tensor pose(pose_vec, {6}, dtype, device);

    std::vector<float> transformation_vec{1, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 1};
    core::Tensor transformation(transformation_vec, {4, 4}, dtype, device);

    core::Tensor transformation_ =
            t::pipelines::kernel::PoseToTransformation(pose);

    EXPECT_EQ(transformation.ToFlatVector<float>(),
              transformation_.ToFlatVector<float>());
}

}  // namespace tests
}  // namespace open3d
