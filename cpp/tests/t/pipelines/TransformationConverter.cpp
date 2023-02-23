// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/TransformationConverter.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TransformationConverterPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TransformationConverter,
                         TransformationConverterPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TransformationConverterPermuteDevices, RtToTransformation) {
    core::Device device = GetParam();

    for (const core::Dtype& dtype : {core::Float32, core::Float64}) {
        core::Tensor rotation = core::Tensor::Eye(3, dtype, device);
        core::Tensor translation = core::Tensor::Zeros({3}, dtype, device);
        core::Tensor transformation_ =
                t::pipelines::kernel::RtToTransformation(rotation, translation);

        core::Tensor transformation = core::Tensor::Eye(4, dtype, device);
        EXPECT_TRUE(transformation_.AllClose(transformation));
    }
}

TEST_P(TransformationConverterPermuteDevices, PoseToTransformation) {
    core::Device device = GetParam();

    for (const core::Dtype& dtype : {core::Float32, core::Float64}) {
        core::Tensor pose = core::Tensor::Zeros({6}, dtype, device);
        core::Tensor transformation_ =
                t::pipelines::kernel::PoseToTransformation(pose);

        core::Tensor transformation = core::Tensor::Eye(4, dtype, device);
        EXPECT_TRUE(transformation_.AllClose(transformation));
    }
}

}  // namespace tests
}  // namespace open3d
