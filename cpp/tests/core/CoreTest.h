// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class PermuteDtypesWithBool : public testing::TestWithParam<core::Dtype> {
public:
    static std::vector<core::Dtype> TestCases();
};

/// Permute one device for each device type, in {CPU, CUDA}.
/// PermuteDevicesWithSYCL should be used if SYCL support is implemented.
class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases();
};

/// Permute one device for each device type, in {CPU, CUDA, SYCL}.
class PermuteDevicesWithSYCL : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases();
};

/// Permute device pairs, in {CPU, CUDA}.
/// PermuteDevicePairsWithSYCL should be used if SYCL support is implemented.
class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases();
};

/// Permute device pairs, in {CPU, CUDA, SYCL}.
class PermuteDevicePairsWithSYCL
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases();
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases();
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases();
};

}  // namespace tests
}  // namespace open3d
