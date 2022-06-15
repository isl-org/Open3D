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
