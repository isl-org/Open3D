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

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAUtils.h"
#endif

namespace open3d {
namespace tests {

class PermuteDtypesWithBool : public testing::TestWithParam<core::Dtype> {
public:
    static std::vector<core::Dtype> TestCases() {
        return {
                core::Bool,  core::UInt8,   core::Int8,    core::UInt16,
                core::Int16, core::UInt32,  core::Int32,   core::UInt64,
                core::Int64, core::Float32, core::Float64,
        };
    }
};

class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases() {
#ifdef BUILD_CUDA_MODULE
        const int device_count = core::cuda::DeviceCount();
        if (device_count >= 1) {
            return {
                    core::Device("CPU:0"),
                    core::Device("CUDA:0"),
            };
        } else {
            return {
                    core::Device("CPU:0"),
            };
        }
#else
        return {
                core::Device("CPU:0"),
        };
#endif
    }
};

class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases() {
#ifdef BUILD_CUDA_MODULE
        const int device_count = core::cuda::DeviceCount();
        if (device_count > 1) {
            // To test multiple CUDA devices, we only need to test CUDA 0 and 1.
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},    // 0
                    {core::Device("CPU:0"), core::Device("CUDA:0")},   // 1
                    {core::Device("CPU:0"), core::Device("CUDA:1")},   // 2
                    {core::Device("CUDA:0"), core::Device("CPU:0")},   // 3
                    {core::Device("CUDA:0"), core::Device("CUDA:0")},  // 4
                    {core::Device("CUDA:0"), core::Device("CUDA:1")},  // 5
                    {core::Device("CUDA:1"), core::Device("CPU:0")},   // 6
                    {core::Device("CUDA:1"), core::Device("CUDA:0")},  // 7
                    {core::Device("CUDA:1"), core::Device("CUDA:1")},  // 8
            };
        } else if (device_count == 1) {
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},
                    {core::Device("CPU:0"), core::Device("CUDA:0")},
                    {core::Device("CUDA:0"), core::Device("CPU:0")},
                    {core::Device("CUDA:0"), core::Device("CUDA:0")},
            };
        } else {
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},
            };
        }
#else
        return {
                {core::Device("CPU:0"), core::Device("CPU:0")},
        };
#endif
    }
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases() {
        return {
                {{}, {}},
                {{0}, {1}},
                {{0, 0}, {1, 1}},
                {{0, 1}, {1, 1}},
                {{1, 0}, {1, 1}},
                {{1}, {1}},
                {{1, 2}, {2, 1}},
                {{1, 2, 3}, {6, 3, 1}},
                {{4, 3, 2}, {6, 2, 1}},
                {{2, 0, 3}, {3, 3, 1}},
        };
    }
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases() {
        std::vector<int64_t> tensor_sizes{
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
        // clang-format off
        std::vector<int64_t> large_sizes{
                (1 << 6 ) - 1, (1 << 6 ), (1 << 6 ) + 1,
                (1 << 10) - 6, (1 << 10), (1 << 10) + 6,
                (1 << 15) - 7, (1 << 15), (1 << 15) + 7,
                (1 << 20) - 1, (1 << 20), (1 << 20) + 1,
                (1 << 25) - 2, (1 << 25), (1 << 25) + 2, // ~128MB for float32
        };
        // clang-format on
        tensor_sizes.insert(tensor_sizes.end(), large_sizes.begin(),
                            large_sizes.end());
        return tensor_sizes;
    }
};

}  // namespace tests
}  // namespace open3d
