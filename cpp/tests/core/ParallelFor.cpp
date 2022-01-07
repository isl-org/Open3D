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

#include "open3d/core/ParallelFor.h"

#include <vector>

#include "open3d/Macro.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

#ifdef BUILD_ISPC_MODULE
#include "ParallelFor_ispc.h"
#endif

namespace open3d {
namespace tests {

TEST(ParallelFor, LambdaCPU) {
    const core::Device device("CPU:0");
    const size_t N = 10000000;
    core::Tensor tensor({N, 1}, core::Int64, device);

    core::ParallelFor(device, tensor.NumElements(), [&](int64_t idx) {
        tensor.GetDataPtr<int64_t>()[idx] = idx;
    });

    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        ASSERT_EQ(tensor.GetDataPtr<int64_t>()[i], i);
    }
}

TEST(ParallelFor, VectorizedLambda1) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);

    core::ParallelFor(
            core::Device("CPU:0"), v.size(), [&](int64_t idx) { v[idx] = idx; },
            OPEN3D_VECTORIZED(ISPCKernel1, v.data()));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
}

TEST(ParallelFor, VectorizedLambda2) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
            },
            OPEN3D_VECTORIZED(ISPCKernel2, v.data(), &arg2));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
}

TEST(ParallelFor, VectorizedLambda3) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
            },
            OPEN3D_VECTORIZED(ISPCKernel3, v.data(), &arg2, &arg3));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
}

TEST(ParallelFor, VectorizedLambda4) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
            },
            OPEN3D_VECTORIZED(ISPCKernel4, v.data(), &arg2, &arg3, &arg4));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
}

TEST(ParallelFor, VectorizedLambda5) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
            },
            OPEN3D_VECTORIZED(ISPCKernel5, v.data(), &arg2, &arg3, &arg4,
                              &arg5));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
    EXPECT_EQ(arg5, static_cast<uint16_t>(5));
}

TEST(ParallelFor, VectorizedLambda6) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
            },
            OPEN3D_VECTORIZED(ISPCKernel6, v.data(), &arg2, &arg3, &arg4, &arg5,
                              &arg6));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
    EXPECT_EQ(arg5, static_cast<uint16_t>(5));
    EXPECT_EQ(arg6, static_cast<int16_t>(6));
}

TEST(ParallelFor, VectorizedLambda7) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
                arg7 = 7;
            },
            OPEN3D_VECTORIZED(ISPCKernel7, v.data(), &arg2, &arg3, &arg4, &arg5,
                              &arg6, &arg7));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
    EXPECT_EQ(arg5, static_cast<uint16_t>(5));
    EXPECT_EQ(arg6, static_cast<int16_t>(6));
    EXPECT_EQ(arg7, static_cast<uint32_t>(7));
}

TEST(ParallelFor, VectorizedLambda8) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;
    int32_t arg8 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
                arg7 = 7;
                arg8 = 8;
            },
            OPEN3D_VECTORIZED(ISPCKernel8, v.data(), &arg2, &arg3, &arg4, &arg5,
                              &arg6, &arg7, &arg8));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
    EXPECT_EQ(arg5, static_cast<uint16_t>(5));
    EXPECT_EQ(arg6, static_cast<int16_t>(6));
    EXPECT_EQ(arg7, static_cast<uint32_t>(7));
    EXPECT_EQ(arg8, static_cast<int32_t>(8));
}

TEST(ParallelFor, VectorizedLambda9) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;
    int32_t arg8 = 0;
    uint64_t arg9 = 0;

    core::ParallelFor(
            core::Device("CPU:0"), v.size(),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
                arg7 = 7;
                arg8 = 8;
                arg9 = 9;
            },
            OPEN3D_VECTORIZED(ISPCKernel9, v.data(), &arg2, &arg3, &arg4, &arg5,
                              &arg6, &arg7, &arg8, &arg9));

    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    EXPECT_EQ(arg4, static_cast<int8_t>(4));
    EXPECT_EQ(arg5, static_cast<uint16_t>(5));
    EXPECT_EQ(arg6, static_cast<int16_t>(6));
    EXPECT_EQ(arg7, static_cast<uint32_t>(7));
    EXPECT_EQ(arg8, static_cast<int32_t>(8));
    EXPECT_EQ(arg9, static_cast<uint64_t>(9));
}

template <typename T>
int64_t GetInitialValue() = delete;

#define GET_INITIAL_VALUE(T, value) \
    template <>                     \
    int64_t GetInitialValue<T>() {  \
        return value;               \
    }

GET_INITIAL_VALUE(bool, -1)
GET_INITIAL_VALUE(uint8_t, -2)
GET_INITIAL_VALUE(int8_t, -3)
GET_INITIAL_VALUE(uint16_t, -4)
GET_INITIAL_VALUE(int16_t, -5)
GET_INITIAL_VALUE(uint32_t, -6)
GET_INITIAL_VALUE(int32_t, -7)
GET_INITIAL_VALUE(uint64_t, -8)
GET_INITIAL_VALUE(int64_t, -9)
GET_INITIAL_VALUE(float, -10)
GET_INITIAL_VALUE(double, -11)

#undef GET_INITIAL_VALUE

class ParallelForPermuteDtypesWithBool : public PermuteDtypesWithBool {};
INSTANTIATE_TEST_SUITE_P(ParallelFor,
                         ParallelForPermuteDtypesWithBool,
                         testing::ValuesIn(PermuteDtypesWithBool::TestCases()));

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda1) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel1,
                                           v.data()));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda2) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel2,
                                           v.data(), &arg2));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda3) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel3,
                                           v.data(), &arg2, &arg3));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda4) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel4,
                                           v.data(), &arg2, &arg3, &arg4));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda5) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel5,
                                           v.data(), &arg2, &arg3, &arg4,
                                           &arg5));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
        EXPECT_EQ(arg5, static_cast<uint16_t>(5));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda6) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel6,
                                           v.data(), &arg2, &arg3, &arg4, &arg5,
                                           &arg6));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
        EXPECT_EQ(arg5, static_cast<uint16_t>(5));
        EXPECT_EQ(arg6, static_cast<int16_t>(6));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda7) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                    arg7 = 7;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel7,
                                           v.data(), &arg2, &arg3, &arg4, &arg5,
                                           &arg6, &arg7));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
        EXPECT_EQ(arg5, static_cast<uint16_t>(5));
        EXPECT_EQ(arg6, static_cast<int16_t>(6));
        EXPECT_EQ(arg7, static_cast<uint32_t>(7));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda8) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;
    int32_t arg8 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                    arg7 = 7;
                    arg8 = 8;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel8,
                                           v.data(), &arg2, &arg3, &arg4, &arg5,
                                           &arg6, &arg7, &arg8));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
        EXPECT_EQ(arg5, static_cast<uint16_t>(5));
        EXPECT_EQ(arg6, static_cast<int16_t>(6));
        EXPECT_EQ(arg7, static_cast<uint32_t>(7));
        EXPECT_EQ(arg8, static_cast<int32_t>(8));
    });
}

TEST_P(ParallelForPermuteDtypesWithBool, VectorizedTemplateLambda9) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;
    int32_t arg8 = 0;
    uint64_t arg9 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelFor(
                core::Device("CPU:0"), v.size(),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                    arg7 = 7;
                    arg8 = 8;
                    arg9 = 9;
                },
                OPEN3D_TEMPLATE_VECTORIZED(scalar_t, TemplateISPCKernel9,
                                           v.data(), &arg2, &arg3, &arg4, &arg5,
                                           &arg6, &arg7, &arg8, &arg9));

        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, static_cast<uint8_t>(3));
        EXPECT_EQ(arg4, static_cast<int8_t>(4));
        EXPECT_EQ(arg5, static_cast<uint16_t>(5));
        EXPECT_EQ(arg6, static_cast<int16_t>(6));
        EXPECT_EQ(arg7, static_cast<uint32_t>(7));
        EXPECT_EQ(arg8, static_cast<int32_t>(8));
        EXPECT_EQ(arg9, static_cast<uint64_t>(9));
    });
}

}  // namespace tests
}  // namespace open3d
