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

#include "open3d/core/ParallelForVectorized.h"

#include <vector>

#include "open3d/Macro.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

#ifdef BUILD_ISPC_MODULE
#include "ParallelForVectorized_ispc.h"
#endif

namespace open3d {
namespace tests {

TEST(ParallelForVectorized, Lambda1) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel1, v.data()),
            [&](int64_t idx) { v[idx] = idx; });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
}

TEST(ParallelForVectorized, Lambda2) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel2, v.data(), &arg2),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
}

TEST(ParallelForVectorized, Lambda3) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel3, v.data(), &arg2, &arg3),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
}

TEST(ParallelForVectorized, Lambda4) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel4, v.data(), &arg2, &arg3,
                                     &arg4),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
}

TEST(ParallelForVectorized, Lambda5) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel5, v.data(), &arg2, &arg3,
                                     &arg4, &arg5),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
    EXPECT_EQ(arg5, 5);
}

TEST(ParallelForVectorized, Lambda6) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel6, v.data(), &arg2, &arg3,
                                     &arg4, &arg5, &arg6),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
    EXPECT_EQ(arg5, 5);
    EXPECT_EQ(arg6, 6);
}

TEST(ParallelForVectorized, Lambda7) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel7, v.data(), &arg2, &arg3,
                                     &arg4, &arg5, &arg6, &arg7),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
                arg7 = 7;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
    EXPECT_EQ(arg5, 5);
    EXPECT_EQ(arg6, 6);
    EXPECT_EQ(arg7, 7);
}

TEST(ParallelForVectorized, Lambda8) {
    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;
    uint32_t arg7 = 0;
    int32_t arg8 = 0;

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel8, v.data(), &arg2, &arg3,
                                     &arg4, &arg5, &arg6, &arg7, &arg8),
            [&](int64_t idx) {
                v[idx] = idx;
                arg2 = true;
                arg3 = 3;
                arg4 = 4;
                arg5 = 5;
                arg6 = 6;
                arg7 = 7;
                arg8 = 8;
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
    EXPECT_EQ(arg5, 5);
    EXPECT_EQ(arg6, 6);
    EXPECT_EQ(arg7, 7);
    EXPECT_EQ(arg8, 8);
}

TEST(ParallelForVectorized, Lambda9) {
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

    core::ParallelForVectorized(
            core::Device("CPU:0"), v.size(),
            OPEN3D_VECTORIZED_LAMBDA(LambdaKernel9, v.data(), &arg2, &arg3,
                                     &arg4, &arg5, &arg6, &arg7, &arg8, &arg9),
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
            });

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i);
    }
    EXPECT_TRUE(arg2);
    EXPECT_EQ(arg3, 3);
    EXPECT_EQ(arg4, 4);
    EXPECT_EQ(arg5, 5);
    EXPECT_EQ(arg6, 6);
    EXPECT_EQ(arg7, 7);
    EXPECT_EQ(arg8, 8);
    EXPECT_EQ(arg9, 9);
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

class PFVPermuteDtypesWithBool : public PermuteDtypesWithBool {};
INSTANTIATE_TEST_SUITE_P(ParallelForVectorized,
                         PFVPermuteDtypesWithBool,
                         testing::ValuesIn(PermuteDtypesWithBool::TestCases()));

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda1) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel1, v.data()),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda2) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel2, v.data(), &arg2),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda3) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(scalar_t,
                                                  TemplateLambdaKernel3,
                                                  v.data(), &arg2, &arg3),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda4) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel4, v.data(), &arg2, &arg3,
                        &arg4),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda5) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel5, v.data(), &arg2, &arg3,
                        &arg4, &arg5),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
        EXPECT_EQ(arg5, 5);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda6) {
    core::Dtype dtype = GetParam();

    const size_t N = 10000000;
    std::vector<int64_t> v(N);
    bool arg2 = false;
    uint8_t arg3 = 0;
    int8_t arg4 = 0;
    uint16_t arg5 = 0;
    int16_t arg6 = 0;

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel6, v.data(), &arg2, &arg3,
                        &arg4, &arg5, &arg6),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
        EXPECT_EQ(arg5, 5);
        EXPECT_EQ(arg6, 6);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda7) {
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
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel7, v.data(), &arg2, &arg3,
                        &arg4, &arg5, &arg6, &arg7),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                    arg7 = 7;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
        EXPECT_EQ(arg5, 5);
        EXPECT_EQ(arg6, 6);
        EXPECT_EQ(arg7, 7);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda8) {
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
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel8, v.data(), &arg2, &arg3,
                        &arg4, &arg5, &arg6, &arg7, &arg8),
                [&](int64_t idx) {
                    v[idx] = idx == 0 ? GetInitialValue<scalar_t>() : idx;
                    arg2 = true;
                    arg3 = 3;
                    arg4 = 4;
                    arg5 = 5;
                    arg6 = 6;
                    arg7 = 7;
                    arg8 = 8;
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
        EXPECT_EQ(arg5, 5);
        EXPECT_EQ(arg6, 6);
        EXPECT_EQ(arg7, 7);
        EXPECT_EQ(arg8, 8);
    });
}

TEST_P(PFVPermuteDtypesWithBool, TemplateLambda9) {
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
        core::ParallelForVectorized(
                core::Device("CPU:0"), v.size(),
                OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(
                        scalar_t, TemplateLambdaKernel9, v.data(), &arg2, &arg3,
                        &arg4, &arg5, &arg6, &arg7, &arg8, &arg9),
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
                });

        for (size_t i = 0; i < v.size(); ++i) {
            ASSERT_EQ(v[i], i == 0 ? GetInitialValue<scalar_t>() : i);
        }
        EXPECT_TRUE(arg2);
        EXPECT_EQ(arg3, 3);
        EXPECT_EQ(arg4, 4);
        EXPECT_EQ(arg5, 5);
        EXPECT_EQ(arg6, 6);
        EXPECT_EQ(arg7, 7);
        EXPECT_EQ(arg8, 8);
        EXPECT_EQ(arg9, 9);
    });
}

}  // namespace tests
}  // namespace open3d
