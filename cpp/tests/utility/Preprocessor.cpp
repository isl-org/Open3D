// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Preprocessor.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Preprocessor, CONCAT) {
#define PREFIX_MACRO 10
#define SUFFIX_MACRO 4

    EXPECT_EQ(OPEN3D_CONCAT(10, 4), 104);
    EXPECT_EQ(OPEN3D_CONCAT(10, SUFFIX_MACRO), 104);
    EXPECT_EQ(OPEN3D_CONCAT(PREFIX_MACRO, 4), 104);
    EXPECT_EQ(OPEN3D_CONCAT(PREFIX_MACRO, SUFFIX_MACRO), 104);

#undef PREFIX_MACRO
#undef SUFFIX_MACRO
}

TEST(Preprocessor, STRINGIFY) {
#define STRING_MACRO test

    EXPECT_EQ(std::string(OPEN3D_STRINGIFY(test)), std::string("test"));
    EXPECT_EQ(std::string(OPEN3D_STRINGIFY(STRING_MACRO)), std::string("test"));

#undef STRING_MACRO
}

TEST(Preprocessor, NUM_ARGS_Zero) {
    // Detecting 0 arguments is not supported and instead returns 1.
    EXPECT_EQ(OPEN3D_NUM_ARGS(), 1);
}

TEST(Preprocessor, NUM_ARGS) {
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2), 2);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3), 3);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4), 4);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5), 5);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6), 6);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6, x7), 7);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6, x7, x8), 8);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6, x7, x8, x9), 9);
    EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), 10);
    // Detecting >10 Arguments is not supported and results in compiler errors.
    // EXPECT_EQ(OPEN3D_NUM_ARGS(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11),
    // 11);
    // ...
}

TEST(Preprocessor, OVERLOAD) {
#define FUNC_1(x1) (x1) + 1
#define FUNC_2(x1, x2) (x1) * (x2)

#define FUNC(...) \
    OPEN3D_FIX_MSVC_(OPEN3D_OVERLOAD(FUNC_, __VA_ARGS__)(__VA_ARGS__))

    EXPECT_EQ(FUNC(0), 0 + 1);
    EXPECT_EQ(FUNC(1), 0 + 2);
    EXPECT_EQ(FUNC(2), 0 + 3);

    EXPECT_EQ(FUNC(0, 1), 0 * 1);
    EXPECT_EQ(FUNC(1, 2), 1 * 2);
    EXPECT_EQ(FUNC(2, 3), 2 * 3);

#undef FUNC

#undef FUNC_1
#undef FUNC_2
}

}  // namespace tests
}  // namespace open3d
