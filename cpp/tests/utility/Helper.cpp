// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Helper.h"

#include "tests/Tests.h"

#ifdef BUILD_ISPC_MODULE
#include "Helper_ispc.h"
#endif

namespace open3d {
namespace tests {

TEST(Helper, JoinStrings) {
    std::vector<std::string> strings;

    strings = {"a", "b", "c"};
    EXPECT_EQ(utility::JoinStrings(strings), "a, b, c");
    EXPECT_EQ(utility::JoinStrings(strings, "-"), "a-b-c");

    strings = {};
    EXPECT_EQ(utility::JoinStrings(strings), "");
    EXPECT_EQ(utility::JoinStrings(strings, "-"), "");
}

TEST(Helper, StringStartsWith) {
    EXPECT_TRUE(utility::StringStartsWith("abc", "a"));
    EXPECT_TRUE(utility::StringStartsWith("abc", "ab"));
    EXPECT_TRUE(utility::StringStartsWith("abc", "abc"));
    EXPECT_FALSE(utility::StringStartsWith("abc", "abcd"));
    EXPECT_TRUE(utility::StringStartsWith("abc", ""));
    EXPECT_FALSE(utility::StringStartsWith("", "a"));
}

TEST(Helper, StringEndsWith) {
    EXPECT_TRUE(utility::StringEndsWith("abc", "c"));
    EXPECT_TRUE(utility::StringEndsWith("abc", "bc"));
    EXPECT_TRUE(utility::StringEndsWith("abc", "abc"));
    EXPECT_FALSE(utility::StringEndsWith("abc", "abcd"));
    EXPECT_TRUE(utility::StringEndsWith("abc", ""));
    EXPECT_FALSE(utility::StringEndsWith("", "c"));
}

TEST(Helper, CHAR_BIT_constant) {
#ifdef BUILD_ISPC_MODULE
    int32_t value;
    ispc::GetCharBit(&value);

    EXPECT_EQ(value, CHAR_BIT);
#endif
}

TEST(Helper, ENSURE_EXPORTED) {
#ifdef BUILD_ISPC_MODULE
    ispc::NotAutomaticallyExportedStruct s;
    s.i = 1;
    s.b = 255;

    EXPECT_EQ(s.i, 1);
    EXPECT_EQ(s.b, 255);
#endif
}

}  // namespace tests
}  // namespace open3d
