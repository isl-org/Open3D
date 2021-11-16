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

TEST(Helper, UniformRandIntGeneratorWithFixedSeed) {
    std::array<int, 1024> values;
    utility::UniformRandIntGenerator rand_generator(0, 9, 42);
    for (auto it = values.begin(); it != values.end(); ++it)
        *it = rand_generator();

    for (int i = 0; i < 10; i++) {
        std::array<int, 1024> new_values;
        utility::UniformRandIntGenerator new_rand_generator(0, 9, 42);
        for (auto it = new_values.begin(); it != new_values.end(); ++it)
            *it = new_rand_generator();
        EXPECT_TRUE(values == new_values);
    }
}

TEST(Helper, UniformRandIntGeneratorWithRandomSeed) {
    std::array<int, 1024> values;
    utility::UniformRandIntGenerator rand_generator(0, 9);
    for (auto it = values.begin(); it != values.end(); ++it)
        *it = rand_generator();

    for (int i = 0; i < 10; i++) {
        std::array<int, 1024> new_values;
        utility::UniformRandIntGenerator new_rand_generator(0, 9);
        for (auto it = new_values.begin(); it != new_values.end(); ++it)
            *it = new_rand_generator();
        EXPECT_FALSE(values == new_values);
    }
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
