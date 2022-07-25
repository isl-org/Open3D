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

#include "open3d/utility/Random.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Random, UniformRandIntGeneratorWithFixedSeed) {
    utility::random::Seed(42);
    std::array<int, 1024> values;
    utility::random::UniformIntGenerator rand_generator(0, 9);
    for (auto it = values.begin(); it != values.end(); ++it) {
        *it = rand_generator();
    }

    for (int i = 0; i < 10; i++) {
        utility::random::Seed(42);
        std::array<int, 1024> new_values;
        utility::random::UniformIntGenerator new_rand_generator(0, 9);
        for (auto it = new_values.begin(); it != new_values.end(); ++it) {
            *it = new_rand_generator();
        }
        EXPECT_TRUE(values == new_values);
    }
}

TEST(Random, UniformRandIntGeneratorWithRandomSeed) {
    std::array<int, 1024> values;
    utility::random::UniformIntGenerator rand_generator(0, 9);
    for (auto it = values.begin(); it != values.end(); ++it) {
        *it = rand_generator();
    }

    for (int i = 0; i < 10; i++) {
        std::array<int, 1024> new_values;
        utility::random::UniformIntGenerator new_rand_generator(0, 9);
        for (auto it = new_values.begin(); it != new_values.end(); ++it) {
            *it = new_rand_generator();
        }
        EXPECT_FALSE(values == new_values);
    }
}

}  // namespace tests
}  // namespace open3d
