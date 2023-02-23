// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Random.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Random, UniformRandIntGeneratorWithFixedSeed) {
    utility::random::Seed(42);
    std::array<int, 1024> values;
    utility::random::UniformIntGenerator<int> rand_generator(0, 9);
    for (auto it = values.begin(); it != values.end(); ++it) {
        *it = rand_generator();
    }

    for (int i = 0; i < 10; i++) {
        utility::random::Seed(42);
        std::array<int, 1024> new_values;
        utility::random::UniformIntGenerator<int> new_rand_generator(0, 9);
        for (auto it = new_values.begin(); it != new_values.end(); ++it) {
            *it = new_rand_generator();
        }
        EXPECT_TRUE(values == new_values);
    }
}

TEST(Random, UniformRandIntGeneratorWithRandomSeed) {
    std::array<int, 1024> values;
    utility::random::UniformIntGenerator<int> rand_generator(0, 9);
    for (auto it = values.begin(); it != values.end(); ++it) {
        *it = rand_generator();
    }

    for (int i = 0; i < 10; i++) {
        std::array<int, 1024> new_values;
        utility::random::UniformIntGenerator<int> new_rand_generator(0, 9);
        for (auto it = new_values.begin(); it != new_values.end(); ++it) {
            *it = new_rand_generator();
        }
        EXPECT_FALSE(values == new_values);
    }
}

}  // namespace tests
}  // namespace open3d
