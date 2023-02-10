// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SizeVector.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(DynamicSizeVector, Constructor) {
    core::DynamicSizeVector dsv{utility::nullopt, 3};
    EXPECT_FALSE(dsv[0].has_value());
    EXPECT_EQ(dsv[1].value(), 3);
}

TEST(DynamicSizeVector, IsCompatible) {
    EXPECT_TRUE(core::SizeVector({}).IsCompatible({}));
    EXPECT_FALSE(core::SizeVector({}).IsCompatible({utility::nullopt}));
    EXPECT_TRUE(core::SizeVector({10, 3}).IsCompatible({utility::nullopt, 3}));
    EXPECT_FALSE(core::SizeVector({10, 3}).IsCompatible({utility::nullopt, 5}));
}

}  // namespace tests
}  // namespace open3d
