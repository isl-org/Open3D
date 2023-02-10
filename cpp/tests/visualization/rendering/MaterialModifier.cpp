// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/MaterialModifier.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(MaterialModifier, TextureSamplerParameters) {
    auto tsp = visualization::rendering::TextureSamplerParameters::Simple();
    EXPECT_EQ(tsp.GetAnisotropy(), 0);
    tsp.SetAnisotropy(0);
    EXPECT_EQ(tsp.GetAnisotropy(), 0);
    tsp.SetAnisotropy(1);
    EXPECT_EQ(tsp.GetAnisotropy(), 1);
    tsp.SetAnisotropy(2);
    EXPECT_EQ(tsp.GetAnisotropy(), 2);
    tsp.SetAnisotropy(4);
    EXPECT_EQ(tsp.GetAnisotropy(), 4);
    tsp.SetAnisotropy(8);
    EXPECT_EQ(tsp.GetAnisotropy(), 8);
    tsp.SetAnisotropy(10);
    EXPECT_EQ(tsp.GetAnisotropy(), 8);
    tsp.SetAnisotropy(100);
    EXPECT_EQ(tsp.GetAnisotropy(), 64);
    tsp.SetAnisotropy(255);
    EXPECT_EQ(tsp.GetAnisotropy(), 128);
}

}  // namespace tests
}  // namespace open3d
