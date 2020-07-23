// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/visualization/rendering/MaterialModifier.h"

#include "tests/UnitTest.h"

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
