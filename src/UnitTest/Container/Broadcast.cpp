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

#include "Open3D/Container/Broadcast.h"

#include "TestUtility/UnitTest.h"

using namespace std;
using namespace open3d;

TEST(Broadcast, IsCompatibleBroadcastShape) {
    // A 0-dim tensor is compatible with any shape.
    EXPECT_TRUE(IsCompatibleBroadcastShape({}, {}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({}, {1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({1}, {}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({}, {2}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2}, {}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({}, {1, 1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({1, 1}, {}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({}, {1, 2}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({1, 2}, {}));

    // Dim with size 0 is compatible with dim with size 0 or 1.
    EXPECT_TRUE(IsCompatibleBroadcastShape({0}, {0}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({0}, {1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({1}, {0}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 0}, {2, 1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 1}, {2, 0}));
    EXPECT_FALSE(IsCompatibleBroadcastShape({2, 0}, {2, 3}));
    EXPECT_FALSE(IsCompatibleBroadcastShape({2, 3}, {2, 0}));

    // Regular cases.
    EXPECT_TRUE(IsCompatibleBroadcastShape({1}, {1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({1}, {2, 1}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 1}, {1}));

    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 1, 3}, {2, 5, 3}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 5, 3}, {2, 1, 3}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({2, 1, 3}, {5, 3}));
    EXPECT_TRUE(IsCompatibleBroadcastShape({5, 3}, {2, 1, 3}));

    EXPECT_FALSE(IsCompatibleBroadcastShape({2, 4, 3}, {2, 5, 3}));
    EXPECT_FALSE(IsCompatibleBroadcastShape({2, 5, 3}, {2, 4, 3}));
    EXPECT_FALSE(IsCompatibleBroadcastShape({2, 4, 3}, {5, 3}));
    EXPECT_FALSE(IsCompatibleBroadcastShape({5, 3}, {2, 4, 3}));
}

TEST(Broadcast, BroadcastedShape) {
    // A 0-dim tensor can be brocasted to any shape.
    EXPECT_EQ(BroadcastedShape({}, {}), SizeVector({}));
    EXPECT_EQ(BroadcastedShape({}, {1}), SizeVector({1}));
    EXPECT_EQ(BroadcastedShape({1}, {}), SizeVector({1}));
    EXPECT_EQ(BroadcastedShape({}, {2}), SizeVector({2}));
    EXPECT_EQ(BroadcastedShape({2}, {}), SizeVector({2}));
    EXPECT_EQ(BroadcastedShape({}, {1, 1}), SizeVector({1, 1}));
    EXPECT_EQ(BroadcastedShape({1, 1}, {}), SizeVector({1, 1}));
    EXPECT_EQ(BroadcastedShape({}, {1, 2}), SizeVector({1, 2}));
    EXPECT_EQ(BroadcastedShape({1, 2}, {}), SizeVector({1, 2}));

    // Dim with size 0 is compatible with dim with size 0 or 1. The brocasted
    // size is 0.
    EXPECT_EQ(BroadcastedShape({0}, {0}), SizeVector({0}));
    EXPECT_EQ(BroadcastedShape({0}, {1}), SizeVector({0}));
    EXPECT_EQ(BroadcastedShape({1}, {0}), SizeVector({0}));
    EXPECT_EQ(BroadcastedShape({2, 0}, {2, 1}), SizeVector({2, 0}));
    EXPECT_EQ(BroadcastedShape({2, 1}, {2, 0}), SizeVector({2, 0}));
    EXPECT_THROW(BroadcastedShape({2, 0}, {2, 3}), std::runtime_error);
    EXPECT_THROW(BroadcastedShape({2, 3}, {2, 0}), std::runtime_error);

    // Regular cases.
    EXPECT_EQ(BroadcastedShape({1}, {1}), SizeVector({1}));
    EXPECT_EQ(BroadcastedShape({1}, {2, 1}), SizeVector({2, 1}));
    EXPECT_EQ(BroadcastedShape({2, 1}, {1}), SizeVector({2, 1}));

    EXPECT_EQ(BroadcastedShape({2, 1, 3}, {2, 5, 3}), SizeVector({2, 5, 3}));
    EXPECT_EQ(BroadcastedShape({2, 5, 3}, {2, 1, 3}), SizeVector({2, 5, 3}));
    EXPECT_EQ(BroadcastedShape({2, 1, 3}, {5, 3}), SizeVector({2, 5, 3}));
    EXPECT_EQ(BroadcastedShape({5, 3}, {2, 1, 3}), SizeVector({2, 5, 3}));

    EXPECT_THROW(BroadcastedShape({2, 4, 3}, {2, 5, 3}), std::runtime_error);
    EXPECT_THROW(BroadcastedShape({2, 5, 3}, {2, 4, 3}), std::runtime_error);
    EXPECT_THROW(BroadcastedShape({2, 4, 3}, {5, 3}), std::runtime_error);
    EXPECT_THROW(BroadcastedShape({5, 3}, {2, 4, 3}), std::runtime_error);
}

TEST(Broadcast, CanBeBrocastedToShape) {
    // A 0-dim tensor can be brocasted to any shape. Not commutative.
    EXPECT_TRUE(CanBeBrocastedToShape({}, {}));
    EXPECT_TRUE(CanBeBrocastedToShape({}, {1}));
    EXPECT_FALSE(CanBeBrocastedToShape({1}, {}));
    EXPECT_TRUE(CanBeBrocastedToShape({}, {2}));
    EXPECT_FALSE(CanBeBrocastedToShape({2}, {}));
    EXPECT_TRUE(CanBeBrocastedToShape({}, {1, 1}));
    EXPECT_FALSE(CanBeBrocastedToShape({1, 1}, {}));
    EXPECT_TRUE(CanBeBrocastedToShape({}, {1, 2}));
    EXPECT_FALSE(CanBeBrocastedToShape({1, 2}, {}));

    // Dim with size 0 can only be brocasteded to 0.
    // Only dim with size 0 or 1 can be brocasted to 0.
    EXPECT_TRUE(CanBeBrocastedToShape({0}, {0}));
    EXPECT_FALSE(CanBeBrocastedToShape({0}, {1}));
    EXPECT_TRUE(CanBeBrocastedToShape({1}, {0}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 0}, {2, 1}));
    EXPECT_TRUE(CanBeBrocastedToShape({2, 1}, {2, 0}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 0}, {2, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 3}, {2, 0}));

    // Regular cases. Not commutative.
    EXPECT_TRUE(CanBeBrocastedToShape({1}, {1}));
    EXPECT_TRUE(CanBeBrocastedToShape({1}, {2, 1}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 1}, {1}));

    EXPECT_TRUE(CanBeBrocastedToShape({2, 1, 3}, {2, 5, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 5, 3}, {2, 1, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 1, 3}, {5, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({5, 3}, {2, 1, 3}));

    EXPECT_FALSE(CanBeBrocastedToShape({2, 4, 3}, {2, 5, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 5, 3}, {2, 4, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({2, 4, 3}, {5, 3}));
    EXPECT_FALSE(CanBeBrocastedToShape({5, 3}, {2, 4, 3}));
}
