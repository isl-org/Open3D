// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/ShapeUtil.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(ShapeUtil, IsCompatibleBroadcastShape) {
    // A 0-dim tensor is compatible with any shape.
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({}, {}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({}, {1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1}, {}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({}, {2}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({2}, {}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({}, {1, 1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1, 1}, {}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({}, {1, 2}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1, 2}, {}));

    // Dim with size 0 is compatible with dim with size 0 or 1.
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({0}, {0}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({0}, {1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1}, {0}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({2, 0}, {2, 1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({2, 1}, {2, 0}));
    EXPECT_FALSE(core::shape_util::IsCompatibleBroadcastShape({2, 0}, {2, 3}));
    EXPECT_FALSE(core::shape_util::IsCompatibleBroadcastShape({2, 3}, {2, 0}));

    // Regular cases.
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1}, {1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({1}, {2, 1}));
    EXPECT_TRUE(core::shape_util::IsCompatibleBroadcastShape({2, 1}, {1}));

    EXPECT_TRUE(
            core::shape_util::IsCompatibleBroadcastShape({2, 1, 3}, {2, 5, 3}));
    EXPECT_TRUE(
            core::shape_util::IsCompatibleBroadcastShape({2, 5, 3}, {2, 1, 3}));
    EXPECT_TRUE(
            core::shape_util::IsCompatibleBroadcastShape({2, 1, 3}, {5, 3}));
    EXPECT_TRUE(
            core::shape_util::IsCompatibleBroadcastShape({5, 3}, {2, 1, 3}));

    EXPECT_FALSE(
            core::shape_util::IsCompatibleBroadcastShape({2, 4, 3}, {2, 5, 3}));
    EXPECT_FALSE(
            core::shape_util::IsCompatibleBroadcastShape({2, 5, 3}, {2, 4, 3}));
    EXPECT_FALSE(
            core::shape_util::IsCompatibleBroadcastShape({2, 4, 3}, {5, 3}));
    EXPECT_FALSE(
            core::shape_util::IsCompatibleBroadcastShape({5, 3}, {2, 4, 3}));
}

TEST(ShapeUtil, BroadcastedShape) {
    // A 0-dim tensor can be brocasted to any shape.
    EXPECT_EQ(core::shape_util::BroadcastedShape({}, {}), core::SizeVector({}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({}, {1}),
              core::SizeVector({1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({1}, {}),
              core::SizeVector({1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({}, {2}),
              core::SizeVector({2}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2}, {}),
              core::SizeVector({2}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({}, {1, 1}),
              core::SizeVector({1, 1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({1, 1}, {}),
              core::SizeVector({1, 1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({}, {1, 2}),
              core::SizeVector({1, 2}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({1, 2}, {}),
              core::SizeVector({1, 2}));

    // Dim with size 0 is compatible with dim with size 0 or 1. The brocasted
    // size is 0.
    EXPECT_EQ(core::shape_util::BroadcastedShape({0}, {0}),
              core::SizeVector({0}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({0}, {1}),
              core::SizeVector({0}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({1}, {0}),
              core::SizeVector({0}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 0}, {2, 1}),
              core::SizeVector({2, 0}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 1}, {2, 0}),
              core::SizeVector({2, 0}));
    EXPECT_THROW(core::shape_util::BroadcastedShape({2, 0}, {2, 3}),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::BroadcastedShape({2, 3}, {2, 0}),
                 std::runtime_error);

    // Regular cases.
    EXPECT_EQ(core::shape_util::BroadcastedShape({1}, {1}),
              core::SizeVector({1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({1}, {2, 1}),
              core::SizeVector({2, 1}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 1}, {1}),
              core::SizeVector({2, 1}));

    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 1, 3}, {2, 5, 3}),
              core::SizeVector({2, 5, 3}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 5, 3}, {2, 1, 3}),
              core::SizeVector({2, 5, 3}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({2, 1, 3}, {5, 3}),
              core::SizeVector({2, 5, 3}));
    EXPECT_EQ(core::shape_util::BroadcastedShape({5, 3}, {2, 1, 3}),
              core::SizeVector({2, 5, 3}));

    EXPECT_THROW(core::shape_util::BroadcastedShape({2, 4, 3}, {2, 5, 3}),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::BroadcastedShape({2, 5, 3}, {2, 4, 3}),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::BroadcastedShape({2, 4, 3}, {5, 3}),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::BroadcastedShape({5, 3}, {2, 4, 3}),
                 std::runtime_error);
}

TEST(ShapeUtil, CanBeBrocastedToShape) {
    // A 0-dim tensor can be brocasted to any shape. Not commutative.
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({}, {}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({}, {1}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({1}, {}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({}, {2}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2}, {}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({}, {1, 1}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({1, 1}, {}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({}, {1, 2}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({1, 2}, {}));

    // Dim with size 0 can only be brocasteded to 0.
    // Only dim with size 0 or 1 can be brocasted to 0.
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({0}, {0}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({0}, {1}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({1}, {0}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 0}, {2, 1}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({2, 1}, {2, 0}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 0}, {2, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 3}, {2, 0}));

    // Regular cases. Not commutative.
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({1}, {1}));
    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({1}, {2, 1}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 1}, {1}));

    EXPECT_TRUE(core::shape_util::CanBeBrocastedToShape({2, 1, 3}, {2, 5, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 5, 3}, {2, 1, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 1, 3}, {5, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({5, 3}, {2, 1, 3}));

    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 4, 3}, {2, 5, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 5, 3}, {2, 4, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({2, 4, 3}, {5, 3}));
    EXPECT_FALSE(core::shape_util::CanBeBrocastedToShape({5, 3}, {2, 4, 3}));
}

TEST(ShapeUtil, ReductionShape) {
    // Empty cases
    EXPECT_EQ(core::shape_util::ReductionShape({}, {}, false),
              core::SizeVector({}));
    EXPECT_EQ(core::shape_util::ReductionShape({}, {}, true),
              core::SizeVector({}));

    // Out-of-range exception.
    EXPECT_THROW(core::shape_util::ReductionShape({}, {1}, false),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::ReductionShape({1}, {2}, false),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::ReductionShape({}, {1}, true),
                 std::runtime_error);
    EXPECT_THROW(core::shape_util::ReductionShape({1}, {2}, true),
                 std::runtime_error);

    // Dimension with size 0 can be reduced to size 1.
    EXPECT_EQ(core::shape_util::ReductionShape({2, 0}, {1}, false),
              core::SizeVector({2}));
    EXPECT_EQ(core::shape_util::ReductionShape({2, 0}, {1}, true),
              core::SizeVector({2, 1}));

    // Regular cases.
    EXPECT_EQ(core::shape_util::ReductionShape({2, 3, 4}, {0, 2}, false),
              core::SizeVector({3}));
    EXPECT_EQ(core::shape_util::ReductionShape({2, 3, 4}, {0, 2}, true),
              core::SizeVector({1, 3, 1}));

    // Wrap-around is fine.
    EXPECT_EQ(core::shape_util::ReductionShape({2, 3, 4}, {0, -1}, false),
              core::SizeVector({3}));
    EXPECT_EQ(core::shape_util::ReductionShape({2, 3, 4}, {0, -1}, true),
              core::SizeVector({1, 3, 1}));
}

}  // namespace tests
}  // namespace open3d
