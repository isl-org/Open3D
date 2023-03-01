// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/TensorMap.h"

#include <vector>

#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorMapPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorMap,
                         TensorMapPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorMapPermuteDevices, Constructor) {
    core::Dtype dtype = core::Float32;
    core::Device device = GetParam();

    // Empty TensorMap.
    t::geometry::TensorMap tm0("positions");
    EXPECT_EQ(tm0.GetPrimaryKey(), "positions");
    EXPECT_EQ(tm0.size(), 0);

    // Primary key is required.
    EXPECT_ANY_THROW(t::geometry::TensorMap());

    // Iterators.
    std::map<std::string, core::Tensor> tensor_map(
            {{"positions", core::Tensor::Zeros({10, 3}, dtype, device)},
             {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});
    t::geometry::TensorMap tm1("positions", tensor_map.begin(),
                               tensor_map.end());
    EXPECT_TRUE(tm1["positions"].IsSame(tensor_map["positions"]));
    EXPECT_TRUE(tm1["colors"].IsSame(tensor_map["colors"]));

    // Initializer list.
    t::geometry::TensorMap tm2(
            "positions",
            {{"positions", core::Tensor::Zeros({10, 3}, dtype, device)},
             {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});

    // Move constructor, Tensors are shallow copied.
    t::geometry::TensorMap tm2_copied(tm2);
    EXPECT_TRUE(tm2_copied["positions"].IsSame(tm2["positions"]));
    EXPECT_TRUE(tm2_copied["colors"].IsSame(tm2["colors"]));

    // Move constructor, Tensors are shallow copied.
    t::geometry::TensorMap tm2_moved = std::move(tm2);
    EXPECT_TRUE(tm2_moved["positions"].IsSame(tm2["positions"]));
    EXPECT_TRUE(tm2_moved["colors"].IsSame(tm2["colors"]));
}

TEST_P(TensorMapPermuteDevices, IsSizeSynchronized) {
    core::Dtype dtype = core::Float32;
    core::Device device = GetParam();

    t::geometry::TensorMap tm(
            "positions",
            {{"positions", core::Tensor::Zeros({5, 3}, dtype, device)},
             {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});
    EXPECT_FALSE(tm.IsSizeSynchronized());

    tm["colors"] = core::Tensor::Ones({5, 3}, dtype, device);
    EXPECT_TRUE(tm.IsSizeSynchronized());
}

TEST_P(TensorMapPermuteDevices, AssertSizeSynchronized) {
    core::Dtype dtype = core::Float32;
    core::Device device = GetParam();

    t::geometry::TensorMap tm(
            "positions",
            {{"positions", core::Tensor::Zeros({5, 3}, dtype, device)},
             {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});
    EXPECT_ANY_THROW(tm.AssertSizeSynchronized());

    tm["colors"] = core::Tensor::Ones({5, 3}, dtype, device);
    tm.AssertSizeSynchronized();
}

TEST_P(TensorMapPermuteDevices, Contains) {
    core::Dtype dtype = core::Float32;
    core::Device device = GetParam();

    t::geometry::TensorMap tm(
            "positions",
            {{"positions", core::Tensor::Zeros({5, 3}, dtype, device)},
             {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});
    EXPECT_TRUE(tm.Contains("positions"));
    EXPECT_TRUE(tm.Contains("colors"));
    EXPECT_FALSE(tm.Contains("normals"));
}

}  // namespace tests
}  // namespace open3d
