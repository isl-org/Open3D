// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/Macro.h"
#include "open3d/data/Dataset.h"
#include "tests/test_utility/Compare.h"
#include "tests/test_utility/Print.h"
#include "tests/test_utility/Rand.h"
#include "tests/test_utility/Raw.h"
#include "tests/test_utility/Sort.h"

namespace open3d {
namespace tests {

// Eigen Zero()
const Eigen::Vector2d Zero2d = Eigen::Vector2d::Zero();
const Eigen::Vector3d Zero3d = Eigen::Vector3d::Zero();
const Eigen::Matrix<double, 6, 1> Zero6d = Eigen::Matrix<double, 6, 1>::Zero();
const Eigen::Vector2i Zero2i = Eigen::Vector2i::Zero();

// Mechanism for reporting unit tests for which there is no implementation yet.
void NotImplemented();

#define AllCloseOrShow(Arr1, Arr2, rtol, atol)                               \
    EXPECT_TRUE(Arr1.AllClose(Arr2, rtol, atol)) << fmt::format(             \
            "Tensors are not close wrt (relative, absolute) tolerance ({}, " \
            "{}). Max error: {}\n{}\n{}",                                    \
            rtol, atol,                                                      \
            (Arr1 - Arr2)                                                    \
                    .Abs()                                                   \
                    .Flatten()                                               \
                    .Max({0})                                                \
                    .To(core::Float32)                                       \
                    .Item<float>(),                                          \
            Arr1.ToString(), Arr2.ToString());

}  // namespace tests
}  // namespace open3d
