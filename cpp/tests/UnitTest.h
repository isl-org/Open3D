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

#pragma once

// TEST_DATA_DIR defined in CMakeLists.txt
// Put it here to avoid editor warnings
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR
#endif

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/Macro.h"
#include "tests/test_utility/Compare.h"
#include "tests/test_utility/Print.h"
#include "tests/test_utility/Rand.h"
#include "tests/test_utility/Raw.h"
#include "tests/test_utility/Sort.h"

// GPU_CONDITIONAL_COMPILE_STR is "" if gpu is available, otherwise "DISABLED_"
// The GPU_CONDITIONAL_COMPILE_STR value is configured in CMake
#define CUDA_CONDITIONAL_TEST(test_name) \
    OPEN3D_CONCATENATE(GPU_CONDITIONAL_TEST_STR, test_name)

namespace open3d {
namespace tests {

// Eigen Zero()
const Eigen::Vector2d Zero2d = Eigen::Vector2d::Zero();
const Eigen::Vector3d Zero3d = Eigen::Vector3d::Zero();
const Eigen::Matrix<double, 6, 1> Zero6d = Eigen::Matrix<double, 6, 1>::Zero();
const Eigen::Vector2i Zero2i = Eigen::Vector2i::Zero();

// Mechanism for reporting unit tests for which there is no implementation yet.
void NotImplemented();

}  // namespace tests
}  // namespace open3d
