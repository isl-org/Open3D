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

#include <Eigen/Core>
#include <tuple>
#include <vector>

namespace Eigen {

typedef Eigen::Matrix<double, 14, 14> Matrix14d;
typedef Eigen::Matrix<double, 14, 1> Vector14d;
typedef Eigen::Matrix<int, 14, 1> Vector14i;

}  // namespace Eigen

namespace open3d {
namespace color_map {

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: this function is almost identical to the functions in
/// Utility/Eigen.h/cpp, but this function takes additional multiplication
/// pattern that can produce JTJ having hundreds of rows and columns.
template <typename VecInTypeDouble,
          typename VecInTypeInt,
          typename MatOutType,
          typename VecOutType>
std::tuple<MatOutType, VecOutType, double> ComputeJTJandJTrNonRigid(
        std::function<void(int, VecInTypeDouble &, double &, VecInTypeInt &)> f,
        int iteration_num,
        int nonrigidval,
        bool verbose = true);

}  // namespace color_map
}  // namespace open3d
