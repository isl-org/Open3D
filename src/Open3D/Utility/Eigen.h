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
#include <Eigen/StdVector>
#include <tuple>
#include <vector>

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// Use Eigen::DontAlign for matrices inside classes which are exposed in the
/// Open3D headers https://github.com/intel-isl/Open3D/issues/653
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign> Matrix6d_u;
typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Matrix4d_u;

}  // namespace Eigen

namespace open3d {
namespace utility {

using Matrix4d_allocator = Eigen::aligned_allocator<Eigen::Matrix4d>;
using Matrix6d_allocator = Eigen::aligned_allocator<Eigen::Matrix6d>;
using Vector2d_allocator = Eigen::aligned_allocator<Eigen::Vector2d>;
using Vector4i_allocator = Eigen::aligned_allocator<Eigen::Vector4i>;
using Vector4d_allocator = Eigen::aligned_allocator<Eigen::Vector4d>;
using Vector6d_allocator = Eigen::aligned_allocator<Eigen::Vector6d>;

/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input);

/// Function to transform 4D motion matrix to 6D motion vector
/// this is consistent with the matlab function in
/// the Aerospace Toolbox
/// Reference:
/// https://github.com/qianyizh/ElasticReconstruction/blob/master/Matlab_Toolbox/Core/mrEvaluateRegistration.m
Eigen::Vector6d TransformMatrix4dToVector6d(const Eigen::Matrix4d &input);

/// Function to solve Ax=b
std::tuple<bool, Eigen::VectorXd> SolveLinearSystemPSD(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        bool prefer_sparse = false,
        bool check_symmetric = false,
        bool check_det = false,
        bool check_psd = false);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
std::tuple<bool, Eigen::Matrix4d> SolveJacobianSystemAndObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr);

/// Function to solve Jacobian system
/// Input: 6nx6n Jacobian matrix and 6n-dim residual vector.
/// Output: tuple of is_success, n 4x4 motion matrices.
std::tuple<bool, std::vector<Eigen::Matrix4d, Matrix4d_allocator>>
SolveJacobianSystemAndObtainExtrinsicMatrixArray(const Eigen::MatrixXd &JTJ,
                                                 const Eigen::VectorXd &JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<void(int, VecType &, double &)> f,
        int iteration_num,
        bool verbose = true);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<
                void(int,
                     std::vector<VecType, Eigen::aligned_allocator<VecType>> &,
                     std::vector<double> &)> f,
        int iteration_num,
        bool verbose = true);

Eigen::Matrix3d RotationMatrixX(double radians);
Eigen::Matrix3d RotationMatrixY(double radians);
Eigen::Matrix3d RotationMatrixZ(double radians);

}  // namespace utility
}  // namespace open3d
