// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <tuple>
#include <vector>

/// @cond
namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3uint8;

/// Use Eigen::DontAlign for matrices inside classes which are exposed in the
/// Open3D headers https://github.com/isl-org/Open3D/issues/653
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign> Matrix6d_u;
typedef Eigen::Matrix<double, 4, 4, Eigen::DontAlign> Matrix4d_u;

}  // namespace Eigen
/// @endcond

namespace open3d {
namespace utility {

using Matrix4d_allocator = Eigen::aligned_allocator<Eigen::Matrix4d>;
using Matrix6d_allocator = Eigen::aligned_allocator<Eigen::Matrix6d>;
using Vector2d_allocator = Eigen::aligned_allocator<Eigen::Vector2d>;
using Vector3uint8_allocator = Eigen::aligned_allocator<Eigen::Vector3uint8>;
using Vector4i_allocator = Eigen::aligned_allocator<Eigen::Vector4i>;
using Vector4d_allocator = Eigen::aligned_allocator<Eigen::Vector4d>;
using Vector6d_allocator = Eigen::aligned_allocator<Eigen::Vector6d>;

/// Genretate a skew-symmetric matrix from a vector 3x1.
Eigen::Matrix3d SkewMatrix(const Eigen::Vector3d &vec);

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
        std::function<void(int, VecType &, double &, double &)> f,
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
                     std::vector<double> &,
                     std::vector<double> &)> f,
        int iteration_num,
        bool verbose = true);

Eigen::Matrix3d RotationMatrixX(double radians);
Eigen::Matrix3d RotationMatrixY(double radians);
Eigen::Matrix3d RotationMatrixZ(double radians);

/// Color conversion from double [0,1] to uint8_t 0-255; this does proper
/// clipping and rounding
Eigen::Vector3uint8 ColorToUint8(const Eigen::Vector3d &color);
/// Color conversion from uint8_t 0-255 to double [0,1]
Eigen::Vector3d ColorToDouble(uint8_t r, uint8_t g, uint8_t b);
Eigen::Vector3d ColorToDouble(const Eigen::Vector3uint8 &rgb);

/// Function to compute the covariance matrix of a set of points.
template <typename IdxType>
Eigen::Matrix3d ComputeCovariance(const std::vector<Eigen::Vector3d> &points,
                                  const std::vector<IdxType> &indices);

/// Function to compute the mean and covariance matrix of a set of points.
template <typename IdxType>
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<IdxType> &indices);

/// Function to compute the mean and covariance matrix of a set of points.
/// \tparam RealType Either float or double.
/// \tparam IdxType Either size_t or int.
/// \param points Contiguous memory with the 3D points.
/// \param indices The indices for which the mean and covariance will be
/// computed. \return The mean and covariance matrix.
template <typename RealType, typename IdxType>
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance(
        const RealType *const points, const std::vector<IdxType> &indices);

}  // namespace utility
}  // namespace open3d
