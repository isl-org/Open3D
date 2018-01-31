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

#include <tuple>
#include <vector>
#include <Eigen/Core>

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

}	// namespace Eigen

namespace three {

/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input);

/// Function to transform 4D motion matrix to 6D motion vector
/// this is consistent with the matlab function in
/// the Aerospace Toolbox
/// Reference: https://github.com/qianyizh/ElasticReconstruction/blob/master/Matlab_Toolbox/Core/mrEvaluateRegistration.m
Eigen::Vector6d TransformMatrix4dToVector6d(const Eigen::Matrix4d &input);

/// Function to solve Ax=b
std::tuple<bool, Eigen::VectorXd> SolveLinearSystem(
	const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
std::tuple<bool, Eigen::Matrix4d>
		SolveJacobianSystemAndObtainExtrinsicMatrix(
		const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr);

/// Function to solve Jacobian system
/// Input: 6nx6n Jacobian matrix and 6n-dim residual vector.
/// Output: tuple of is_success, n 4x4 motion matrices.
std::tuple<bool, std::vector<Eigen::Matrix4d>>
		SolveJacobianSystemAndObtainExtrinsicMatrixArray(
		const Eigen::MatrixXd &JTJ, const Eigen::VectorXd &JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ and JTr
/// Note: f takes index of row, and outputs corresponding residual and row vector.
template<typename MatType, typename VecType>
std::tuple<MatType, VecType> ComputeJTJandJTr(
		std::function<void(int, VecType &, double &)> f,
		int iteration_num);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ and JTr
/// Note: f takes index of row, and outputs corresponding residual and row vector.
template<typename MatType, typename VecType>
std::tuple<MatType, VecType> ComputeJTJandJTr(
		std::function<void(int, std::vector<VecType> &, std::vector<double> &)> f,
		int iteration_num);

}	// namespace three
