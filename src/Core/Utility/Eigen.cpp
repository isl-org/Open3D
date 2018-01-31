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

#include "Eigen.h"

#include <Eigen/Geometry>
#include <Core/Utility/Console.h>

namespace three{

/// Function to solve Ax=b
std::tuple<bool, Eigen::VectorXd> SolveLinearSystem(
		const Eigen::MatrixXd &A, const Eigen::VectorXd &b)
{
	bool solution_exist = true;
	// note: computing determinant for large scale matrix would be bottleneck.
	double det = A.determinant();
	if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det))
		solution_exist = false;

	if (solution_exist) {
		// Robust Cholesky decomposition of a matrix with pivoting.
		Eigen::MatrixXd x = A.ldlt().solve(b);
		return std::make_tuple(solution_exist, std::move(x));
	} else {
		return std::make_tuple(false, std::move(Eigen::VectorXd::Zero(b.rows())));
	}
}

Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input)
{
	Eigen::Matrix4d output;
	output.setIdentity();
	output.block<3, 3>(0, 0) =
		(Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
			Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX())).matrix();
	output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
	return output;
}

Eigen::Vector6d TransformMatrix4dToVector6d(const Eigen::Matrix4d &input)
{
	Eigen::Vector6d output;
	Eigen::Matrix3d R = input.block<3, 3>(0, 0);
	double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
	if (!(sy < 1e-6)) {
		output(0) = atan2(R(2, 1), R(2, 2));
		output(1) = atan2(-R(2, 0), sy);
		output(2) = atan2(R(1, 0), R(0, 0));
	} else {
		output(0) = atan2(-R(1, 2), R(1, 1));
		output(1) = atan2(-R(2, 0), sy);
		output(2) = 0;
	}
	output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
	return output;
}

std::tuple<bool, Eigen::Matrix4d>
		SolveJacobianSystemAndObtainExtrinsicMatrix(
		const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr)
{
	std::vector<Eigen::Matrix4d> output_matrix_array;
	output_matrix_array.clear();

	bool solution_exist;
	Eigen::Vector6d x;
	std::tie(solution_exist, x) = SolveLinearSystem(JTJ, -JTr);

	if (solution_exist) {
		Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(x);
		return std::make_tuple(solution_exist, std::move(extrinsic));
	}
	else {
		return std::make_tuple(false, std::move(Eigen::Matrix4d::Identity()));
	}
}

std::tuple<bool, std::vector<Eigen::Matrix4d>>
		SolveJacobianSystemAndObtainExtrinsicMatrixArray(
		const Eigen::MatrixXd &JTJ, const Eigen::VectorXd &JTr)
{
	std::vector<Eigen::Matrix4d> output_matrix_array;
	output_matrix_array.clear();
	if (JTJ.rows() != JTr.rows() || JTJ.cols() % 6 != 0) {
		PrintWarning("[SolveJacobianSystemAndObtainExtrinsicMatrixArray] Unsupported matrix format.\n");
		return std::make_tuple(false, std::move(output_matrix_array));
	}

	bool solution_exist;
	Eigen::VectorXd x;
	std::tie(solution_exist, x) = SolveLinearSystem(JTJ, JTr);

	if (solution_exist) {
		int nposes = (int)x.rows() / 6;
		for (int i = 0; i < nposes; i++) {
			Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(
					x.block<6, 1>(i * 6, 0));
			output_matrix_array.push_back(extrinsic);
		}
		return std::make_tuple(solution_exist, std::move(output_matrix_array));
	}
	else {
		return std::make_tuple(false, std::move(output_matrix_array));
	}
}

template<typename MatType, typename VecType>
std::tuple<MatType, VecType> ComputeJTJandJTr(
		std::function<void(int, VecType &, double &)> f,
		int iteration_num)
{
	MatType JTJ;
	VecType JTr;
	double r2_sum = 0.0;
	JTJ.setZero();
	JTr.setZero();
#ifdef _OPENMP
#pragma omp parallel
	{
#endif
		MatType JTJ_private;
		VecType JTr_private;
		double r2_sum_private = 0.0;
		JTJ_private.setZero();
		JTr_private.setZero();
		VecType J_r;
		double r;
#ifdef _OPENMP
#pragma omp for nowait
#endif
		for (int i = 0; i < iteration_num; i++) {
			f(i, J_r, r);
			JTJ_private.noalias() += J_r * J_r.transpose();
			JTr_private.noalias() += J_r * r;
			r2_sum_private += r * r;
		}
#ifdef _OPENMP
#pragma omp critical
		{
#endif
			JTJ += JTJ_private;
			JTr += JTr_private;
			r2_sum += r2_sum_private;
#ifdef _OPENMP
		}
	}
#endif
	r2_sum /= (double)iteration_num;
	PrintDebug("Residual : %.2e (# of elements : %d)\n", r2_sum,
			iteration_num);
	return std::make_tuple(std::move(JTJ), std::move(JTr));
}

template<typename MatType, typename VecType>
std::tuple<MatType, VecType> ComputeJTJandJTr(
		std::function<void(int, std::vector<VecType> &, std::vector<double> &)> f,
		int iteration_num)
{
	MatType JTJ;
	VecType JTr;
	double r2_sum = 0.0;
	JTJ.setZero();
	JTr.setZero();
#ifdef _OPENMP
#pragma omp parallel
	{
#endif
		MatType JTJ_private;
		VecType JTr_private;
		double r2_sum_private = 0.0;
		JTJ_private.setZero();
		JTr_private.setZero();
		std::vector<double> r;
		std::vector<VecType> J_r;
#ifdef _OPENMP
#pragma omp for nowait
#endif
		for (int i = 0; i < iteration_num; i++) {
			f(i, J_r, r);
			for (int j = 0; j < (int)r.size(); j++) {
				JTJ_private.noalias() += J_r[j] * J_r[j].transpose();
				JTr_private.noalias() += J_r[j] * r[j];
				r2_sum_private += r[j] * r[j];
			}
		}
#ifdef _OPENMP
#pragma omp critical
		{
#endif
			JTJ += JTJ_private;
			JTr += JTr_private;
			r2_sum += r2_sum_private;
#ifdef _OPENMP
		}
	}
#endif
	r2_sum /= (double)iteration_num;
	PrintDebug("Residual : %.2e (# of elements : %d)\n", r2_sum,
			iteration_num);
	return std::make_tuple(std::move(JTJ), std::move(JTr));
}

template std::tuple<Eigen::Matrix6d, Eigen::Vector6d> ComputeJTJandJTr(
		std::function<void(int, Eigen::Vector6d &, double &)> f,
		int iteration_num);

template std::tuple<Eigen::Matrix6d, Eigen::Vector6d> ComputeJTJandJTr(
		std::function<void(int, std::vector<Eigen::Vector6d> &,
		std::vector<double> &)> f, int iteration_num);

}	// namespace three
