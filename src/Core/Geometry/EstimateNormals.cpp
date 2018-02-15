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

#include "PointCloud.h"

#include <Eigen/Eigenvalues>
#include <Core/Utility/Console.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace three{

namespace {

double sqr(double x) { return x * x; }

Eigen::Vector3d FastEigen3x3(const Eigen::Matrix3d &A)
{
	// Based on:
	// https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
	double p1 = sqr(A(0, 1)) + sqr(A(0, 2)) + sqr(A(1, 2));
	Eigen::Vector3d eigenvalues;
	if (p1 == 0.0) {
		eigenvalues(2) = std::min(A(0, 0), std::min(A(1, 1), A(2, 2)));
		eigenvalues(0) = std::max(A(0, 0), std::max(A(1, 1), A(2, 2)));
		eigenvalues(1) = A.trace() - eigenvalues(0) - eigenvalues(2);
	} else {
		double q = A.trace() / 3.0;
		double p2 = sqr((A(0, 0) - q)) + sqr(A(1, 1) - q) + sqr(A(2, 2) - q) +
				2 * p1;
		double p = sqrt(p2 / 6.0);
		Eigen::Matrix3d B = (1.0 / p) * (A - q * Eigen::Matrix3d::Identity());
		double r = B.determinant() / 2.0;
		double phi;
		if (r <= -1) {
			phi = M_PI / 3.0;
		} else if (r >= 1) {
			phi = 0.0;
		} else {
			phi = std::acos(r) / 3.0;
		}
		eigenvalues(0) = q + 2.0 * p * std::cos(phi);
		eigenvalues(2) = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
		eigenvalues(1) = q * 3.0 - eigenvalues(0) - eigenvalues(2);
	}

	Eigen::Vector3d eigenvector =
			(A - Eigen::Matrix3d::Identity() * eigenvalues(0)) *
			(A.col(0) - Eigen::Vector3d(eigenvalues(1), 0.0, 0.0));
	double len = eigenvector.norm();
	if (len == 0.0) {
		return Eigen::Vector3d::Zero();
	} else {
		return eigenvector.normalized();
	}
}

Eigen::Vector3d ComputeNormal(const PointCloud &cloud,
		const std::vector<int> &indices)
{
	if (indices.size() == 0) {
		return Eigen::Vector3d::Zero();
	}
	Eigen::Matrix3d covariance;
	Eigen::Matrix<double, 9, 1> cumulants;
	cumulants.setZero();
	for (size_t i = 0; i < indices.size(); i++) {
		const Eigen::Vector3d &point = cloud.points_[indices[i]];
		cumulants(0) += point(0);
		cumulants(1) += point(1);
		cumulants(2) += point(2);
		cumulants(3) += point(0) * point(0);
		cumulants(4) += point(0) * point(1);
		cumulants(5) += point(0) * point(2);
		cumulants(6) += point(1) * point(1);
		cumulants(7) += point(1) * point(2);
		cumulants(8) += point(2) * point(2);
	}
	cumulants /= (double)indices.size();
	covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
	covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
	covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
	covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
	covariance(1, 0) = covariance(0, 1);
	covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
	covariance(2, 0) = covariance(0, 2);
	covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
	covariance(2, 1) = covariance(1, 2);

	return FastEigen3x3(covariance);
	//Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
	//solver.compute(covariance, Eigen::ComputeEigenvectors);
	//return solver.eigenvectors().col(0);
}

}	// unnamed namespace

bool EstimateNormals(PointCloud &cloud,
		const KDTreeSearchParam &search_param/* = KDTreeSearchParamKNN()*/)
{
	bool has_normal = cloud.HasNormals();
	if (cloud.HasNormals() == false) {
		cloud.normals_.resize(cloud.points_.size());
	}
	KDTreeFlann kdtree;
	kdtree.SetGeometry(cloud);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)cloud.points_.size(); i++) {
		std::vector<int> indices;
		std::vector<double> distance2;
		Eigen::Vector3d normal;
		if (kdtree.Search(cloud.points_[i], search_param, indices,
				distance2) >= 3) {
			normal = ComputeNormal(cloud, indices);
			if (normal.norm() == 0.0) {
				if (has_normal) {
					normal = cloud.normals_[i];
				} else {
					normal = Eigen::Vector3d(0.0, 0.0, 1.0);
				}
			}
			if (has_normal && normal.dot(cloud.normals_[i]) < 0.0) {
				normal *= -1.0;
			}
			cloud.normals_[i] = normal;
		} else {
			cloud.normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
		}
	}

	return true;
}

bool OrientNormalsToAlignWithDirection(PointCloud &cloud,
		const Eigen::Vector3d &orientation_reference
		/* = Eigen::Vector3d(0.0, 0.0, 1.0)*/)
{
	if (cloud.HasNormals() == false) {
		PrintDebug("[OrientNormalsToAlignWithDirection] No normals in the PointCloud. Call EstimateNormals() first.\n");
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)cloud.points_.size(); i++) {
		auto &normal = cloud.normals_[i];
		if (normal.norm() == 0.0) {
			normal = orientation_reference;
		} else if (normal.dot(orientation_reference) < 0.0) {
			normal *= -1.0;
		}
	}
	return true;
}

bool OrientNormalsTowardsCameraLocation(PointCloud &cloud,
		const Eigen::Vector3d &camera_location/* = Eigen::Vector3d::Zero()*/)
{
	if (cloud.HasNormals() == false) {
		PrintDebug("[OrientNormalsTowardsCameraLocation] No normals in the PointCloud. Call EstimateNormals() first.\n");
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)cloud.points_.size(); i++) {
		Eigen::Vector3d orientation_reference = camera_location -
				cloud.points_[i];
		auto &normal = cloud.normals_[i];
		if (normal.norm() == 0.0) {
			normal = orientation_reference;
			if (normal.norm() == 0.0) {
				normal = Eigen::Vector3d(0.0, 0.0, 1.0);
			} else {
				normal.normalize();
			}
		} else if (normal.dot(orientation_reference) < 0.0) {
			normal *= -1.0;
		}
	}
	return true;
}

}	// namespace three
