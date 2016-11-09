// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <vector>
#include <memory>
#include <Eigen/Core>

namespace Eigen {
typedef Matrix<double, 33, 1> Vector33d;
}	// namespace Eigen

namespace three {

class PointCloud;

class FPFHFeature
{
public:
	bool ComputeFPFHFeature(const PointCloud &cloud,
			const std::vector<int> &key_fpfh_points = {});

private:
	void CreateKeySPFHPoints(const std::vector<int> key_fpfh_points,
			std::vector<int> &key_spfh_points);

	void ComputeSPFHFeature(const PointCloud &cloud,
			const std::vector<int> &key_spfh_points,
			std::vector<Eigen::Vector33d> & spfh);

public:
	std::vector<Eigen::Vector33d> features_;
};

/// Factory function to create dense FPFH feature for a point cloud
/// This function is a wrapper of FPFHFeature::ComputeFPFHFeature()
std::shared_ptr<FPFHFeature> CreateFPFHFeatureFromPointCloud(
		const PointCloud &cloud);

}	// namespace three
