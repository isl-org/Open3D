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
#include <Eigen/Core>
#include "Geometry.h"
#include "KDTreeSearchParam.h"

namespace three {

class PointCloud : public Geometry
{
public:
	PointCloud();
	virtual ~PointCloud();

public:
	Eigen::Vector3d GetMinBound() const override;
	Eigen::Vector3d GetMaxBound() const override;
	void Clear() override;
	bool IsEmpty() const override;
	void Transform(const Eigen::Matrix4d &transformation) override;

public:
	virtual PointCloud &operator+=(const PointCloud &cloud);
	virtual const PointCloud operator+(const PointCloud &cloud);

public:
	bool HasPoints() const {
		return points_.size() > 0;
	}

	bool HasNormals() const {
		return points_.size() > 0 && normals_.size() == points_.size();
	}

	bool HasColors() const {
		return points_.size() > 0 && colors_.size() == points_.size();
	}
	
	bool HasCurvatures() const {
		return points_.size() > 0 && curvatures_.size() == points_.size();
	}
	
	void NormalizeNormals() {
		for (size_t i = 0; i < normals_.size(); i++) {
			normals_[i].normalize();
		}
	}
	
public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
	std::vector<Eigen::Vector3d> colors_;
	std::vector<double> curvatures_;
};

// basic operations

/// Function to downsample input_cloud into output_cloud with a voxel
/// \param voxel_size defines the resolution of the voxel grid, smaller value 
/// leads to denser output point cloud.
/// Normals, colors, and curvatures are averaged if they exist.
bool VoxelDownSample(const PointCloud &input_cloud, double voxel_size,
		PointCloud &output_cloud);

/// Function to compute the normals and curvatures of a point cloud
/// \param cloud is the input point cloud. It also stores the output normals
/// and curvatures.
/// Normals are oriented with respect to the input point cloud is normals exist
/// in the input.
bool EstimateNormalsAndCurvatures(PointCloud &cloud,
		const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());

/// Function to compute the normals and curvatures of a point cloud
/// \param cloud is the input point cloud. It also stores the output normals
/// and curvatures.
/// Normals are oriented with respect to \param orientation_reference.
bool EstimateNormalsAndCurvatures(PointCloud &cloud,
		const Eigen::Vector3d &orientation_reference,
		const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());

}	// namespace three
