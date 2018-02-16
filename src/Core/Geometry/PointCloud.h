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
#include <memory>
#include <Eigen/Core>
#include <Core/Geometry/Geometry3D.h>
#include <Core/Geometry/KDTreeSearchParam.h>

namespace three {

class Image;
class RGBDImage;
class PinholeCameraIntrinsic;

class PointCloud : public Geometry3D
{
public:
	PointCloud() : Geometry3D(Geometry::GeometryType::PointCloud) {};
	~PointCloud() override {};

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector3d GetMinBound() const override;
	Eigen::Vector3d GetMaxBound() const override;
	void Transform(const Eigen::Matrix4d &transformation) override;

public:
	PointCloud &operator+=(const PointCloud &cloud);
	PointCloud operator+(const PointCloud &cloud) const;

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

	void NormalizeNormals() {
		for (size_t i = 0; i < normals_.size(); i++) {
			normals_[i].normalize();
		}
	}

	void PaintUniformColor(const Eigen::Vector3d &color) {
		colors_.resize(points_.size());
		for (size_t i = 0; i < points_.size(); i++) {
			colors_[i] = color;
		}
	}

public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
	std::vector<Eigen::Vector3d> colors_;
};

/// Factory function to create a pointcloud from a file (PointCloudFactory.cpp)
/// Return an empty pointcloud if fail to read the file.
std::shared_ptr<PointCloud> CreatePointCloudFromFile(
		const std::string &filename);

/// Factory function to create a pointcloud from a depth image and a camera
/// model (PointCloudFactory.cpp)
/// The input depth image can be either a float image, or a uint16_t image. In
/// the latter case, the depth is scaled by 1 / depth_scale, and truncated at
/// depth_trunc distance. The depth image is also sampled with stride, in order
/// to support (fast) coarse point cloud extraction.
/// Return an empty pointcloud if the conversion fails.
std::shared_ptr<PointCloud> CreatePointCloudFromDepthImage(
		const Image &depth, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
		double depth_scale = 1000.0, double depth_trunc = 1000.0,
		int stride = 1);

/// Factory function to create a pointcloud from an RGB-D image and a camera
/// model (PointCloudFactory.cpp)
/// Return an empty pointcloud if the conversion fails.
std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImage(
		const RGBDImage &image, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity());

/// Function to select points from input pointcloud into output pointcloud
/// Points with indices in \param indices are selected.
std::shared_ptr<PointCloud> SelectDownSample(const PointCloud &input,
		const std::vector<size_t> &indices);

/// Function to downsample input pointcloud into output pointcloud with a voxel
/// \param voxel_size defines the resolution of the voxel grid, smaller value
/// leads to denser output point cloud.
/// Normals and colors are averaged if they exist.
std::shared_ptr<PointCloud> VoxelDownSample(const PointCloud &input,
		double voxel_size);

/// Function to downsample input pointcloud into output pointcloud uniformly
/// \param every_k_points indicates the sample rate.
std::shared_ptr<PointCloud> UniformDownSample(const PointCloud &input,
		size_t every_k_points);

/// Function to crop input pointcloud into output pointcloud
/// All points with coordinates less than \param min_bound or larger than
/// \param max_bound are clipped.
std::shared_ptr<PointCloud> CropPointCloud(const PointCloud &input,
		const Eigen::Vector3d &min_bound, const Eigen::Vector3d &max_bound);

/// Function to compute the normals of a point cloud
/// \param cloud is the input point cloud. It also stores the output normals.
/// Normals are oriented with respect to the input point cloud if normals exist
/// in the input.
bool EstimateNormals(PointCloud &cloud,
		const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());

/// Function to orient the normals of a point cloud
/// \param cloud is the input point cloud. It must have normals.
/// Normals are oriented with respect to \param orientation_reference.
bool OrientNormalsToAlignWithDirection(PointCloud &cloud,
		const Eigen::Vector3d &orientation_reference =
		Eigen::Vector3d(0.0, 0.0, 1.0));

/// Function to orient the normals of a point cloud
/// \param cloud is the input point cloud. It also stores the output normals.
/// Normals are oriented with towards \param camera_location.
bool OrientNormalsTowardsCameraLocation(PointCloud &cloud,
		const Eigen::Vector3d &camera_location = Eigen::Vector3d::Zero());

/// Function to compute the ponit to point distances between point clouds
/// \param distances is the output distance. It has the same size as the number
/// of points in \param source.
std::vector<double> ComputePointCloudToPointCloudDistance(
		const PointCloud &source, const PointCloud &target);

/// Function to compute the mean and covariance matrix of a point cloud
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputePointCloudMeanAndCovariance(
		const PointCloud &input);

/// Function to compute the Mahalanobis distance for points in a point cloud
/// https://en.wikipedia.org/wiki/Mahalanobis_distance
std::vector<double> ComputePointCloudMahalanobisDistance(
		const PointCloud &input);

/// Function to compute the distance from a point to its nearest neighbor in the
/// point cloud
std::vector<double> ComputePointCloudNearestNeighborDistance(
		const PointCloud &input);

}	// namespace three
