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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Core.h>
using namespace three;

void pybind_pointcloud(py::module &m)
{
	py::class_<PointCloud, PyGeometry3D<PointCloud>,
			std::shared_ptr<PointCloud>, Geometry3D> pointcloud(m,
			"PointCloud");
	pointcloud.def(py::init<>())
		.def("__repr__", [](const PointCloud &pcd) {
			return std::string("PointCloud with ") + 
					std::to_string(pcd.points_.size()) + " points.";
		})
		.def("HasPoints", &PointCloud::HasPoints)
		.def("HasNormals", &PointCloud::HasNormals)
		.def("HasColors", &PointCloud::HasColors)
		.def("NormalizeNormals", &PointCloud::NormalizeNormals)
		.def_readwrite("points", &PointCloud::points_)
		.def_readwrite("normals", &PointCloud::normals_)
		.def_readwrite("colors", &PointCloud::colors_);
}

void pybind_pointcloud_methods(py::module &m)
{
	m.def("CreatePointCloudFromFile", &CreatePointCloudFromFile,
			"Factory function to create a pointcloud from a file",
			"filename"_a);
	m.def("SelectDownSample", [](const PointCloud &input,
			const std::vector<size_t> &indices) {
			PointCloud output;
			if (SelectDownSample(input, indices, output) == false)
				throw std::runtime_error("SelectDownSample() error!");
			return output;
		}, "Function to select points from input pointcloud into output pointcloud",
			"input"_a, "indices"_a);
	m.def("VoxelDownSample", [](const PointCloud &input, double voxel_size) {
			PointCloud output;
			if (VoxelDownSample(input, voxel_size, output) == false)
				throw std::runtime_error("VoxelDownSample() error!");
			return output;
		}, "Function to downsample input pointcloud into output pointcloud with a voxel",
			"input"_a, "voxel_size"_a);
	m.def("UniformDownSample", [](const PointCloud &input,
			size_t every_k_points) {
			PointCloud output;
			if (UniformDownSample(input, every_k_points, output) == false)
				throw std::runtime_error("UniformDownSample() error!");
			return output;
		}, "Function to downsample input pointcloud into output pointcloud uniformly",
			"input"_a, "every_k_points"_a);
	m.def("CropPointCloud", [](const PointCloud &input,
			const Eigen::Vector3d &min_bound,
			const Eigen::Vector3d &max_bound) {
			PointCloud output;
			if (CropPointCloud(input, min_bound, max_bound, output) == false)
				throw std::runtime_error("CropPointCloud() error!");
			return output;
		}, "Function to crop input pointcloud into output pointcloud",
			"input"_a, "min_bound"_a, "max_bound"_a);
	m.def("EstimateNormals", &EstimateNormals,
			"Function to compute the normals of a point cloud",
			"cloud"_a, "search_param"_a = KDTreeSearchParamKNN());
	m.def("OrientNormalsToAlignWithDirection",
			&OrientNormalsToAlignWithDirection,
			"Function to orient the normals of a point cloud",
			"cloud"_a, "orientation_reference"_a = 
			Eigen::Vector3d(0.0, 0.0, 1.0));
	m.def("OrientNormalsTowardsCameraLocation",
			&OrientNormalsTowardsCameraLocation,
			"Function to orient the normals of a point cloud",
			"cloud"_a, "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
	m.def("ComputePointCloudToPointCloudDistance", [](const PointCloud &source,
			const PointCloud &target) {
			std::vector<double> distances;
			ComputePointCloudToPointCloudDistance(source, target, distances);
			return distances;
		}, "Function to compute the ponit to point distances between point clouds",
			"source"_a, "target"_a);
	m.def("ComputePointCloudMeanAndCovariance", [](const PointCloud &input) {
			Eigen::Vector3d mean; Eigen::Matrix3d covariance;
			ComputePointCloudMeanAndCovariance(input, mean, covariance);
			return std::make_tuple(mean, covariance);
		}, "Function to compute the mean and covariance matrix of a point cloud",
			"input"_a);
	m.def("ComputePointCloudMahalanobisDistance", [](const PointCloud &input) {
			std::vector<double> distance;
			ComputePointCloudMahalanobisDistance(input, distance);
			return distance;
		}, "Function to compute the Mahalanobis distance for points in a point cloud",
			"input"_a);
	m.def("ComputePointCloudNearestNeighborDistance", [](
			const PointCloud &input) {
			std::vector<double> distance;
			ComputePointCloudNearestNeighborDistance(input, distance);
			return distance;
		}, "Function to compute the distance from a point to its nearest neighbor in the point cloud",
			"input"_a);
}
