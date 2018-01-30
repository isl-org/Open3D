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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>
#include <IO/ClassIO/PointCloudIO.h>
using namespace three;

void pybind_pointcloud(py::module &m)
{
	py::class_<PointCloud, PyGeometry3D<PointCloud>,
			std::shared_ptr<PointCloud>, Geometry3D> pointcloud(m,
			"PointCloud");
	py::detail::bind_default_constructor<PointCloud>(pointcloud);
	py::detail::bind_copy_functions<PointCloud>(pointcloud);
	pointcloud
		.def("__repr__", [](const PointCloud &pcd) {
			return std::string("PointCloud with ") +
					std::to_string(pcd.points_.size()) + " points.";
		})
		.def(py::self + py::self)
		.def(py::self += py::self)
		.def("has_points", &PointCloud::HasPoints)
		.def("has_normals", &PointCloud::HasNormals)
		.def("has_colors", &PointCloud::HasColors)
		.def("normalize_normals", &PointCloud::NormalizeNormals)
		.def("paint_uniform_color", &PointCloud::PaintUniformColor)
		.def_readwrite("points", &PointCloud::points_)
		.def_readwrite("normals", &PointCloud::normals_)
		.def_readwrite("colors", &PointCloud::colors_);
}

void pybind_pointcloud_methods(py::module &m)
{
	m.def("read_point_cloud", [](const std::string &filename) {
		PointCloud pcd;
		ReadPointCloud(filename, pcd);
		return pcd;
	}, "Function to read PointCloud from file", "filename"_a);
	m.def("write_point_cloud", [](const std::string &filename,
			const PointCloud &pointcloud, bool write_ascii, bool compressed) {
		return WritePointCloud(filename, pointcloud, write_ascii, compressed);
	}, "Function to write PointCloud to file", "filename"_a, "pointcloud"_a,
			"write_ascii"_a = false, "compressed"_a = false);
	m.def("create_point_cloud_from_depth_image", &CreatePointCloudFromDepthImage,
			"Factory function to create a pointcloud from a depth image and a camera.\n"
			"Given depth value d at (u, v) image coordinate, the corresponding 3d point is:\n"
			"    z = d / depth_scale\n"
			"    x = (u - cx) * z / fx\n"
			"    y = (v - cy) * z / fy",
			"depth"_a, "intrinsic"_a,
			"extrinsic"_a = Eigen::Matrix4d::Identity(),
			"depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0, "stride"_a = 1);
	m.def("create_point_cloud_from_rgbd_image", &CreatePointCloudFromRGBDImage,
			"Factory function to create a pointcloud from an RGB-D image and a camera.\n"
			"Given depth value d at (u, v) image coordinate, the corresponding 3d point is:\n"
			"    z = d / depth_scale\n"
			"    x = (u - cx) * z / fx\n"
			"    y = (v - cy) * z / fy",
			"image"_a, "intrinsic"_a,
			"extrinsic"_a = Eigen::Matrix4d::Identity());
	m.def("select_down_sample", &SelectDownSample,
			"Function to select points from input pointcloud into output pointcloud",
			"input"_a, "indices"_a);
	m.def("voxel_down_sample", &VoxelDownSample,
			"Function to downsample input pointcloud into output pointcloud with a voxel",
			"input"_a, "voxel_size"_a);
	m.def("uniform_down_sample", &UniformDownSample,
			"Function to downsample input pointcloud into output pointcloud uniformly",
			"input"_a, "every_k_points"_a);
	m.def("crop_point_cloud", &CropPointCloud,
			"Function to crop input pointcloud into output pointcloud",
			"input"_a, "min_bound"_a, "max_bound"_a);
	m.def("estimate_normals", &EstimateNormals,
			"Function to compute the normals of a point cloud",
			"cloud"_a, "search_param"_a = KDTreeSearchParamKNN());
	m.def("orient_normals_to_align_with_direction",
			&OrientNormalsToAlignWithDirection,
			"Function to orient the normals of a point cloud",
			"cloud"_a, "orientation_reference"_a =
			Eigen::Vector3d(0.0, 0.0, 1.0));
	m.def("orient_normals_towards_camera_location",
			&OrientNormalsTowardsCameraLocation,
			"Function to orient the normals of a point cloud",
			"cloud"_a, "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
	m.def("compute_point_cloud_to_point_cloud_distance",
			&ComputePointCloudToPointCloudDistance,
			"Function to compute the ponit to point distances between point clouds",
			"source"_a, "target"_a);
	m.def("compute_point_cloud_mean_and_covariance",
			&ComputePointCloudMeanAndCovariance,
			"Function to compute the mean and covariance matrix of a point cloud",
			"input"_a);
	m.def("compute_point_cloud_mahalanobis_distance",
			&ComputePointCloudMahalanobisDistance,
			"Function to compute the Mahalanobis distance for points in a point cloud",
			"input"_a);
	m.def("compute_point_cloud_nearest_neighbor_distance",
			&ComputePointCloudNearestNeighborDistance,
			"Function to compute the distance from a point to its nearest neighbor in the point cloud",
			"input"_a);
}
