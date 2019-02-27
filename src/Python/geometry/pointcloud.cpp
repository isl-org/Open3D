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

#include "Python/geometry/geometry_trampoline.h"
#include "Python/geometry/geometry.h"

#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/Image.h>
#include <Open3D/Geometry/RGBDImage.h>
#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <Open3D/IO/ClassIO/PointCloudIO.h>
using namespace open3d;

void pybind_pointcloud(py::module &m) {
    py::class_<geometry::PointCloud, PyGeometry3D<geometry::PointCloud>,
               std::shared_ptr<geometry::PointCloud>, geometry::Geometry3D>
            pointcloud(m, "PointCloud", "PointCloud");
    py::detail::bind_default_constructor<geometry::PointCloud>(pointcloud);
    py::detail::bind_copy_functions<geometry::PointCloud>(pointcloud);
    pointcloud
            .def("__repr__",
                 [](const geometry::PointCloud &pcd) {
                     return std::string("geometry::PointCloud with ") +
                            std::to_string(pcd.points_.size()) + " points.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_points", &geometry::PointCloud::HasPoints)
            .def("has_normals", &geometry::PointCloud::HasNormals)
            .def("has_colors", &geometry::PointCloud::HasColors)
            .def("normalize_normals", &geometry::PointCloud::NormalizeNormals)
            .def("paint_uniform_color",
                 &geometry::PointCloud::PaintUniformColor)
            .def_readwrite("points", &geometry::PointCloud::points_)
            .def_readwrite("normals", &geometry::PointCloud::normals_)
            .def_readwrite("colors", &geometry::PointCloud::colors_);
}

void pybind_pointcloud_methods(py::module &m) {
    m.def("read_point_cloud",
          [](const std::string &filename, const std::string &format) {
              geometry::PointCloud pcd;
              io::ReadPointCloud(filename, pcd, format);
              return pcd;
          },
          "Function to read geometry::PointCloud from file", "filename"_a,
          "format"_a = "auto");
    m.def("write_point_cloud",
          [](const std::string &filename,
             const geometry::PointCloud &pointcloud, bool write_ascii,
             bool compressed) {
              return io::WritePointCloud(filename, pointcloud, write_ascii,
                                         compressed);
          },
          "Function to write geometry::PointCloud to file", "filename"_a,
          "pointcloud"_a, "write_ascii"_a = false, "compressed"_a = false);
    m.def("create_point_cloud_from_depth_image",
          &geometry::CreatePointCloudFromDepthImage,
          "Factory function to create a pointcloud from a depth image and a "
          "camera.\n"
          "Given depth value d at (u, v) image coordinate, the corresponding "
          "3d point is:\n"
          "    z = d / depth_scale\n"
          "    x = (u - cx) * z / fx\n"
          "    y = (v - cy) * z / fy",
          "depth"_a, "intrinsic"_a, "extrinsic"_a = Eigen::Matrix4d::Identity(),
          "depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0, "stride"_a = 1);
    m.def("create_point_cloud_from_rgbd_image",
          &geometry::CreatePointCloudFromRGBDImage,
          "Factory function to create a pointcloud from an RGB-D image and a "
          "camera.\n"
          "Given depth value d at (u, v) image coordinate, the corresponding "
          "3d point is:\n"
          "    z = d / depth_scale\n"
          "    x = (u - cx) * z / fx\n"
          "    y = (v - cy) * z / fy",
          "image"_a, "intrinsic"_a,
          "extrinsic"_a = Eigen::Matrix4d::Identity());
    m.def("select_down_sample", &geometry::SelectDownSample,
          "Function to select points from input pointcloud into output "
          "pointcloud",
          "input"_a, "indices"_a, "invert"_a = false);
    m.def("voxel_down_sample", &geometry::VoxelDownSample,
          "Function to downsample input pointcloud into output pointcloud with "
          "a voxel",
          "input"_a, "voxel_size"_a);
    m.def("voxel_down_sample_and_trace", &geometry::VoxelDownSampleAndTrace,
          "Function to downsample using geometry::VoxelDownSample also records "
          "point "
          "cloud index before downsampling",
          "input"_a, "voxel_size"_a, "min_bound"_a, "max_bound"_a,
          "approximate_class"_a = false);
    m.def("uniform_down_sample", &geometry::UniformDownSample,
          "Function to downsample input pointcloud into output pointcloud "
          "uniformly",
          "input"_a, "every_k_points"_a);
    m.def("crop_point_cloud", &geometry::CropPointCloud,
          "Function to crop input pointcloud into output pointcloud", "input"_a,
          "min_bound"_a, "max_bound"_a);
    m.def("radius_outlier_removal", &geometry::RemoveRadiusOutliers,
          "Function to remove points that have less than nb_points"
          " in a given sphere of a given radius",
          "input"_a, "nb_points"_a, "radius"_a);
    m.def("statistical_outlier_removal", &geometry::RemoveStatisticalOutliers,
          "Function to remove points that are further away from their "
          "neighbours in average",
          "input"_a, "nb_neighbors"_a, "std_ratio"_a);
    m.def("estimate_normals", &geometry::EstimateNormals,
          "Function to compute the normals of a point cloud", "cloud"_a,
          "search_param"_a = geometry::KDTreeSearchParamKNN());
    m.def("orient_normals_to_align_with_direction",
          &geometry::OrientNormalsToAlignWithDirection,
          "Function to orient the normals of a point cloud", "cloud"_a,
          "orientation_reference"_a = Eigen::Vector3d(0.0, 0.0, 1.0));
    m.def("orient_normals_towards_camera_location",
          &geometry::OrientNormalsTowardsCameraLocation,
          "Function to orient the normals of a point cloud", "cloud"_a,
          "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
    m.def("compute_point_cloud_to_point_cloud_distance",
          &geometry::ComputePointCloudToPointCloudDistance,
          "Function to compute the ponit to point distances between point "
          "clouds",
          "source"_a, "target"_a);
    m.def("compute_point_cloud_mean_and_covariance",
          &geometry::ComputePointCloudMeanAndCovariance,
          "Function to compute the mean and covariance matrix of a point cloud",
          "input"_a);
    m.def("compute_point_cloud_mahalanobis_distance",
          &geometry::ComputePointCloudMahalanobisDistance,
          "Function to compute the Mahalanobis distance for points in a point "
          "cloud",
          "input"_a);
    m.def("compute_point_cloud_nearest_neighbor_distance",
          &geometry::ComputePointCloudNearestNeighborDistance,
          "Function to compute the distance from a point to its nearest "
          "neighbor in the point cloud",
          "input"_a);
}
