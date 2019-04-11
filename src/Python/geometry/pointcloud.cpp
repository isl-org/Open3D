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

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Python/docstring.h"
#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_pointcloud(py::module &m) {
    py::class_<geometry::PointCloud, PyGeometry3D<geometry::PointCloud>,
               std::shared_ptr<geometry::PointCloud>, geometry::Geometry3D>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
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
            .def("has_points", &geometry::PointCloud::HasPoints,
                 "Returns ``True`` if the point cloud contains points.")
            .def("has_normals", &geometry::PointCloud::HasNormals,
                 "Returns ``True`` if the point cloud contains point normals.")
            .def("has_colors", &geometry::PointCloud::HasColors,
                 "Returns ``True`` if the point cloud contains point colors.")
            .def("normalize_normals", &geometry::PointCloud::NormalizeNormals,
                 "Normalize point normals to length 1.")
            .def("paint_uniform_color",
                 &geometry::PointCloud::PaintUniformColor, "color"_a,
                 "Assign uniform color to all points.")
            .def_readwrite("points", &geometry::PointCloud::points_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "coordinates.")
            .def_readwrite("normals", &geometry::PointCloud::normals_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "normals.")
            .def_readwrite(
                    "colors", &geometry::PointCloud::colors_,
                    "``float64`` array of shape ``(num_points, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of points.");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_colors");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_normals");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_points");
    docstring::ClassMethodDocInject(m, "PointCloud", "normalize_normals");
    docstring::ClassMethodDocInject(m, "PointCloud", "paint_uniform_color",
                                    {{"color", "RGB color."}});
}

void pybind_pointcloud_methods(py::module &m) {
    m.def("create_point_cloud_from_depth_image",
          &geometry::CreatePointCloudFromDepthImage,
          R"(Factory function to create a pointcloud from a depth image and a
camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
point is:

      - z = d / depth_scale
      - x = (u - cx) * z / fx
      - y = (v - cy) * z / fy
)",
          "depth"_a, "intrinsic"_a, "extrinsic"_a = Eigen::Matrix4d::Identity(),
          "depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0, "stride"_a = 1);
    docstring::FunctionDocInject(m, "create_point_cloud_from_depth_image");

    m.def("create_point_cloud_from_rgbd_image",
          &geometry::CreatePointCloudFromRGBDImage,
          R"(Factory function to create a pointcloud from an RGB-D image and a
camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
point is:

      - z = d / depth_scale
      - x = (u - cx) * z / fx
      - y = (v - cy) * z / fy
)",
          "image"_a, "intrinsic"_a,
          "extrinsic"_a = Eigen::Matrix4d::Identity());
    docstring::FunctionDocInject(m, "create_point_cloud_from_rgbd_image");

    // Overloaded function, do not inject docs. Keep commented out for future.
    m.def("select_down_sample",
          (std::shared_ptr<geometry::PointCloud>(*)(
                  const geometry::PointCloud &, const std::vector<size_t> &,
                  bool)) &
                  geometry::SelectDownSample,
          "Function to select points from input pointcloud into output "
          "pointcloud. ``input``: The input triangle point cloud. ``indices``: "
          "Indices of points to be selected. ``invert``: Set to ``True`` to "
          "invert the selection of indices.",
          "input"_a, "indices"_a, "invert"_a = false);
    // docstring::FunctionDocInject(
    //         m, "select_down_sample",
    //         {{"input", "The input point cloud."},
    //          {"indices", "Indices of points to be selected."},
    //          {"invert",
    //           "Set to ``True`` to invert the selection of indices."}});

    m.def("voxel_down_sample", &geometry::VoxelDownSample,
          "Function to downsample input pointcloud into output pointcloud with "
          "a voxel",
          "input"_a, "voxel_size"_a);
    docstring::FunctionDocInject(
            m, "voxel_down_sample",
            {{"input", "The input point cloud."},
             {"voxel_size", "Voxel size to downsample into."},
             {"invert", "set to ``True`` to invert the selection of indices"}});

    m.def("voxel_down_sample_and_trace", &geometry::VoxelDownSampleAndTrace,
          "Function to downsample using geometry::VoxelDownSample also records "
          "point "
          "cloud index before downsampling",
          "input"_a, "voxel_size"_a, "min_bound"_a, "max_bound"_a,
          "approximate_class"_a = false);
    docstring::FunctionDocInject(
            m, "voxel_down_sample_and_trace",
            {{"input", "The input point cloud."},
             {"voxel_size", "Voxel size to downsample into."},
             {"min_bound", "Minimum coordinate of voxel boundaries"},
             {"max_bound", "Maximum coordinate of voxel boundaries"}});

    m.def("uniform_down_sample", &geometry::UniformDownSample,
          "Function to downsample input pointcloud into output pointcloud "
          "uniformly. The sample is performed in the order of the points with "
          "the 0-th point always chosen, not at random.",
          "input"_a, "every_k_points"_a);
    docstring::FunctionDocInject(
            m, "uniform_down_sample",
            {{"input", "The input point cloud."},
             {"every_k_points",
              "Sample rate, the selected point indices are [0, k, 2k, ...]"}});

    m.def("crop_point_cloud", &geometry::CropPointCloud,
          "Function to crop input pointcloud into output pointcloud", "input"_a,
          "min_bound"_a, "max_bound"_a);
    docstring::FunctionDocInject(
            m, "crop_point_cloud",
            {{"input", "The input point cloud."},
             {"min_bound", "Minimum bound for point coordinate"},
             {"max_bound", "Maximum bound for point coordinate"}});

    m.def("radius_outlier_removal", &geometry::RemoveRadiusOutliers,
          "Function to remove points that have less than nb_points"
          " in a given sphere of a given radius",
          "input"_a, "nb_points"_a, "radius"_a);
    docstring::FunctionDocInject(
            m, "radius_outlier_removal",
            {{"input", "The input point cloud."},
             {"nb_points", "Number of points within the radius."},
             {"radius", "Radius of the sphere."}});

    m.def("statistical_outlier_removal", &geometry::RemoveStatisticalOutliers,
          "Function to remove points that are further away from their "
          "neighbors in average",
          "input"_a, "nb_neighbors"_a, "std_ratio"_a);
    docstring::FunctionDocInject(
            m, "statistical_outlier_removal",
            {{"input", "The input point cloud."},
             {"nb_neighbors", "Number of neighbors around the target point."},
             {"std_ratio", "Standard deviation ratio."}});

    m.def("estimate_normals", &geometry::EstimateNormals,
          "Function to compute the normals of a point cloud. Normals are "
          "oriented with respect to the input point cloud if normals exist",
          "cloud"_a, "search_param"_a = geometry::KDTreeSearchParamKNN());
    docstring::FunctionDocInject(
            m, "estimate_normals",
            {{"cloud",
              "The input point cloud. It also stores the output normals."},
             {"search_param",
              "The KDTree search parameters for neighborhood search."}});

    m.def("orient_normals_to_align_with_direction",
          &geometry::OrientNormalsToAlignWithDirection,
          "Function to orient the normals of a point cloud", "cloud"_a,
          "orientation_reference"_a = Eigen::Vector3d(0.0, 0.0, 1.0));
    docstring::FunctionDocInject(
            m, "orient_normals_to_align_with_direction",
            {{"cloud", "The input point cloud. It must have normals."},
             {"orientation_reference",
              "Normals are oriented with respect to orientation_reference."}});

    m.def("orient_normals_towards_camera_location",
          &geometry::OrientNormalsTowardsCameraLocation,
          "Function to orient the normals of a point cloud", "cloud"_a,
          "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
    docstring::FunctionDocInject(
            m, "orient_normals_towards_camera_location",
            {{"cloud",
              "The input point cloud. It also stores the output normals."},
             {"camera_location",
              "Normals are oriented with towards the camera_location."}});

    m.def("compute_point_cloud_to_point_cloud_distance",
          &geometry::ComputePointCloudToPointCloudDistance,
          "For each point in the source point cloud, compute the distance to "
          "the target point cloud.",
          "source"_a, "target"_a);
    docstring::FunctionDocInject(
            m, "compute_point_cloud_to_point_cloud_distance",
            {{"source",
              "The source point cloud. The results has the same number of "
              "elements as the size of the source point cloud."},
             {"target", "The target point cloud."}});

    m.def("compute_point_cloud_mean_and_covariance",
          &geometry::ComputePointCloudMeanAndCovariance,
          "Function to compute the mean and covariance matrix of a point "
          "cloud.",
          "input"_a);
    docstring::FunctionDocInject(m, "compute_point_cloud_mean_and_covariance",
                                 {{"input", "The input point cloud."}});

    m.def("compute_point_cloud_mahalanobis_distance",
          &geometry::ComputePointCloudMahalanobisDistance,
          "Function to compute the Mahalanobis distance for points in a point "
          "cloud. See: https://en.wikipedia.org/wiki/Mahalanobis_distance.",
          "input"_a);
    docstring::FunctionDocInject(m, "compute_point_cloud_mahalanobis_distance",
                                 {{"input", "The input point cloud."}});

    m.def("compute_point_cloud_nearest_neighbor_distance",
          &geometry::ComputePointCloudNearestNeighborDistance,
          "Function to compute the distance from a point to its nearest "
          "neighbor in the point cloud",
          "input"_a);
    docstring::FunctionDocInject(
            m, "compute_point_cloud_nearest_neighbor_distance",
            {{"input", "The input point cloud."}});
}
