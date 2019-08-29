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
                 "Assigns each point in the PointCloud the same color.")
            .def("select_down_sample", &geometry::PointCloud::SelectDownSample,
                 "Function to select points from input pointcloud into output "
                 "pointcloud. ``indices``: "
                 "Indices of points to be selected. ``invert``: Set to "
                 "``True`` to "
                 "invert the selection of indices.",
                 "indices"_a, "invert"_a = false)
            .def("voxel_down_sample", &geometry::PointCloud::VoxelDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with "
                 "a voxel",
                 "voxel_size"_a)
            .def("voxel_down_sample_and_trace",
                 &geometry::PointCloud::VoxelDownSampleAndTrace,
                 "Function to downsample using "
                 "geometry::PointCloud::VoxelDownSample also records point "
                 "cloud index before downsampling",
                 "voxel_size"_a, "min_bound"_a, "max_bound"_a,
                 "approximate_class"_a = false)
            .def("uniform_down_sample",
                 &geometry::PointCloud::UniformDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "uniformly. The sample is performed in the order of the "
                 "points with "
                 "the 0-th point always chosen, not at random.",
                 "every_k_points"_a)
            .def("crop", &geometry::PointCloud::Crop,
                 "Function to crop input pointcloud into output pointcloud",
                 "min_bound"_a, "max_bound"_a)
            .def("remove_none_finite_points",
                 &geometry::PointCloud::RemoveNoneFinitePoints,
                 "Function to remove none-finite points from the PointCloud",
                 "remove_nan"_a = true, "remove_infinite"_a = true)
            .def("remove_radius_outlier",
                 &geometry::PointCloud::RemoveRadiusOutliers,
                 "Function to remove points that have less than nb_points"
                 " in a given sphere of a given radius",
                 "nb_points"_a, "radius"_a)
            .def("remove_statistical_outlier",
                 &geometry::PointCloud::RemoveStatisticalOutliers,
                 "Function to remove points that are further away from their "
                 "neighbors in average",
                 "nb_neighbors"_a, "std_ratio"_a)
            .def("estimate_normals", &geometry::PointCloud::EstimateNormals,
                 "Function to compute the normals of a point cloud. Normals "
                 "are oriented with respect to the input point cloud if "
                 "normals exist",
                 "search_param"_a = geometry::KDTreeSearchParamKNN(),
                 "fast_normal_computation"_a = true)
            .def("orient_normals_to_align_with_direction",
                 &geometry::PointCloud::OrientNormalsToAlignWithDirection,
                 "Function to orient the normals of a point cloud",
                 "orientation_reference"_a = Eigen::Vector3d(0.0, 0.0, 1.0))
            .def("orient_normals_towards_camera_location",
                 &geometry::PointCloud::OrientNormalsTowardsCameraLocation,
                 "Function to orient the normals of a point cloud",
                 "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0))
            .def("compute_point_cloud_distance",
                 &geometry::PointCloud::ComputePointCloudDistance,
                 "For each point in the source point cloud, compute the "
                 "distance to "
                 "the target point cloud.",
                 "target"_a)
            .def("compute_mean_and_covariance",
                 &geometry::PointCloud::ComputeMeanAndCovariance,
                 "Function to compute the mean and covariance matrix of a "
                 "point "
                 "cloud.")
            .def("compute_mahalanobis_distance",
                 &geometry::PointCloud::ComputeMahalanobisDistance,
                 "Function to compute the Mahalanobis distance for points in a "
                 "point "
                 "cloud. See: "
                 "https://en.wikipedia.org/wiki/Mahalanobis_distance.")
            .def("compute_nearest_neighbor_distance",
                 &geometry::PointCloud::ComputeNearestNeighborDistance,
                 "Function to compute the distance from a point to its nearest "
                 "neighbor in the point cloud")
            .def("compute_convex_hull",
                 &geometry::PointCloud::ComputeConvexHull,
                 "Computes the convex hull of the point cloud.")
            .def("cluster_dbscan", &geometry::PointCloud::ClusterDBSCAN,
                 "Cluster PointCloud using the DBSCAN algorithm  Ester et al., "
                 "'A Density-Based Algorithm for Discovering Clusters in Large "
                 "Spatial Databases with Noise', 1996. Returns a list of point "
                 "labels, -1 indicates noise according to the algorithm.",
                 "eps"_a, "min_points"_a, "print_progress"_a = false)
            .def_static(
                    "create_from_depth_image",
                    &geometry::PointCloud::CreateFromDepthImage,
                    R"(Factory function to create a pointcloud from a depth image and a
        camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
        point is:

              - z = d / depth_scale
              - x = (u - cx) * z / fx
              - y = (v - cy) * z / fy
        )",
                    "depth"_a, "intrinsic"_a,
                    "extrinsic"_a = Eigen::Matrix4d::Identity(),
                    "depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0,
                    "stride"_a = 1)
            .def_static(
                    "create_from_rgbd_image",
                    &geometry::PointCloud::CreateFromRGBDImage,
                    R"(Factory function to create a pointcloud from an RGB-D image and a
        camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
        point is:

              - z = d / depth_scale
              - x = (u - cx) * z / fx
              - y = (v - cy) * z / fy
        )",
                    "image"_a, "intrinsic"_a,
                    "extrinsic"_a = Eigen::Matrix4d::Identity())
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
    docstring::ClassMethodDocInject(
            m, "PointCloud", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "select_down_sample",
            {{"indices", "Indices of points to be selected."},
             {"invert",
              "Set to ``True`` to invert the selection of indices."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "voxel_down_sample",
            {{"voxel_size", "Voxel size to downsample into."},
             {"invert", "set to ``True`` to invert the selection of indices"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "voxel_down_sample_and_trace",
            {{"voxel_size", "Voxel size to downsample into."},
             {"min_bound", "Minimum coordinate of voxel boundaries"},
             {"max_bound", "Maximum coordinate of voxel boundaries"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "uniform_down_sample",
            {{"every_k_points",
              "Sample rate, the selected point indices are [0, k, 2k, ...]"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "crop",
            {{"min_bound", "Minimum bound for point coordinate"},
             {"max_bound", "Maximum bound for point coordinate"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_none_finite_points",
            {{"remove_nan", "Remove NaN values from the PointCloud"},
             {"remove_infinite",
              "Remove infinite values from the PointCloud"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_radius_outlier",
            {{"nb_points", "Number of points within the radius."},
             {"radius", "Radius of the sphere."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_statistical_outlier",
            {{"nb_neighbors", "Number of neighbors around the target point."},
             {"std_ratio", "Standard deviation ratio."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "estimate_normals",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."},
             {"fast_normal_computation",
              "If true, the normal estiamtion uses a non-iterative method to "
              "extract the eigenvector from the covariance matrix. This is "
              "faster, but is not as numerical stable."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_to_align_with_direction",
            {{"orientation_reference",
              "Normals are oriented with respect to orientation_reference."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_towards_camera_location",
            {{"camera_location",
              "Normals are oriented with towards the camera_location."}});
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_point_cloud_distance",
                                    {{"target", "The target point cloud."}});
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_mean_and_covariance");
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_mahalanobis_distance");
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_nearest_neighbor_distance");
    docstring::ClassMethodDocInject(m, "PointCloud", "compute_convex_hull",
                                    {{"input", "The input point cloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "cluster_dbscan",
            {{"eps",
              "Density parameter that is used to find neighbouring points."},
             {"min_points", "Minimum number of points to form a cluster."},
             {"print_progress",
              "If true the progress is visualized in the console."}});
    docstring::ClassMethodDocInject(m, "PointCloud", "create_from_depth_image");
    docstring::ClassMethodDocInject(m, "PointCloud", "create_from_rgbd_image");
}

void pybind_pointcloud_methods(py::module &m) {}
