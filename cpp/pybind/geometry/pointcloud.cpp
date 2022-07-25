// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/PointCloud.h"

#include <vector>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_pointcloud(py::module &m) {
    py::class_<PointCloud, PyGeometry3D<PointCloud>,
               std::shared_ptr<PointCloud>, Geometry3D>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<PointCloud>(pointcloud);
    py::detail::bind_copy_functions<PointCloud>(pointcloud);
    pointcloud
            .def(py::init<const std::vector<Eigen::Vector3d> &>(),
                 "Create a PointCloud from points", "points"_a)
            .def("__repr__",
                 [](const PointCloud &pcd) {
                     return std::string("PointCloud with ") +
                            std::to_string(pcd.points_.size()) + " points.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_points", &PointCloud::HasPoints,
                 "Returns ``True`` if the point cloud contains points.")
            .def("has_normals", &PointCloud::HasNormals,
                 "Returns ``True`` if the point cloud contains point normals.")
            .def("has_colors", &PointCloud::HasColors,
                 "Returns ``True`` if the point cloud contains point colors.")
            .def("has_covariances", &PointCloud::HasCovariances,
                 "Returns ``True`` if the point cloud contains covariances.")
            .def("normalize_normals", &PointCloud::NormalizeNormals,
                 "Normalize point normals to length 1.")
            .def("paint_uniform_color", &PointCloud::PaintUniformColor,
                 "color"_a,
                 "Assigns each point in the PointCloud the same color.")
            .def("select_by_index", &PointCloud::SelectByIndex,
                 "Function to select points from input pointcloud into output "
                 "pointcloud.",
                 "indices"_a, "invert"_a = false)
            .def("voxel_down_sample", &PointCloud::VoxelDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with "
                 "a voxel. Normals and colors are averaged if they exist.",
                 "voxel_size"_a)
            .def("voxel_down_sample_and_trace",
                 &PointCloud::VoxelDownSampleAndTrace,
                 "Function to downsample using "
                 "PointCloud::VoxelDownSample. Also records point "
                 "cloud index before downsampling",
                 "voxel_size"_a, "min_bound"_a, "max_bound"_a,
                 "approximate_class"_a = false)
            .def("uniform_down_sample", &PointCloud::UniformDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "uniformly. The sample is performed in the order of the "
                 "points with "
                 "the 0-th point always chosen, not at random.",
                 "every_k_points"_a)
            .def("random_down_sample", &PointCloud::RandomDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "randomly. The sample is generated by randomly sampling "
                 "the indexes from the point cloud.",
                 "sampling_ratio"_a)
            .def("farthest_point_down_sample",
                 &PointCloud::FarthestPointDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with a set of points has farthest distance. The "
                 "sample is performed by selecting the farthest point from "
                 "previous selected points iteratively.",
                 "num_samples"_a)
            .def("crop",
                 (std::shared_ptr<PointCloud>(PointCloud::*)(
                         const AxisAlignedBoundingBox &) const) &
                         PointCloud::Crop,
                 "Function to crop input pointcloud into output pointcloud",
                 "bounding_box"_a)
            .def("crop",
                 (std::shared_ptr<PointCloud>(PointCloud::*)(
                         const OrientedBoundingBox &) const) &
                         PointCloud::Crop,
                 "Function to crop input pointcloud into output pointcloud",
                 "bounding_box"_a)
            .def("remove_non_finite_points", &PointCloud::RemoveNonFinitePoints,
                 "Function to remove non-finite points from the PointCloud",
                 "remove_nan"_a = true, "remove_infinite"_a = true)
            .def("remove_radius_outlier", &PointCloud::RemoveRadiusOutliers,
                 "Function to remove points that have less than nb_points"
                 " in a given sphere of a given radius",
                 "nb_points"_a, "radius"_a, "print_progress"_a = false)
            .def("remove_statistical_outlier",
                 &PointCloud::RemoveStatisticalOutliers,
                 "Function to remove points that are further away from their "
                 "neighbors in average",
                 "nb_neighbors"_a, "std_ratio"_a, "print_progress"_a = false)
            .def("estimate_normals", &PointCloud::EstimateNormals,
                 "Function to compute the normals of a point cloud. Normals "
                 "are oriented with respect to the input point cloud if "
                 "normals exist",
                 "search_param"_a = KDTreeSearchParamKNN(),
                 "fast_normal_computation"_a = true)
            .def("orient_normals_to_align_with_direction",
                 &PointCloud::OrientNormalsToAlignWithDirection,
                 "Function to orient the normals of a point cloud",
                 "orientation_reference"_a = Eigen::Vector3d(0.0, 0.0, 1.0))
            .def("orient_normals_towards_camera_location",
                 &PointCloud::OrientNormalsTowardsCameraLocation,
                 "Function to orient the normals of a point cloud",
                 "camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0))
            .def("orient_normals_consistent_tangent_plane",
                 &PointCloud::OrientNormalsConsistentTangentPlane,
                 "Function to orient the normals with respect to consistent "
                 "tangent planes",
                 "k"_a)
            .def("compute_point_cloud_distance",
                 &PointCloud::ComputePointCloudDistance,
                 "For each point in the source point cloud, compute the "
                 "distance to the target point cloud.",
                 "target"_a)
            .def_static(
                    "estimate_point_covariances",
                    &PointCloud::EstimatePerPointCovariances,
                    "Static function to compute the covariance matrix for "
                    "each "
                    "point in the given point cloud, doesn't change the input",
                    "input"_a, "search_param"_a = KDTreeSearchParamKNN())
            .def("estimate_covariances", &PointCloud::EstimateCovariances,
                 "Function to compute the covariance matrix for each point "
                 "in the point cloud",
                 "search_param"_a = KDTreeSearchParamKNN())
            .def("compute_mean_and_covariance",
                 &PointCloud::ComputeMeanAndCovariance,
                 "Function to compute the mean and covariance matrix of a "
                 "point cloud.")
            .def("compute_mahalanobis_distance",
                 &PointCloud::ComputeMahalanobisDistance,
                 "Function to compute the Mahalanobis distance for points in a "
                 "point cloud. See: "
                 "https://en.wikipedia.org/wiki/Mahalanobis_distance.")
            .def("compute_nearest_neighbor_distance",
                 &PointCloud::ComputeNearestNeighborDistance,
                 "Function to compute the distance from a point to its nearest "
                 "neighbor in the point cloud")
            .def("compute_convex_hull", &PointCloud::ComputeConvexHull,
                 "joggle_inputs"_a = false, R"doc(
Computes the convex hull of the point cloud.

Args:
     joggle_inputs (bool): If True allows the algorithm to add random noise to
          the points to work around degenerate inputs. This adds the 'QJ'
          option to the qhull command.

Returns:
     tuple(open3d.geometry.TriangleMesh, list): The triangle mesh of the convex
     hull and the list of point indices that are part of the convex hull.
)doc")
            .def("hidden_point_removal", &PointCloud::HiddenPointRemoval,
                 "Removes hidden points from a point cloud and returns a mesh "
                 "of the remaining points. Based on Katz et al. 'Direct "
                 "Visibility of Point Sets', 2007. Additional information "
                 "about the choice of radius for noisy point clouds can be "
                 "found in Mehra et. al. 'Visibility of Noisy Point Cloud "
                 "Data', 2010.",
                 "camera_location"_a, "radius"_a)
            .def("cluster_dbscan", &PointCloud::ClusterDBSCAN,
                 "Cluster PointCloud using the DBSCAN algorithm  Ester et al., "
                 "'A Density-Based Algorithm for Discovering Clusters in Large "
                 "Spatial Databases with Noise', 1996. Returns a list of point "
                 "labels, -1 indicates noise according to the algorithm.",
                 "eps"_a, "min_points"_a, "print_progress"_a = false)
            .def("segment_plane", &PointCloud::SegmentPlane,
                 "Segments a plane in the point cloud using the RANSAC "
                 "algorithm.",
                 "distance_threshold"_a, "ransac_n"_a, "num_iterations"_a,
                 "probability"_a = 0.99999999)
            .def_static(
                    "create_from_depth_image",
                    &PointCloud::CreateFromDepthImage,
                    R"(Factory function to create a pointcloud from a depth image and a
camera. Given depth value d at (u, v) image coordinate, the corresponding 3d point is:

    - z = d / depth_scale
    - x = (u - cx) * z / fx
    - y = (v - cy) * z / fy)",
                    "depth"_a, "intrinsic"_a,
                    "extrinsic"_a = Eigen::Matrix4d::Identity(),
                    "depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0,
                    "stride"_a = 1, "project_valid_depth_only"_a = true)
            .def_static(
                    "create_from_rgbd_image", &PointCloud::CreateFromRGBDImage,
                    "Factory function to create a pointcloud from an RGB-D "
                    "image and a camera. Given depth value d at (u, "
                    "v) image coordinate, the corresponding 3d point is:\n\n"
                    R"(    - z = d / depth_scale
    - x = (u - cx) * z / fx
    - y = (v - cy) * z / fy)",
                    "image"_a, "intrinsic"_a,
                    "extrinsic"_a = Eigen::Matrix4d::Identity(),
                    "project_valid_depth_only"_a = true)
            .def_readwrite("points", &PointCloud::points_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "coordinates.")
            .def_readwrite("normals", &PointCloud::normals_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "normals.")
            .def_readwrite(
                    "colors", &PointCloud::colors_,
                    "``float64`` array of shape ``(num_points, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of points.")
            .def_readwrite("covariances", &PointCloud::covariances_,
                           "``float64`` array of shape ``(num_points, 3, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "covariances.");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_colors");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_normals");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_points");
    docstring::ClassMethodDocInject(m, "PointCloud", "normalize_normals");
    docstring::ClassMethodDocInject(
            m, "PointCloud", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "select_by_index",
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
            m, "PointCloud", "random_down_sample",
            {{"sampling_ratio",
              "Sampling ratio, the ratio of number of selected points to total "
              "number of points[0-1]"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "crop",
            {{"bounding_box", "AxisAlignedBoundingBox to crop points"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_non_finite_points",
            {{"remove_nan", "Remove NaN values from the PointCloud"},
             {"remove_infinite",
              "Remove infinite values from the PointCloud"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_radius_outlier",
            {{"nb_points", "Number of points within the radius."},
             {"radius", "Radius of the sphere."},
             {"print_progress", "Set to True to print progress bar."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_statistical_outlier",
            {{"nb_neighbors", "Number of neighbors around the target point."},
             {"std_ratio", "Standard deviation ratio."},
             {"print_progress", "Set to True to print progress bar."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "estimate_normals",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."},
             {"fast_normal_computation",
              "If true, the normal estimation uses a non-iterative method to "
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
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_consistent_tangent_plane",
            {{"k",
              "Number of k nearest neighbors used in constructing the "
              "Riemannian graph used to propagate normal orientation."}});
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_point_cloud_distance",
                                    {{"target", "The target point cloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "estimate_point_covariances",
            {{"input", "The input point cloud."},
             {"search_param",
              "The KDTree search parameters for neighborhood search."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "estimate_covariances",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."}});
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_mean_and_covariance");
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_mahalanobis_distance");
    docstring::ClassMethodDocInject(m, "PointCloud",
                                    "compute_nearest_neighbor_distance");
    docstring::ClassMethodDocInject(
            m, "PointCloud", "hidden_point_removal",
            {{"input", "The input point cloud."},
             {"camera_location",
              "All points not visible from that location will be removed"},
             {"radius", "The radius of the sperical projection"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "cluster_dbscan",
            {{"eps",
              "Density parameter that is used to find neighbouring points."},
             {"min_points", "Minimum number of points to form a cluster."},
             {"print_progress",
              "If true the progress is visualized in the console."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "segment_plane",
            {{"distance_threshold",
              "Max distance a point can be from the plane model, and still be "
              "considered an inlier."},
             {"ransac_n",
              "Number of initial points to be considered inliers in each "
              "iteration."},
             {"num_iterations", "Number of iterations."},
             {"probability",
              "Expected probability of finding the optimal plane."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "create_from_depth_image",
            {{"depth",
              "The input depth image can be either a float image, or a "
              "uint16_t image."},
             {"intrinsic", "Intrinsic parameters of the camera."},
             {"extrnsic", "Extrinsic parameters of the camera."},
             {"depth_scale", "The depth is scaled by 1 / depth_scale."},
             {"depth_trunc", "Truncated at depth_trunc distance."},
             {"stride",
              "Sampling factor to support coarse point cloud extraction."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "create_from_rgbd_image",
            {{"image", "The input image."},
             {"intrinsic", "Intrinsic parameters of the camera."},
             {"extrnsic", "Extrinsic parameters of the camera."}});
}

void pybind_pointcloud_methods(py::module &m) {}

}  // namespace geometry
}  // namespace open3d
