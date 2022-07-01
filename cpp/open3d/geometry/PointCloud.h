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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/utility/Optional.h"

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;
class RGBDImage;
class TriangleMesh;
class VoxelGrid;

/// \class PointCloud
///
/// \brief A point cloud consists of point coordinates, and optionally point
/// colors and point normals.
class PointCloud : public Geometry3D {
public:
    /// \brief Default Constructor.
    PointCloud() : Geometry3D(Geometry::GeometryType::PointCloud) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param points Points coordinates.
    PointCloud(const std::vector<Eigen::Vector3d> &points)
        : Geometry3D(Geometry::GeometryType::PointCloud), points_(points) {}
    ~PointCloud() override {}

public:
    PointCloud &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override;
    PointCloud &Transform(const Eigen::Matrix4d &transformation) override;
    PointCloud &Translate(const Eigen::Vector3d &translation,
                          bool relative = true) override;
    PointCloud &Scale(const double scale,
                      const Eigen::Vector3d &center) override;
    PointCloud &Rotate(const Eigen::Matrix3d &R,
                       const Eigen::Vector3d &center) override;

    PointCloud &operator+=(const PointCloud &cloud);
    PointCloud operator+(const PointCloud &cloud) const;

    /// Returns 'true' if the point cloud contains points.
    bool HasPoints() const { return points_.size() > 0; }

    /// Returns `true` if the point cloud contains point normals.
    bool HasNormals() const {
        return points_.size() > 0 && normals_.size() == points_.size();
    }

    /// Returns `true` if the point cloud contains point colors.
    bool HasColors() const {
        return points_.size() > 0 && colors_.size() == points_.size();
    }

    /// Returns 'true' if the point cloud contains per-point covariance matrix.
    bool HasCovariances() const {
        return !points_.empty() && covariances_.size() == points_.size();
    }

    /// Normalize point normals to length 1.
    PointCloud &NormalizeNormals() {
        for (size_t i = 0; i < normals_.size(); i++) {
            normals_[i].normalize();
        }
        return *this;
    }

    /// Assigns each point in the PointCloud the same color.
    ///
    /// \param color  RGB colors of points.
    PointCloud &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(colors_, points_.size(), color);
        return *this;
    }

    /// \brief Remove all points from the point cloud that have a nan entry, or
    /// infinite entries.
    ///
    /// Also removes the corresponding normals and color entries.
    ///
    /// \param remove_nan Remove NaN values from the PointCloud.
    /// \param remove_infinite Remove infinite values from the PointCloud.
    PointCloud &RemoveNonFinitePoints(bool remove_nan = true,
                                      bool remove_infinite = true);

    /// \brief Function to select points from \p input pointcloud into
    /// \p output pointcloud.
    ///
    /// Points with indices in \p indices are selected.
    ///
    /// \param indices Indices of points to be selected.
    /// \param invert Set to `True` to invert the selection of indices.
    std::shared_ptr<PointCloud> SelectByIndex(
            const std::vector<size_t> &indices, bool invert = false) const;

    /// \brief Function to downsample input pointcloud into output pointcloud
    /// with a voxel.
    ///
    /// Normals and colors are averaged if they exist.
    ///
    /// \param voxel_size Defines the resolution of the voxel grid,
    /// smaller value leads to denser output point cloud.
    std::shared_ptr<PointCloud> VoxelDownSample(double voxel_size) const;

    /// \brief Function to downsample using geometry.PointCloud.VoxelDownSample
    ///
    /// Also records point cloud index before downsampling.
    ///
    /// \param voxel_size Voxel size to downsample into.
    /// \param min_bound Minimum coordinate of voxel boundaries
    /// \param max_bound Maximum coordinate of voxel boundaries
    /// \param approximate_class Whether to approximate.
    std::tuple<std::shared_ptr<PointCloud>,
               Eigen::MatrixXi,
               std::vector<std::vector<int>>>
    VoxelDownSampleAndTrace(double voxel_size,
                            const Eigen::Vector3d &min_bound,
                            const Eigen::Vector3d &max_bound,
                            bool approximate_class = false) const;

    /// \brief Function to downsample input pointcloud into output pointcloud
    /// uniformly.
    ///
    /// The sample is performed in the order of the points with the 0-th point
    /// always chosen, not at random.
    ///
    /// \param every_k_points Sample rate, the selected point indices are [0, k,
    /// 2k, …].
    std::shared_ptr<PointCloud> UniformDownSample(size_t every_k_points) const;

    /// \brief Function to downsample input pointcloud into output pointcloud
    /// randomly.
    ///
    /// The sample is performed by randomly selecting the index of the points
    /// in the pointcloud.
    ///
    /// \param sampling_ratio Sampling ratio, the ratio of sample to total
    /// number of points in the pointcloud.
    std::shared_ptr<PointCloud> RandomDownSample(double sampling_ratio) const;

    /// \brief Function to downsample input pointcloud into output pointcloud
    /// with a set of points has farthest distance.
    ///
    /// The sample is performed by selecting the farthest point from previous
    /// selected points iteratively.
    ///
    /// \param num_samples Number of points to be sampled.
    std::shared_ptr<PointCloud> FarthestPointDownSample(
            size_t num_samples) const;

    /// \brief Function to crop pointcloud into output pointcloud
    ///
    /// All points with coordinates outside the bounding box \p bbox are
    /// clipped.
    ///
    /// \param bbox AxisAlignedBoundingBox to crop points.
    std::shared_ptr<PointCloud> Crop(const AxisAlignedBoundingBox &bbox) const;

    /// \brief Function to crop pointcloud into output pointcloud
    ///
    /// All points with coordinates outside the bounding box \p bbox are
    /// clipped.
    ///
    /// \param bbox OrientedBoundingBox to crop points.
    std::shared_ptr<PointCloud> Crop(const OrientedBoundingBox &bbox) const;

    /// \brief Function to remove points that have less than \p nb_points in a
    /// sphere of a given radius.
    ///
    /// \param nb_points Number of points within the radius.
    /// \param search_radius Radius of the sphere.
    /// \param print_progress Whether to print the progress bar.
    std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
    RemoveRadiusOutliers(size_t nb_points,
                         double search_radius,
                         bool print_progress = false) const;

    /// \brief Function to remove points that are further away from their
    /// \p nb_neighbor neighbors in average.
    ///
    /// \param nb_neighbors Number of neighbors around the target point.
    /// \param std_ratio Standard deviation ratio.
    std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
    RemoveStatisticalOutliers(size_t nb_neighbors,
                              double std_ratio,
                              bool print_progress = false) const;

    /// \brief Function to compute the normals of a point cloud.
    ///
    /// Normals are oriented with respect to the input point cloud if normals
    /// exist.
    ///
    /// \param search_param The KDTree search parameters for neighborhood
    /// search.
    /// \param fast_normal_computation If true, the normal estimation
    /// uses a non-iterative method to extract the eigenvector from the
    /// covariance matrix. This is faster, but is not as numerical stable.
    void EstimateNormals(
            const KDTreeSearchParam &search_param = KDTreeSearchParamKNN(),
            bool fast_normal_computation = true);

    /// \brief Function to orient the normals of a point cloud.
    ///
    /// \param orientation_reference Normals are oriented with respect to
    /// orientation_reference.
    void OrientNormalsToAlignWithDirection(
            const Eigen::Vector3d &orientation_reference =
                    Eigen::Vector3d(0.0, 0.0, 1.0));

    /// \brief Function to orient the normals of a point cloud.
    ///
    /// \param camera_location Normals are oriented with towards the
    /// camera_location.
    void OrientNormalsTowardsCameraLocation(
            const Eigen::Vector3d &camera_location = Eigen::Vector3d::Zero());

    /// \brief Function to consistently orient estimated normals based on
    /// consistent tangent planes as described in Hoppe et al., "Surface
    /// Reconstruction from Unorganized Points", 1992.
    ///
    /// \param k k nearest neighbour for graph reconstruction for normal
    /// propagation.
    void OrientNormalsConsistentTangentPlane(size_t k);

    /// \brief Function to compute the point to point distances between point
    /// clouds.
    ///
    /// For each point in the \p source point cloud, compute the distance to the
    /// \p target point cloud.
    ///
    /// \param target The target point cloud.
    std::vector<double> ComputePointCloudDistance(const PointCloud &target);

    /// \brief Static function to compute the covariance matrix for each point
    /// of a point cloud. Doesn't change the input PointCloud, just outputs the
    /// covariance matrices.
    ///
    ///
    /// \param input PointCloud to use for covariance computation \param
    /// search_param The KDTree search parameters for neighborhood search.
    static std::vector<Eigen::Matrix3d> EstimatePerPointCovariances(
            const PointCloud &input,
            const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());

    /// \brief Function to compute the covariance matrix for each point of a
    /// point cloud.
    ///
    ///
    /// \param search_param The KDTree search parameters for neighborhood
    /// search.
    void EstimateCovariances(
            const KDTreeSearchParam &search_param = KDTreeSearchParamKNN());

    /// Function to compute the mean and covariance matrix
    /// of a point cloud.
    std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance()
            const;

    /// \brief Function to compute the Mahalanobis distance for points
    /// in an input point cloud.
    ///
    /// See: https://en.wikipedia.org/wiki/Mahalanobis_distance
    std::vector<double> ComputeMahalanobisDistance() const;

    /// Function to compute the distance from a point to its nearest neighbor in
    /// the input point cloud
    std::vector<double> ComputeNearestNeighborDistance() const;

    /// Function that computes the convex hull of the point cloud using qhull
    /// \param joggle_inputs If true allows the algorithm to add random noise
    ///        to the points to work around degenerate inputs. This adds the
    ///        'QJ' option to the qhull command.
    /// \returns The triangle mesh of the convex hull and the list of point
    ///          indices that are part of the convex hull.
    std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
    ComputeConvexHull(bool joggle_inputs = false) const;

    /// \brief This is an implementation of the Hidden Point Removal operator
    /// described in Katz et. al. 'Direct Visibility of Point Sets', 2007.
    ///
    /// Additional information about the choice of radius
    /// for noisy point clouds can be found in Mehra et. al. 'Visibility of
    /// Noisy Point Cloud Data', 2010.
    ///
    /// \param camera_location All points not visible from that location will be
    /// removed. \param radius The radius of the spherical projection.
    std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
    HiddenPointRemoval(const Eigen::Vector3d &camera_location,
                       const double radius) const;

    /// \brief Cluster PointCloud using the DBSCAN algorithm
    /// Ester et al., "A Density-Based Algorithm for Discovering Clusters
    /// in Large Spatial Databases with Noise", 1996
    ///
    /// Returns a list of point labels, -1 indicates noise according to
    /// the algorithm.
    ///
    /// \param eps Density parameter that is used to find neighbouring points.
    /// \param min_points Minimum number of points to form a cluster.
    /// \param print_progress If `true` the progress is visualized in the
    /// console.
    std::vector<int> ClusterDBSCAN(double eps,
                                   size_t min_points,
                                   bool print_progress = false) const;

    /// \brief Segment PointCloud plane using the RANSAC algorithm.
    ///
    /// \param distance_threshold Max distance a point can be from the plane
    /// model, and still be considered an inlier.
    /// \param ransac_n Number of initial points to be considered inliers in
    /// each iteration.
    /// \param num_iterations Maximum number of iterations.
    /// \param probability Expected probability of finding the optimal plane.
    /// \return Returns the plane model ax + by + cz + d = 0 and the indices of
    /// the plane inliers.
    std::tuple<Eigen::Vector4d, std::vector<size_t>> SegmentPlane(
            const double distance_threshold = 0.01,
            const int ransac_n = 3,
            const int num_iterations = 100,
            const double probability = 0.99999999) const;

    /// \brief Factory function to create a pointcloud from a depth image and a
    /// camera model.
    ///
    /// Given depth value d at (u, v) image coordinate, the corresponding 3d
    /// point is: z = d / depth_scale\n x = (u - cx) * z / fx\n y = (v - cy) * z
    /// / fy\n
    ///
    /// \param depth The input depth image can be either a float image, or a
    /// uint16_t image. \param intrinsic Intrinsic parameters of the camera.
    /// \param extrinsic Extrinsic parameters of the camera.
    /// \param depth_scale The depth is scaled by 1 / \p depth_scale.
    /// \param depth_trunc Truncated at \p depth_trunc distance.
    /// \param stride Sampling factor to support coarse point cloud extraction.
    ///
    /// \return An empty pointcloud if the conversion fails.
    /// If \param project_valid_depth_only is true, return point cloud, which
    /// doesn't
    /// have nan point. If the value is false, return point cloud, which has
    /// a point for each pixel, whereas invalid depth results in NaN points.
    static std::shared_ptr<PointCloud> CreateFromDepthImage(
            const Image &depth,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
            double depth_scale = 1000.0,
            double depth_trunc = 1000.0,
            int stride = 1,
            bool project_valid_depth_only = true);

    /// \brief Factory function to create a pointcloud from an RGB-D image and a
    /// camera model.
    ///
    /// Given depth value d at (u, v) image coordinate, the corresponding 3d
    /// point is: z = d / depth_scale\n x = (u - cx) * z / fx\n y = (v - cy) * z
    /// / fy\n
    ///
    /// \param image The input image.
    /// \param intrinsic Intrinsic parameters of the camera.
    /// \param extrinsic Extrinsic parameters of the camera.
    ///
    /// \return An empty pointcloud if the conversion fails.
    /// If \param project_valid_depth_only is true, return point cloud, which
    /// doesn't
    /// have nan point. If the value is false, return point cloud, which has
    /// a point for each pixel, whereas invalid depth results in NaN points.
    static std::shared_ptr<PointCloud> CreateFromRGBDImage(
            const RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
            bool project_valid_depth_only = true);

    /// \brief Function to create a PointCloud from a VoxelGrid.
    ///
    /// It transforms the voxel centers to 3D points using the original point
    /// cloud coordinate (with respect to the center of the voxel grid).
    ///
    /// \param voxel_grid The input VoxelGrid.
    std::shared_ptr<PointCloud> CreateFromVoxelGrid(
            const VoxelGrid &voxel_grid);

public:
    /// Points coordinates.
    std::vector<Eigen::Vector3d> points_;
    /// Points normals.
    std::vector<Eigen::Vector3d> normals_;
    /// RGB colors of points.
    std::vector<Eigen::Vector3d> colors_;
    /// Covariance Matrix for each point
    std::vector<Eigen::Matrix3d> covariances_;
};

}  // namespace geometry
}  // namespace open3d
