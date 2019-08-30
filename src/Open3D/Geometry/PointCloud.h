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

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/KDTreeSearchParam.h"

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;
class RGBDImage;
class TriangleMesh;
class VoxelGrid;

class PointCloud : public Geometry3D {
public:
    PointCloud() : Geometry3D(Geometry::GeometryType::PointCloud) {}
    ~PointCloud() override {}

public:
    PointCloud &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const override;
    PointCloud &Transform(const Eigen::Matrix4d &transformation) override;
    PointCloud &Translate(const Eigen::Vector3d &translation,
                          bool relative = true) override;
    PointCloud &Scale(const double scale, bool center = true) override;
    PointCloud &Rotate(const Eigen::Vector3d &rotation,
                       bool center = true,
                       RotationType type = RotationType::XYZ) override;

    PointCloud &operator+=(const PointCloud &cloud);
    PointCloud operator+(const PointCloud &cloud) const;

    bool HasPoints() const { return points_.size() > 0; }

    bool HasNormals() const {
        return points_.size() > 0 && normals_.size() == points_.size();
    }

    bool HasColors() const {
        return points_.size() > 0 && colors_.size() == points_.size();
    }

    PointCloud &NormalizeNormals() {
        for (size_t i = 0; i < normals_.size(); i++) {
            normals_[i].normalize();
        }
        return *this;
    }

    /// Assigns each point in the PointCloud the same color \param color.
    PointCloud &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(colors_, points_.size(), color);
        return *this;
    }

    /// Remove all points fromt he point cloud that have a nan entry, or
    /// infinite entries.
    /// Also removes the corresponding normals and color entries.
    PointCloud &RemoveNoneFinitePoints(bool remove_nan = true,
                                       bool remove_infinite = true);

    /// Function to select points from \param input pointcloud into
    /// \return output pointcloud
    /// Points with indices in \param indices are selected.
    std::shared_ptr<PointCloud> SelectDownSample(
            const std::vector<size_t> &indices, bool invert = false) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// with a voxel \param voxel_size defines the resolution of the voxel grid,
    /// smaller value leads to denser output point cloud. Normals and colors are
    /// averaged if they exist.
    std::shared_ptr<PointCloud> VoxelDownSample(double voxel_size) const;

    /// Function to downsample using VoxelDownSample, but specialized for
    /// Surface convolution project. Experimental function.
    std::tuple<std::shared_ptr<PointCloud>, Eigen::MatrixXi>
    VoxelDownSampleAndTrace(double voxel_size,
                            const Eigen::Vector3d &min_bound,
                            const Eigen::Vector3d &max_bound,
                            bool approximate_class = false) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// uniformly \param every_k_points indicates the sample rate.
    std::shared_ptr<PointCloud> UniformDownSample(size_t every_k_points) const;

    /// Function to crop \param input pointcloud into output pointcloud
    /// All points with coordinates less than \param min_bound or larger than
    /// \param max_bound are clipped.
    std::shared_ptr<PointCloud> Crop(const Eigen::Vector3d &min_bound,
                                     const Eigen::Vector3d &max_bound) const;

    /// Function to remove points that have less than \param nb_points in a
    /// sphere of radius \param search_radius
    std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
    RemoveRadiusOutliers(size_t nb_points, double search_radius) const;

    /// Function to remove points that are further away from their
    /// \param nb_neighbor neighbors in average.
    std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
    RemoveStatisticalOutliers(size_t nb_neighbors, double std_ratio) const;

    /// Function to compute the normals of a point cloud
    /// \param cloud is the input point cloud. It also stores the output
    /// normals. Normals are oriented with respect to the input point cloud if
    /// normals exist in the input. \param search_param The KDTree search
    /// parameters
    bool EstimateNormals(
            const KDTreeSearchParam &search_param = KDTreeSearchParamKNN(),
            bool fast_normal_computation = true);

    /// Function to orient the normals of a point cloud
    /// \param cloud is the input point cloud. It must have normals.
    /// Normals are oriented with respect to \param orientation_reference
    bool OrientNormalsToAlignWithDirection(
            const Eigen::Vector3d &orientation_reference =
                    Eigen::Vector3d(0.0, 0.0, 1.0));

    /// Function to orient the normals of a point cloud
    /// \param cloud is the input point cloud. It also stores the output
    /// normals. Normals are oriented with towards \param camera_location
    bool OrientNormalsTowardsCameraLocation(
            const Eigen::Vector3d &camera_location = Eigen::Vector3d::Zero());

    /// Function to compute the point to point distances between point clouds
    /// \param source is the first point cloud.
    /// \param target is the second point cloud.
    /// \return the output distance. It has the same size as the number
    /// of points in \param source
    std::vector<double> ComputePointCloudDistance(const PointCloud &target);

    /// Function to compute the mean and covariance matrix
    /// of an \param input point cloud
    std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance()
            const;

    /// Function to compute the Mahalanobis distance for points
    /// in an \param input point cloud
    /// https://en.wikipedia.org/wiki/Mahalanobis_distance
    std::vector<double> ComputeMahalanobisDistance() const;

    /// Function to compute the distance from a point to its nearest neighbor in
    /// the \param input point cloud
    std::vector<double> ComputeNearestNeighborDistance() const;

    /// Function that computes the convex hull of the point cloud using qhull
    std::shared_ptr<TriangleMesh> ComputeConvexHull() const;

    /// Cluster PointCloud using the DBSCAN algorithm
    /// Ester et al., "A Density-Based Algorithm for Discovering Clusters
    /// in Large Spatial Databases with Noise", 1996
    /// Returns a vector of point labels, -1 indicates noise according to
    /// the algorithm.
    std::vector<int> ClusterDBSCAN(double eps,
                                   size_t min_points,
                                   bool print_progress = false) const;

    /// Factory function to create a pointcloud from a depth image and a camera
    /// model (PointCloudFactory.cpp)
    /// The input depth image can be either a float image, or a uint16_t image.
    /// In the latter case, the depth is scaled by 1 / depth_scale, and
    /// truncated at depth_trunc distance. The depth image is also sampled with
    /// stride, in order to support (fast) coarse point cloud extraction. Return
    /// an empty pointcloud if the conversion fails.
    static std::shared_ptr<PointCloud> CreateFromDepthImage(
            const Image &depth,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity(),
            double depth_scale = 1000.0,
            double depth_trunc = 1000.0,
            int stride = 1);

    /// Factory function to create a pointcloud from an RGB-D image and a camera
    /// model (PointCloudFactory.cpp)
    /// Return an empty pointcloud if the conversion fails.
    static std::shared_ptr<PointCloud> CreateFromRGBDImage(
            const RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic = Eigen::Matrix4d::Identity());

    /// Function to create a PointCloud from a VoxelGrid.
    /// It transforms the voxel centers to 3D points using the original point
    /// cloud coordinate (with respect to the center of the voxel grid).
    std::shared_ptr<PointCloud> CreateFromVoxelGrid(
            const VoxelGrid &voxel_grid);

public:
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Vector3d> normals_;
    std::vector<Eigen::Vector3d> colors_;
};

}  // namespace geometry
}  // namespace open3d
