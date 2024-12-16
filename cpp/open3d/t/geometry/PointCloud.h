// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/DrawableGeometry.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {

class LineSet;

/// \class PointCloud
/// \brief A point cloud contains a list of 3D points.
///
/// The point cloud class stores the attribute data in key-value maps, where
/// the key is a string representing the attribute name and the value is a
/// Tensor containing the attribute data. In most cases, the length of an
/// attribute should be equal to the length of the point cloud's "positions".
///
/// - Default attribute: "positions".
///     - Usage
///         - PointCloud::GetPointPositions()
///         - PointCloud::SetPointPositions(const Tensor& positions)
///         - PointCloud::HasPointPositions()
///     - Created by default, required for all pointclouds.
///     - Value tensor must have shape {N, 3}.
///     - The device of "positions" determines the device of the point cloud.
///
/// - Common attributes: "normals", "colors".
///     - Usage
///         - PointCloud::GetPointNormals()
///         - PointCloud::SetPointNormals(const Tensor& normals)
///         - PointCloud::HasPointNormals()
///         - PointCloud::GetPointColors()
///         - PointCloud::SetPointColors(const Tensor& colors)
///         - PointCloud::HasPointColors()
///     - Not created by default.
///     - Value tensor must have shape {N, 3}.
///     - Value tensor must be on the same device as the point cloud.
///     - Value tensor can have any dtype.
///
/// - Custom attributes, e.g., "labels", "intensities".
///     - Usage
///         - PointCloud::GetPointAttr(const std::string& key)
///         - PointCloud::SetPointAttr(const std::string& key,
///                                    const Tensor& value)
///         - PointCloud::HasPointAttr(const std::string& key)
///     - Not created by default. Users can add their own custom attributes.
///     - Value tensor must be on the same device as the point cloud.
///     - Value tensor can have any dtype.
///
/// PointCloud::GetPointAttr(), PointCloud::SetPointAttr(),
/// PointCloud::HasPointAttr() also work for default attribute "position" and
/// common attributes "normals" and "colors", e.g.,
///     - PointCloud::GetPointPositions() is the same as
///       PointCloud::GetPointAttr("positions")
///     - PointCloud::HasPointNormals() is the same as
///       PointCloud::HasPointAttr("normals")
class PointCloud : public Geometry, public DrawableGeometry {
public:
    /// Construct an empty point cloud on the provided device.
    /// \param device The device on which to initialize the point cloud
    /// (default: 'CPU:0').
    PointCloud(const core::Device &device = core::Device("CPU:0"));

    /// Construct a point cloud from points.
    ///
    /// The input tensor will be directly used as the underlying storage of the
    /// point cloud (no memory copy).
    ///
    /// \param points A tensor with element shape {3}.
    PointCloud(const core::Tensor &points);

    /// Construct from points and other attributes of the points.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor containing
    /// points and their attributes. point_dict must contain at least the
    /// "positions" key.
    PointCloud(const std::unordered_map<std::string, core::Tensor>
                       &map_keys_to_tensors);

    virtual ~PointCloud() override {}

    /// \brief Text description.
    std::string ToString() const;

    /// Getter for point_attr_ TensorMap. Used in Pybind.
    const TensorMap &GetPointAttr() const { return point_attr_; }

    /// Getter for point_attr_ TensorMap.
    TensorMap &GetPointAttr() { return point_attr_; }

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    core::Tensor &GetPointAttr(const std::string &key) {
        return point_attr_.at(key);
    }

    /// Get the value of the "positions" attribute. Convenience function.
    core::Tensor &GetPointPositions() { return GetPointAttr("positions"); }

    /// Get the value of the "colors" attribute. Convenience function.
    core::Tensor &GetPointColors() { return GetPointAttr("colors"); }

    /// Get the value of the "normals" attribute. Convenience function.
    core::Tensor &GetPointNormals() { return GetPointAttr("normals"); }

    /// Get attributes. Throws exception if the attribute does not exist.
    ///
    /// \param key Attribute name.
    const core::Tensor &GetPointAttr(const std::string &key) const {
        return point_attr_.at(key);
    }

    /// Get the value of the "positions" attribute. Convenience function.
    const core::Tensor &GetPointPositions() const {
        return GetPointAttr("positions");
    }

    /// Get the value of the "colors" attribute. Convenience function.
    const core::Tensor &GetPointColors() const {
        return GetPointAttr("colors");
    }

    /// Get the value of the "normals" attribute. Convenience function.
    const core::Tensor &GetPointNormals() const {
        return GetPointAttr("normals");
    }

    /// Set attributes. If the attribute key already exists, its value
    /// will be overwritten, otherwise, the new key will be created.
    ///
    /// \param key Attribute name.
    /// \param value A tensor.
    void SetPointAttr(const std::string &key, const core::Tensor &value) {
        if (value.GetDevice() != device_) {
            utility::LogError("Attribute device {} != Pointcloud's device {}.",
                              value.GetDevice().ToString(), device_.ToString());
        }
        point_attr_[key] = value;
    }

    /// Set the value of the "positions" attribute. Convenience function.
    void SetPointPositions(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetPointAttr("positions", value);
    }

    /// Set the value of the "colors" attribute. Convenience function.
    void SetPointColors(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetPointAttr("colors", value);
    }

    /// Set the value of the "normals" attribute. Convenience function.
    void SetPointNormals(const core::Tensor &value) {
        core::AssertTensorShape(value, {utility::nullopt, 3});
        SetPointAttr("normals", value);
    }

    /// Returns true if all of the following are true:
    /// 1) attribute key exist
    /// 2) attribute's length as points' length
    /// 3) attribute's length > 0
    bool HasPointAttr(const std::string &key) const {
        return point_attr_.Contains(key) && GetPointAttr(key).GetLength() > 0 &&
               GetPointAttr(key).GetLength() == GetPointPositions().GetLength();
    }

    /// Removes point attribute by key value. Primary attribute "positions"
    /// cannot be removed. Throws warning if attribute key does not exists.
    ///
    /// \param key Attribute name.
    void RemovePointAttr(const std::string &key) { point_attr_.Erase(key); }

    /// Check if the "positions" attribute's value has length > 0.
    /// This is a convenience function.
    bool HasPointPositions() const { return HasPointAttr("positions"); }

    /// Returns true if all of the following are true:
    /// 1) attribute "colors" exist
    /// 2) attribute "colors"'s length as points' length
    /// 3) attribute "colors"'s length > 0
    /// This is a convenience function.
    bool HasPointColors() const { return HasPointAttr("colors"); }

    /// Returns true if all of the following are true:
    /// 1) attribute "normals" exist
    /// 2) attribute "normals"'s length as points' length
    /// 3) attribute "normals"'s length > 0
    /// This is a convenience function.
    bool HasPointNormals() const { return HasPointAttr("normals"); }

public:
    /// Transfer the point cloud to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new point cloud is always created; if false, the
    /// copy is avoided when the original point cloud is already on the targeted
    /// device.
    PointCloud To(const core::Device &device, bool copy = false) const;

    /// Returns copy of the point cloud on the same device.
    PointCloud Clone() const;

    /// Clear all data in the point cloud.
    PointCloud &Clear() override {
        point_attr_.clear();
        return *this;
    }

    /// Returns !HasPointPositions().
    bool IsEmpty() const override { return !HasPointPositions(); }

    /// Returns the min bound for point coordinates.
    core::Tensor GetMinBound() const;

    /// Returns the max bound for point coordinates.
    core::Tensor GetMaxBound() const;

    /// Returns the center for point coordinates.
    core::Tensor GetCenter() const;

    /// Append a point cloud and returns the resulting point cloud.
    ///
    /// The point cloud being appended, must have all the attributes
    /// present in the point cloud it is being appended to, with same
    /// dtype, device and same shape other than the first dimension / length.
    PointCloud Append(const PointCloud &other) const;

    /// operator+ for t::PointCloud appends the compatible attributes to the
    /// point cloud.
    PointCloud operator+(const PointCloud &other) const {
        return Append(other);
    }

    /// \brief Transforms the PointPositions and PointNormals (if exist)
    /// of the PointCloud.
    ///
    /// Transformation matrix is a 4x4 matrix.
    ///  T (4x4) =   [[ R(3x3)  t(3x1) ],
    ///               [ O(1x3)  s(1x1) ]]
    ///  (s = 1 for Transformation without scaling)
    ///
    ///  It applies the following general transform to each `positions` and
    ///  `normals`.
    ///   |x'|   | R(0,0) R(0,1) R(0,2) t(0)|   |x|
    ///   |y'| = | R(1,0) R(1,1) R(1,2) t(1)| @ |y|
    ///   |z'|   | R(2,0) R(2,1) R(2,2) t(2)|   |z|
    ///   |w'|   | O(0,0) O(0,1) O(0,2)  s  |   |1|
    ///
    ///   [x, y, z] = [x', y', z'] / w'
    ///
    /// \param transformation Transformation [Tensor of dim {4,4}].
    /// \return Transformed point cloud
    PointCloud &Transform(const core::Tensor &transformation);

    /// \brief Translates the PointPositions of the PointCloud.
    /// \param translation translation tensor of dimension {3}
    /// Should be on the same device as the PointCloud
    /// \param relative if true (default): translates relative to Center
    /// \return Translated point cloud
    PointCloud &Translate(const core::Tensor &translation,
                          bool relative = true);

    /// \brief Scales the PointPositions of the PointCloud.
    /// \param scale Scale [double] of dimension
    /// \param center Center [Tensor of dim {3}] about which the PointCloud is
    /// to be scaled. Should be on the same device as the PointCloud
    /// \return Scaled point cloud
    PointCloud &Scale(double scale, const core::Tensor &center);

    /// \brief Rotates the PointPositions and PointNormals (if exists).
    /// \param R Rotation [Tensor of dim {3,3}].
    /// Should be on the same device as the PointCloud
    /// \param center Center [Tensor of dim {3}] about which the PointCloud is
    /// to be scaled. Should be on the same device as the PointCloud
    /// \return Rotated point cloud
    PointCloud &Rotate(const core::Tensor &R, const core::Tensor &center);

    /// \brief Assigns uniform color to the point cloud.
    ///
    /// \param color  RGB color for the point cloud. {3,} shaped Tensor.
    /// Floating color values are clipped between 0.0 and 1.0.
    PointCloud &PaintUniformColor(const core::Tensor &color);

    /// \brief Select points from input pointcloud, based on boolean mask
    /// indices into output point cloud.
    ///
    /// \param boolean_mask Boolean indexing tensor of shape {n,} containing
    /// true value for the indices that is to be selected.
    /// \param invert Set to `True` to invert the selection of indices.
    PointCloud SelectByMask(const core::Tensor &boolean_mask,
                            bool invert = false) const;

    /// \brief Select points from input pointcloud, based on indices list into
    /// output point cloud.
    ///
    /// \param indices Int64 indexing tensor of shape {n,} containing
    /// index value that is to be selected.
    /// \param invert Set to `True` to invert the selection of indices, and also
    /// ignore the duplicated indices.
    /// \param remove_duplicates Set to `True` to remove the duplicated indices.
    PointCloud SelectByIndex(const core::Tensor &indices,
                             bool invert = false,
                             bool remove_duplicates = false) const;

    /// \brief Downsamples a point cloud with a specified voxel size.
    ///
    /// \param voxel_size Voxel size. A positive number.
    /// \param reduction Reduction type. Currently only support "mean".
    PointCloud VoxelDownSample(double voxel_size,
                               const std::string &reduction = "mean") const;

    /// \brief Downsamples a point cloud by selecting every kth index point and
    /// its attributes.
    ///
    /// \param every_k_points Sample rate, the selected point indices are [0, k,
    /// 2k, …].
    PointCloud UniformDownSample(size_t every_k_points) const;

    /// \brief Downsample a pointcloud by selecting random index point and its
    /// attributes.
    ///
    /// \param sampling_ratio Sampling ratio, the ratio of sample to total
    /// number of points in the pointcloud.
    PointCloud RandomDownSample(double sampling_ratio) const;

    /// \brief Downsample a pointcloud into output pointcloud with a set of
    /// points has farthest distance.
    ///
    /// The sampling is performed by selecting the farthest point from previous
    /// selected points iteratively, starting from `start_index`.
    ///
    /// \param num_samples Number of points to be sampled.
    /// \param start_index Index to start downsampling from.
    PointCloud FarthestPointDownSample(const size_t num_samples,
                                       const size_t start_index = 0) const;

    /// \brief Remove points that have less than \p nb_points neighbors in a
    /// sphere of a given radius.
    ///
    /// \param nb_points Number of neighbor points required within the radius.
    /// \param search_radius Radius of the sphere.
    /// \return Tuple of filtered point cloud and boolean mask tensor for
    /// selected values w.r.t. input point cloud.
    std::tuple<PointCloud, core::Tensor> RemoveRadiusOutliers(
            size_t nb_points, double search_radius) const;

    /// \brief Remove points that are further away from their \p nb_neighbor
    /// neighbors in average. This function is not recommended to use on GPU.
    ///
    /// \param nb_neighbors Number of neighbors around the target point.
    /// \param std_ratio Standard deviation ratio.
    /// \return Tuple of filtered point cloud and boolean mask tensor for
    /// selected values w.r.t. input point cloud.
    std::tuple<PointCloud, core::Tensor> RemoveStatisticalOutliers(
            size_t nb_neighbors, double std_ratio) const;

    /// \brief Remove duplicated points and there associated attributes.
    ///
    /// \return Tuple of filtered PointCloud and boolean indexing tensor w.r.t.
    /// input point cloud.
    std::tuple<PointCloud, core::Tensor> RemoveDuplicatedPoints() const;

    /// \brief Remove all points from the point cloud that have a nan entry, or
    /// infinite value. It also removes the corresponding attributes.
    ///
    /// \param remove_nan Remove NaN values from the PointCloud.
    /// \param remove_infinite Remove infinite values from the PointCloud.
    /// \return Tuple of filtered point cloud and boolean mask tensor for
    /// selected values w.r.t. input point cloud.
    std::tuple<PointCloud, core::Tensor> RemoveNonFinitePoints(
            bool remove_nan = true, bool remove_infinite = true) const;

    /// \brief Returns the device attribute of this PointCloud.
    core::Device GetDevice() const override { return device_; }

    /// \brief This is an implementation of the Hidden Point Removal operator
    /// described in Katz et. al. 'Direct Visibility of Point Sets', 2007.
    ///
    /// Additional information about the choice of radius
    /// for noisy point clouds can be found in Mehra et. al. 'Visibility of
    /// Noisy Point Cloud Data', 2010.
    ///
    /// This is a wrapper for a CPU implementation and a copy of the point cloud
    /// data and resulting visible triangle mesh and indiecs will be made.
    ///
    /// \param camera_location All points not visible from that location will be
    /// removed.
    /// \param radius The radius of the spherical projection.
    /// \return Tuple of visible triangle mesh and indices of visible points on
    /// the same device as the point cloud.
    std::tuple<TriangleMesh, core::Tensor> HiddenPointRemoval(
            const core::Tensor &camera_location, double radius) const;

    /// \brief Cluster PointCloud using the DBSCAN algorithm
    /// Ester et al., "A Density-Based Algorithm for Discovering Clusters
    /// in Large Spatial Databases with Noise", 1996
    /// This is a wrapper for a CPU implementation and a copy of the point cloud
    /// data and resulting labels will be made.
    ///
    /// \param eps Density parameter that is used to find neighbouring points.
    /// \param min_points Minimum number of points to form a cluster.
    /// \param print_progress If `true` the progress is visualized in the
    /// console.
    /// \return A Tensor list of point labels on the same device as the point
    /// cloud, -1 indicates noise according to the algorithm.
    core::Tensor ClusterDBSCAN(double eps,
                               size_t min_points,
                               bool print_progress = false) const;

    /// \brief Segment PointCloud plane using the RANSAC algorithm.
    /// This is a wrapper for a CPU implementation and a copy of the point cloud
    /// data and resulting plane model and inlier indiecs will be made.
    ///
    /// \param distance_threshold Max distance a point can be from the plane
    /// model, and still be considered an inlier.
    /// \param ransac_n Number of initial points to be considered inliers in
    /// each iteration.
    /// \param num_iterations Maximum number of iterations.
    /// \param probability Expected probability of finding the optimal plane.
    /// \return Tuple of the plane model ax + by + cz + d = 0 and the indices of
    /// the plane inliers on the same device as the point cloud.
    std::tuple<core::Tensor, core::Tensor> SegmentPlane(
            const double distance_threshold = 0.01,
            const int ransac_n = 3,
            const int num_iterations = 100,
            const double probability = 0.99999999) const;

    /// Compute the convex hull of a point cloud using qhull.
    ///
    /// This runs on the CPU.
    ///
    /// \param joggle_inputs (default False). Handle precision problems by
    /// randomly perturbing the input data. Set to True if perturbing the input
    /// iis acceptable but you need convex simplicial output. If False,
    /// neighboring facets may be merged in case of precision problems. See
    /// [QHull docs](http://www.qhull.org/html/qh-impre.htm#joggle) for more
    /// details.
    ///
    /// \return TriangleMesh representing the convexh hull. This contains an
    /// extra vertex property "point_map" that contains the index of the
    /// corresponding vertex in the original mesh.
    TriangleMesh ComputeConvexHull(bool joggle_inputs = false) const;

    /// \brief Compute the boundary points of a point cloud.
    /// The implementation is inspired by the PCL implementation. Reference:
    /// https://pointclouds.org/documentation/classpcl_1_1_boundary_estimation.html
    ///
    /// \param radius Neighbor search radius parameter.
    /// \param max_nn Neighbor search max neighbors parameter
    /// [Default = 30].
    /// \param angle_threshold Angle threshold to decide if a point is on the
    /// boundary [Default = 90.0].
    /// \return Tensor of boundary points and its boolean mask tensor.
    std::tuple<PointCloud, core::Tensor> ComputeBoundaryPoints(
            double radius,
            int max_nn = 30,
            double angle_threshold = 90.0) const;

public:
    /// Normalize point normals to length 1.
    PointCloud &NormalizeNormals();

    /// \brief Function to estimate point normals. If the point cloud normals
    /// exist, the estimated normals are oriented with respect to the same.
    /// It uses KNN search (Not recommended to use on GPU) if only max_nn
    /// parameter is provided, Radius search (Not recommended to use on GPU) if
    /// only radius is provided and Hybrid Search (Recommended) if radius
    /// parameter is also provided.
    ///
    /// \param max_nn [optional] Neighbor search max neighbors parameter
    /// [Default = 30].
    /// \param radius [optional] Neighbor search radius parameter. [Recommended
    /// ~1.4x voxel size].
    void EstimateNormals(
            const utility::optional<int> max_nn = 30,
            const utility::optional<double> radius = utility::nullopt);

    /// \brief Function to orient the normals of a point cloud.
    ///
    /// \param orientation_reference Normals are oriented with respect to
    /// orientation_reference.
    void OrientNormalsToAlignWithDirection(
            const core::Tensor &orientation_reference =
                    core::Tensor::Init<float>({0, 0, 1},
                                              core::Device("CPU:0")));

    /// \brief Function to orient the normals of a point cloud.
    ///
    /// \param camera_location Normals are oriented with towards the
    /// camera_location.
    void OrientNormalsTowardsCameraLocation(
            const core::Tensor &camera_location = core::Tensor::Zeros(
                    {3}, core::Float32, core::Device("CPU:0")));

    /// \brief Function to consistently orient estimated normals based on
    /// consistent tangent planes as described in Hoppe et al., "Surface
    /// Reconstruction from Unorganized Points", 1992.
    /// Further details on parameters are described in
    /// Piazza, Valentini, Varetti, "Mesh Reconstruction from Point Cloud",
    /// 2023.
    ///
    /// \param k k nearest neighbour for graph reconstruction for normal
    /// propagation.
    /// \param lambda penalty constant on the distance of a point from the
    /// tangent plane \param cos_alpha_tol treshold that defines the amplitude
    /// of the cone spanned by the reference normal
    void OrientNormalsConsistentTangentPlane(size_t k,
                                             const double lambda = 0.0,
                                             const double cos_alpha_tol = 1.0);

    /// \brief Function to compute point color gradients. If radius is provided,
    /// then HybridSearch is used, otherwise KNN-Search is used.
    /// Reference: Park, Q.-Y. Zhou, and V. Koltun,
    /// Colored Point Cloud Registration Revisited, ICCV, 2017.
    /// It uses KNN search (Not recommended to use on GPU) if only max_nn
    /// parameter is provided, Radius search (Not recommended to use on GPU) if
    /// only radius is provided and Hybrid Search (Recommended) if radius
    /// parameter is also provided.
    ///
    /// \param max_nn [optional] Neighbor search max neighbors parameter
    /// [Default = 30].
    /// \param radius [optional] Neighbor search radius parameter to use
    /// HybridSearch. [Recommended ~1.4x voxel size].
    void EstimateColorGradients(
            const utility::optional<int> max_nn = 30,
            const utility::optional<double> radius = utility::nullopt);

public:
    /// \brief Factory function to create a point cloud from a depth image and a
    /// camera model.
    ///
    /// Given depth value d at (u, v) image coordinate, the corresponding 3d
    /// point is:
    /// - z = d / depth_scale
    /// - x = (u - cx) * z / fx
    /// - y = (v - cy) * z / fy
    ///
    /// \param depth The input depth image should be a uint16_t or float image.
    /// \param intrinsics Intrinsic parameters of the camera.
    /// \param extrinsics Extrinsic parameters of the camera.
    /// \param depth_scale The depth is scaled by 1 / \p depth_scale.
    /// \param depth_max Truncated at \p depth_max distance.
    /// \param stride Sampling factor to support coarse point cloud extraction.
    /// Unless \p with_normals=true, there is no low pass filtering, so aliasing
    /// is possible for \p stride>1.
    /// \param with_normals Also compute normals for the point cloud. If
    /// True, the point cloud will only contain points with valid normals. If
    /// normals are requested, the depth map is first filtered to ensure smooth
    /// normals.
    ///
    /// \return Created point cloud with the 'points' property set. Thus is
    /// empty if the conversion fails.
    static PointCloud CreateFromDepthImage(
            const Image &depth,
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            float depth_scale = 1000.0f,
            float depth_max = 3.0f,
            int stride = 1,
            bool with_normals = false);

    /// \brief Factory function to create a point cloud from an RGB-D image and
    /// a camera model.
    ///
    /// Given depth value d at (u, v) image coordinate, the corresponding 3d
    /// point is:
    /// - z = d / depth_scale
    /// - x = (u - cx) * z / fx
    /// - y = (v - cy) * z / fy
    ///
    /// \param rgbd_image The input RGBD image should have a uint16_t or float
    /// depth image and RGB image with any DType and the same size.
    /// \param intrinsics Intrinsic parameters of the camera.
    /// \param extrinsics Extrinsic parameters of the camera.
    /// \param depth_scale The depth is scaled by 1 / \p depth_scale.
    /// \param depth_max Truncated at \p depth_max distance.
    /// \param stride Sampling factor to support coarse point cloud extraction.
    /// Unless \p with_normals=true, there is no low pass filtering, so aliasing
    /// is possible for \p stride>1.
    /// \param with_normals Also compute normals for the point cloud. If True,
    /// the point cloud will only contain points with valid normals. If
    /// normals are requested, the depth map is first filtered to ensure smooth
    /// normals.
    ///
    /// \return Created point cloud with the 'points' and 'colors' properties
    /// set. This is empty if the conversion fails.
    static PointCloud CreateFromRGBDImage(
            const RGBDImage &rgbd_image,
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            float depth_scale = 1000.0f,
            float depth_max = 3.0f,
            int stride = 1,
            bool with_normals = false);

    /// Create a PointCloud from a legacy Open3D PointCloud.
    static PointCloud FromLegacy(
            const open3d::geometry::PointCloud &pcd_legacy,
            core::Dtype dtype = core::Float32,
            const core::Device &device = core::Device("CPU:0"));

    /// Convert to a legacy Open3D PointCloud.
    open3d::geometry::PointCloud ToLegacy() const;

    /// Project a point cloud to a depth image.
    geometry::Image ProjectToDepthImage(
            int width,
            int height,
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            float depth_scale = 1000.0f,
            float depth_max = 3.0f);

    /// Project a point cloud to an RGBD image.
    geometry::RGBDImage ProjectToRGBDImage(
            int width,
            int height,
            const core::Tensor &intrinsics,
            const core::Tensor &extrinsics =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            float depth_scale = 1000.0f,
            float depth_max = 3.0f);

    /// Create an axis-aligned bounding box from attribute "positions".
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const;

    /// Create an oriented bounding box from attribute "positions".
    OrientedBoundingBox GetOrientedBoundingBox() const;

    /// \brief Function to crop pointcloud into output pointcloud.
    ///
    /// \param aabb AxisAlignedBoundingBox to crop points.
    /// \param invert Crop the points outside of the bounding box or inside of
    /// the bounding box.
    PointCloud Crop(const AxisAlignedBoundingBox &aabb,
                    bool invert = false) const;

    /// \brief Function to crop pointcloud into output pointcloud.
    ///
    /// \param obb OrientedBoundingBox to crop points.
    /// \param invert Crop the points outside of the bounding box or inside of
    /// the bounding box.
    PointCloud Crop(const OrientedBoundingBox &obb, bool invert = false) const;

    /// Sweeps the point cloud rotationally about an axis.
    /// \param angle The rotation angle in degree.
    /// \param axis The rotation axis.
    /// \param resolution The resolution defines the number of intermediate
    /// sweeps about the rotation axis.
    /// \param translation The translation along the rotation axis.
    /// \param capping If true adds caps to the mesh.
    /// \return A line set with the result of the sweep operation.
    LineSet ExtrudeRotation(double angle,
                            const core::Tensor &axis,
                            int resolution = 16,
                            double translation = 0.0,
                            bool capping = true) const;

    /// Sweeps the point cloud along a direction vector.
    /// \param vector The direction vector.
    /// \param scale Scalar factor which essentially scales the direction
    /// vector. \param capping If true adds caps to the mesh. \return A line set
    /// with the result of the sweep operation.
    LineSet ExtrudeLinear(const core::Tensor &vector,
                          double scale = 1.0,
                          bool capping = true) const;

    /// Partition the point cloud by recursively doing PCA.
    /// This function creates a new point attribute with the name
    /// "partition_ids".
    /// \param max_points The maximum allowed number of points in a partition.
    /// \return The number of partitions.
    int PCAPartition(int max_points);

    /// Compute various metrics between two point clouds. Currently, Chamfer
    /// distance, Hausdorff distance and F-Score
    /// <a href="../tutorial/reference.html#Knapitsch2017">[[Knapitsch2017]]</a>
    /// are supported. The Chamfer distance is the sum of the mean distance to
    /// the nearest neighbor from the points of the first point cloud to the
    /// second point cloud. The F-Score at a fixed threshold radius is the
    /// harmonic mean of the Precision and Recall. Recall is the percentage of
    /// surface points from the first point cloud that have the second point
    /// cloud points within the threshold radius, while Precision is the
    /// percentage of points from the second point cloud that have the first
    /// point cloud points within the threhold radius.

    /// \f{eqnarray*}{
    ///   \text{Chamfer Distance: } d_{CD}(X,Y) &=& \frac{1}{|X|}\sum_{i \in X}
    ///   || x_i - n(x_i, Y) || + \frac{1}{|Y|}\sum_{i \in Y} || y_i - n(y_i, X)
    ///   || \\{}
    ///   \text{Hausdorff distance: } d_H(X,Y) &=& \max \left\{ \max_{i \in X}
    ///   || x_i - n(x_i, Y) ||, \max_{i \in Y} || y_i - n(y_i, X) || \right\}
    ///   \\{}
    ///   \text{Precision: } P(X,Y|d) &=& \frac{100}{|X|} \sum_{i \in X} || x_i
    ///   - n(x_i, Y) || < d \\{}
    ///   \text{Recall: } R(X,Y|d) &=& \frac{100}{|Y|} \sum_{i \in Y} || y_i -
    ///   n(y_i, X) || < d \\{}
    ///   \text{F-Score: } F(X,Y|d) &=& \frac{2 P(X,Y|d) R(X,Y|d)}{P(X,Y|d) +
    ///   R(X,Y|d)}
    /// \f}

    /// \param pcd2 Other point cloud to compare with.
    /// \param metrics List of Metric s to compute.  Multiple metrics can be
    /// computed at once for efficiency.
    /// \param params MetricParameters struct holds parameters required by
    /// different metrics.
    /// \returns Tensor containing the requested metrics.
    core::Tensor ComputeMetrics(
            const PointCloud &pcd2,
            std::vector<Metric> metrics = {Metric::ChamferDistance},
            MetricParameters params = MetricParameters()) const;

protected:
    core::Device device_ = core::Device("CPU:0");
    TensorMap point_attr_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
