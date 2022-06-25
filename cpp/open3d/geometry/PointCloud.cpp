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

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/ProgressBar.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace geometry {

PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    covariances_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3d PointCloud::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3d PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d PointCloud::GetCenter() const { return ComputeCenter(points_); }

AxisAlignedBoundingBox PointCloud::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
}

OrientedBoundingBox PointCloud::GetOrientedBoundingBox(bool robust) const {
    return OrientedBoundingBox::CreateFromPoints(points_, robust);
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    TransformNormals(transformation, normals_);
    TransformCovariances(transformation, covariances_);
    return *this;
}

PointCloud &PointCloud::Translate(const Eigen::Vector3d &translation,
                                  bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

PointCloud &PointCloud::Scale(const double scale,
                              const Eigen::Vector3d &center) {
    ScalePoints(scale, points_, center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Eigen::Matrix3d &R,
                               const Eigen::Vector3d &center) {
    RotatePoints(R, points_, center);
    RotateNormals(R, normals_);
    RotateCovariances(R, covariances_);
    return *this;
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            normals_[old_vert_num + i] = cloud.normals_[i];
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            colors_[old_vert_num + i] = cloud.colors_[i];
    } else {
        colors_.clear();
    }
    if ((!HasPoints() || HasCovariances()) && cloud.HasCovariances()) {
        covariances_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            covariances_[old_vert_num + i] = cloud.covariances_[i];
    } else {
        covariances_.clear();
    }
    points_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        points_[old_vert_num + i] = cloud.points_[i];
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

std::vector<double> PointCloud::ComputePointCloudDistance(
        const PointCloud &target) {
    std::vector<double> distances(points_.size());
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)points_.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        if (kdtree.SearchKNN(points_[i], 1, indices, dists) == 0) {
            utility::LogDebug(
                    "[ComputePointCloudToPointCloudDistance] Found a point "
                    "without neighbors.");
            distances[i] = 0.0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    return distances;
}

PointCloud &PointCloud::RemoveNonFinitePoints(bool remove_nan,
                                              bool remove_infinite) {
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    bool has_covariance = HasCovariances();
    size_t old_point_num = points_.size();
    size_t k = 0;                                 // new index
    for (size_t i = 0; i < old_point_num; i++) {  // old index
        bool is_nan = remove_nan &&
                      (std::isnan(points_[i](0)) || std::isnan(points_[i](1)) ||
                       std::isnan(points_[i](2)));
        bool is_infinite = remove_infinite && (std::isinf(points_[i](0)) ||
                                               std::isinf(points_[i](1)) ||
                                               std::isinf(points_[i](2)));
        if (!is_nan && !is_infinite) {
            points_[k] = points_[i];
            if (has_normal) normals_[k] = normals_[i];
            if (has_color) colors_[k] = colors_[i];
            if (has_covariance) covariances_[k] = covariances_[i];
            k++;
        }
    }
    points_.resize(k);
    if (has_normal) normals_.resize(k);
    if (has_color) colors_.resize(k);
    if (has_covariance) covariances_.resize(k);
    utility::LogDebug(
            "[RemoveNonFinitePoints] {:d} nan points have been removed.",
            (int)(old_point_num - k));
    return *this;
}

std::shared_ptr<PointCloud> PointCloud::SelectByIndex(
        const std::vector<size_t> &indices, bool invert /* = false */) const {
    auto output = std::make_shared<PointCloud>();
    bool has_normals = HasNormals();
    bool has_colors = HasColors();
    bool has_covariance = HasCovariances();

    std::vector<bool> mask = std::vector<bool>(points_.size(), invert);
    for (size_t i : indices) {
        mask[i] = !invert;
    }

    for (size_t i = 0; i < points_.size(); i++) {
        if (mask[i]) {
            output->points_.push_back(points_[i]);
            if (has_normals) output->normals_.push_back(normals_[i]);
            if (has_colors) output->colors_.push_back(colors_[i]);
            if (has_covariance) output->covariances_.push_back(covariances_[i]);
        }
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.",
            (int)points_.size(), (int)output->points_.size());
    return output;
}

// helper classes for VoxelDownSample and VoxelDownSampleAndTrace
namespace {
class AccumulatedPoint {
public:
    void AddPoint(const PointCloud &cloud, int index) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            color_ += cloud.colors_[index];
        }
        if (cloud.HasCovariances()) {
            covariance_ += cloud.covariances_[index];
        }
        num_of_points_++;
    }

    Eigen::Vector3d GetAveragePoint() const {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageNormal() const {
        // Call NormalizeNormals() afterwards if necessary
        return normal_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageColor() const {
        return color_ / double(num_of_points_);
    }

    Eigen::Matrix3d GetAverageCovariance() const {
        return covariance_ / double(num_of_points_);
    }

public:
    int num_of_points_ = 0;
    Eigen::Vector3d point_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d color_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance_ = Eigen::Matrix3d::Zero();
};

class point_cubic_id {
public:
    size_t point_id;
    int cubic_id;
};

class AccumulatedPointForTrace : public AccumulatedPoint {
public:
    void AddPoint(const PointCloud &cloud,
                  size_t index,
                  int cubic_index,
                  bool approximate_class) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            if (approximate_class) {
                auto got = classes.find(int(cloud.colors_[index][0]));
                if (got == classes.end())
                    classes[int(cloud.colors_[index][0])] = 1;
                else
                    classes[int(cloud.colors_[index][0])] += 1;
            } else {
                color_ += cloud.colors_[index];
            }
        }
        if (cloud.HasCovariances()) {
            covariance_ += cloud.covariances_[index];
        }
        point_cubic_id new_id;
        new_id.point_id = index;
        new_id.cubic_id = cubic_index;
        original_id.push_back(new_id);
        num_of_points_++;
    }

    Eigen::Vector3d GetMaxClass() {
        int max_class = -1;
        int max_count = -1;
        for (auto it = classes.begin(); it != classes.end(); it++) {
            if (it->second > max_count) {
                max_count = it->second;
                max_class = it->first;
            }
        }
        return Eigen::Vector3d(max_class, max_class, max_class);
    }

    std::vector<point_cubic_id> GetOriginalID() { return original_id; }

private:
    // original point cloud id in higher resolution + its cubic id
    std::vector<point_cubic_id> original_id;
    std::unordered_map<int, int> classes;
};
}  // namespace

std::shared_ptr<PointCloud> PointCloud::VoxelDownSample(
        double voxel_size) const {
    auto output = std::make_shared<PointCloud>();
    if (voxel_size <= 0.0) {
        utility::LogError("voxel_size <= 0.");
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogError("voxel_size is too small.");
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
                       utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)points_.size(); i++) {
        ref_coord = (points_[i] - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(*this, i);
    }
    bool has_normals = HasNormals();
    bool has_colors = HasColors();
    bool has_covariances = HasCovariances();
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            output->colors_.push_back(accpoint.second.GetAverageColor());
        }
        if (has_covariances) {
            output->covariances_.emplace_back(
                    accpoint.second.GetAverageCovariance());
        }
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.",
            (int)points_.size(), (int)output->points_.size());
    return output;
}

std::tuple<std::shared_ptr<PointCloud>,
           Eigen::MatrixXi,
           std::vector<std::vector<int>>>
PointCloud::VoxelDownSampleAndTrace(double voxel_size,
                                    const Eigen::Vector3d &min_bound,
                                    const Eigen::Vector3d &max_bound,
                                    bool approximate_class) const {
    auto output = std::make_shared<PointCloud>();
    Eigen::MatrixXi cubic_id;
    if (voxel_size <= 0.0) {
        utility::LogError("voxel_size <= 0.");
    }
    // Note: this is different from VoxelDownSample.
    // It is for fixing coordinate for multiscale voxel space
    auto voxel_min_bound = min_bound;
    auto voxel_max_bound = max_bound;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogError("voxel_size is too small.");
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPointForTrace,
                       utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    int cid_temp[3] = {1, 2, 4};
    for (size_t i = 0; i < points_.size(); i++) {
        auto ref_coord = (points_[i] - voxel_min_bound) / voxel_size;
        auto voxel_index = Eigen::Vector3i(int(floor(ref_coord(0))),
                                           int(floor(ref_coord(1))),
                                           int(floor(ref_coord(2))));
        int cid = 0;
        for (int c = 0; c < 3; c++) {
            if ((ref_coord(c) - voxel_index(c)) >= 0.5) {
                cid += cid_temp[c];
            }
        }
        voxelindex_to_accpoint[voxel_index].AddPoint(*this, i, cid,
                                                     approximate_class);
    }
    bool has_normals = HasNormals();
    bool has_colors = HasColors();
    bool has_covariances = HasCovariances();
    int cnt = 0;
    cubic_id.resize(voxelindex_to_accpoint.size(), 8);
    cubic_id.setConstant(-1);
    std::vector<std::vector<int>> original_indices(
            voxelindex_to_accpoint.size());
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            if (approximate_class) {
                output->colors_.push_back(accpoint.second.GetMaxClass());
            } else {
                output->colors_.push_back(accpoint.second.GetAverageColor());
            }
        }
        if (has_covariances) {
            output->covariances_.emplace_back(
                    accpoint.second.GetAverageCovariance());
        }
        auto original_id = accpoint.second.GetOriginalID();
        for (int i = 0; i < (int)original_id.size(); i++) {
            size_t pid = original_id[i].point_id;
            int cid = original_id[i].cubic_id;
            cubic_id(cnt, cid) = int(pid);
            original_indices[cnt].push_back(int(pid));
        }
        cnt++;
    }
    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.",
            (int)points_.size(), (int)output->points_.size());
    return std::make_tuple(output, cubic_id, original_indices);
}

std::shared_ptr<PointCloud> PointCloud::UniformDownSample(
        size_t every_k_points) const {
    if (every_k_points == 0) {
        utility::LogError("Illegal sample rate.");
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < points_.size(); i += every_k_points) {
        indices.push_back(i);
    }
    return SelectByIndex(indices);
}

std::shared_ptr<PointCloud> PointCloud::RandomDownSample(
        double sampling_ratio) const {
    if (sampling_ratio < 0 || sampling_ratio > 1) {
        utility::LogError(
                "Illegal sampling_ratio {}, sampling_ratio must be between 0 "
                "and 1.");
    }
    std::vector<size_t> indices(points_.size());
    std::iota(std::begin(indices), std::end(indices), (size_t)0);
    {
        std::lock_guard<std::mutex> lock(*utility::random::GetMutex());
        std::shuffle(indices.begin(), indices.end(),
                     *utility::random::GetEngine());
    }
    indices.resize((int)(sampling_ratio * points_.size()));
    return SelectByIndex(indices);
}

std::shared_ptr<PointCloud> PointCloud::FarthestPointDownSample(
        size_t num_samples) const {
    if (num_samples == 0) {
        return std::make_shared<PointCloud>();
    } else if (num_samples == points_.size()) {
        return std::make_shared<PointCloud>(*this);
    } else if (num_samples > points_.size()) {
        utility::LogError(
                "Illegal number of samples: {}, must <= point size: {}",
                num_samples, points_.size());
    }
    // We can also keep track of the non-selected indices with unordered_set,
    // but since typically num_samples << num_points, it may not be worth it.
    std::vector<size_t> selected_indices;
    selected_indices.reserve(num_samples);
    const size_t num_points = points_.size();
    std::vector<double> distances(num_points,
                                  std::numeric_limits<double>::infinity());
    size_t farthest_index = 0;
    for (size_t i = 0; i < num_samples; i++) {
        selected_indices.push_back(farthest_index);
        const Eigen::Vector3d &selected = points_[farthest_index];
        double max_dist = 0;
        for (size_t j = 0; j < num_points; j++) {
            double dist = (points_[j] - selected).squaredNorm();
            distances[j] = std::min(distances[j], dist);
            if (distances[j] > max_dist) {
                max_dist = distances[j];
                farthest_index = j;
            }
        }
    }
    return SelectByIndex(selected_indices);
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const AxisAlignedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "AxisAlignedBoundingBox either has zeros size, or has wrong "
                "bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}
std::shared_ptr<PointCloud> PointCloud::Crop(
        const OrientedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "AxisAlignedBoundingBox either has zeros size, or has wrong "
                "bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}

std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
PointCloud::RemoveRadiusOutliers(size_t nb_points,
                                 double search_radius,
                                 bool print_progress /* = false */) const {
    if (nb_points < 1 || search_radius <= 0) {
        utility::LogError(
                "Illegal input parameters, the number of points and radius "
                "must be positive.");
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    std::vector<bool> mask = std::vector<bool>(points_.size());
    utility::OMPProgressBar progress_bar(
            points_.size(), "Remove radius outliers: ", print_progress);
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < int(points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        size_t nb_neighbors = kdtree.SearchRadius(points_[i], search_radius,
                                                  tmp_indices, dist);
        mask[i] = (nb_neighbors > nb_points);
        ++progress_bar;
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(SelectByIndex(indices), indices);
}

std::tuple<std::shared_ptr<PointCloud>, std::vector<size_t>>
PointCloud::RemoveStatisticalOutliers(size_t nb_neighbors,
                                      double std_ratio,
                                      bool print_progress /* = false */) const {
    if (nb_neighbors < 1 || std_ratio <= 0) {
        utility::LogError(
                "Illegal input parameters, the number of neighbors and "
                "standard deviation ratio must be positive.");
    }
    if (points_.size() == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    std::vector<double> avg_distances = std::vector<double>(points_.size());
    std::vector<size_t> indices;
    size_t valid_distances = 0;
    utility::OMPProgressBar progress_bar(
            points_.size(), "Remove statistical outliers: ", print_progress);

#pragma omp parallel for reduction(+ : valid_distances) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < int(points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        kdtree.SearchKNN(points_[i], int(nb_neighbors), tmp_indices, dist);
        double mean = -1.0;
        if (dist.size() > 0u) {
            valid_distances++;
            std::for_each(dist.begin(), dist.end(),
                          [](double &d) { d = std::sqrt(d); });
            mean = std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size();
        }
        avg_distances[i] = mean;
        ++progress_bar;
    }
    if (valid_distances == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    double cloud_mean = std::accumulate(
            avg_distances.begin(), avg_distances.end(), 0.0,
            [](double const &x, double const &y) { return y > 0 ? x + y : x; });
    cloud_mean /= valid_distances;
    double sq_sum = std::inner_product(
            avg_distances.begin(), avg_distances.end(), avg_distances.begin(),
            0.0, [](double const &x, double const &y) { return x + y; },
            [cloud_mean](double const &x, double const &y) {
                return x > 0 ? (x - cloud_mean) * (y - cloud_mean) : 0;
            });
    // Bessel's correction
    double std_dev = std::sqrt(sq_sum / (valid_distances - 1));
    double distance_threshold = cloud_mean + std_ratio * std_dev;
    for (size_t i = 0; i < avg_distances.size(); i++) {
        if (avg_distances[i] > 0 && avg_distances[i] < distance_threshold) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(SelectByIndex(indices), indices);
}

std::vector<Eigen::Matrix3d> PointCloud::EstimatePerPointCovariances(
        const PointCloud &input,
        const KDTreeSearchParam &search_param /* = KDTreeSearchParamKNN()*/) {
    const auto &points = input.points_;
    std::vector<Eigen::Matrix3d> covariances;
    covariances.resize(points.size());

    KDTreeFlann kdtree;
    kdtree.SetGeometry(input);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)points.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(points[i], search_param, indices, distance2) >= 3) {
            auto covariance = utility::ComputeCovariance(points, indices);
            if (input.HasCovariances() && covariance.isIdentity(1e-4)) {
                covariances[i] = input.covariances_[i];
            } else {
                covariances[i] = covariance;
            }
        } else {
            covariances[i] = Eigen::Matrix3d::Identity();
        }
    }
    return covariances;
}
void PointCloud::EstimateCovariances(
        const KDTreeSearchParam &search_param /* = KDTreeSearchParamKNN()*/) {
    this->covariances_ = EstimatePerPointCovariances(*this, search_param);
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d>
PointCloud::ComputeMeanAndCovariance() const {
    if (IsEmpty()) {
        return std::make_tuple(Eigen::Vector3d::Zero(),
                               Eigen::Matrix3d::Identity());
    }
    std::vector<size_t> all_idx(points_.size());
    std::iota(all_idx.begin(), all_idx.end(), 0);
    return utility::ComputeMeanAndCovariance(points_, all_idx);
}

std::vector<double> PointCloud::ComputeMahalanobisDistance() const {
    std::vector<double> mahalanobis(points_.size());
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    std::tie(mean, covariance) = ComputeMeanAndCovariance();
    Eigen::Matrix3d cov_inv = covariance.inverse();
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)points_.size(); i++) {
        Eigen::Vector3d p = points_[i] - mean;
        mahalanobis[i] = std::sqrt(p.transpose() * cov_inv * p);
    }
    return mahalanobis;
}

std::vector<double> PointCloud::ComputeNearestNeighborDistance() const {
    if (points_.size() < 2) {
        return std::vector<double>(points_.size(), 0);
    }

    std::vector<double> nn_dis(points_.size());
    KDTreeFlann kdtree(*this);
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)points_.size(); i++) {
        std::vector<int> indices(2);
        std::vector<double> dists(2);
        if (kdtree.SearchKNN(points_[i], 2, indices, dists) <= 1) {
            utility::LogDebug(
                    "[ComputePointCloudNearestNeighborDistance] Found a point "
                    "without neighbors.");
            nn_dis[i] = 0.0;
        } else {
            nn_dis[i] = std::sqrt(dists[1]);
        }
    }
    return nn_dis;
}

std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
PointCloud::ComputeConvexHull(bool joggle_inputs) const {
    return Qhull::ComputeConvexHull(points_, joggle_inputs);
}

std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
PointCloud::HiddenPointRemoval(const Eigen::Vector3d &camera_location,
                               const double radius) const {
    if (radius <= 0) {
        utility::LogError("radius must be larger than zero.");
    }

    // perform spherical projection
    std::vector<Eigen::Vector3d> spherical_projection;
    for (size_t pidx = 0; pidx < points_.size(); ++pidx) {
        Eigen::Vector3d projected_point = points_[pidx] - camera_location;
        double norm = projected_point.norm();
        spherical_projection.push_back(
                projected_point + 2 * (radius - norm) * projected_point / norm);
    }

    // add origin
    size_t origin_pidx = spherical_projection.size();
    spherical_projection.push_back(Eigen::Vector3d(0, 0, 0));

    // calculate convex hull of spherical projection
    std::shared_ptr<TriangleMesh> visible_mesh;
    std::vector<size_t> pt_map;
    std::tie(visible_mesh, pt_map) =
            Qhull::ComputeConvexHull(spherical_projection);

    // reassign original points to mesh
    size_t origin_vidx = pt_map.size();
    for (size_t vidx = 0; vidx < pt_map.size(); vidx++) {
        size_t pidx = pt_map[vidx];
        if (pidx != origin_pidx) {
            visible_mesh->vertices_[vidx] = points_[pidx];
        } else {
            origin_vidx = vidx;
            visible_mesh->vertices_[vidx] = camera_location;
        }
    }

    // erase origin if part of mesh
    if (origin_vidx < visible_mesh->vertices_.size()) {
        visible_mesh->vertices_.erase(visible_mesh->vertices_.begin() +
                                      origin_vidx);
        pt_map.erase(pt_map.begin() + origin_vidx);
        for (size_t tidx = visible_mesh->triangles_.size(); tidx-- > 0;) {
            if (visible_mesh->triangles_[tidx](0) == (int)origin_vidx ||
                visible_mesh->triangles_[tidx](1) == (int)origin_vidx ||
                visible_mesh->triangles_[tidx](2) == (int)origin_vidx) {
                visible_mesh->triangles_.erase(
                        visible_mesh->triangles_.begin() + tidx);
            } else {
                if (visible_mesh->triangles_[tidx](0) > (int)origin_vidx)
                    visible_mesh->triangles_[tidx](0) -= 1;
                if (visible_mesh->triangles_[tidx](1) > (int)origin_vidx)
                    visible_mesh->triangles_[tidx](1) -= 1;
                if (visible_mesh->triangles_[tidx](2) > (int)origin_vidx)
                    visible_mesh->triangles_[tidx](2) -= 1;
            }
        }
    }
    return std::make_tuple(visible_mesh, pt_map);
}

}  // namespace geometry
}  // namespace open3d
