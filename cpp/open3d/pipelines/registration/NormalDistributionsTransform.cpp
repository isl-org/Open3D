// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/NormalDistributionsTransform.h"

// This implements a Gauss-Newton point-to-distribution NDT variant based on
// the 3D formulation described in
// https://github.com/gaoxiang12/slam_in_autonomous_driving/blob/master/src/ch7/ndt_3d.cc.
// It models target voxels as regularized Gaussians, rejects outliers by their
// Mahalanobis distance, and applies left-perturbation SE(3) updates.

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace pipelines {
namespace registration {

namespace {

struct VoxelKey {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    bool operator==(const VoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey &key) const {
        size_t seed = 0;
        utility::hash_combine(seed, key.x);
        utility::hash_combine(seed, key.y);
        utility::hash_combine(seed, key.z);
        return seed;
    }
};

struct VoxelGaussian {
    int count = 0;
    int representative_index = -1;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
};

using VoxelMap = std::unordered_map<VoxelKey, VoxelGaussian, VoxelKeyHash>;

struct VoxelAccumulator {
    int count = 0;
    std::vector<int> point_indices;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance_accumulator = Eigen::Matrix3d::Zero();

    void AddPoint(const Eigen::Vector3d &point, int point_index) {
        ++count;
        point_indices.push_back(point_index);
        const Eigen::Vector3d delta = point - mean;
        mean += delta / static_cast<double>(count);
        const Eigen::Vector3d delta_after_update = point - mean;
        covariance_accumulator += delta * delta_after_update.transpose();
    }
};

using VoxelAccumulatorMap =
        std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash>;

struct NDTLinearSystem {
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
    double residual2 = 0.0;
    int residual_count = 0;
    double euclidean_error2 = 0.0;
    int correspondence_count = 0;

    double MeanObjective() const {
        return residual2 / static_cast<double>(residual_count);
    }

    double Fitness(std::size_t source_size) const {
        return static_cast<double>(correspondence_count) /
               static_cast<double>(source_size);
    }

    double InlierRMSE() const {
        return correspondence_count == 0
                       ? 0.0
                       : std::sqrt(euclidean_error2 /
                                   static_cast<double>(correspondence_count));
    }
};

VoxelKey GetVoxelKey(const Eigen::Vector3d &point, double inv_voxel_size) {
    if (!point.allFinite()) {
        utility::LogError("Point coordinates must be finite.");
    }
    const Eigen::Vector3d scaled = point * inv_voxel_size;
    if (!scaled.allFinite()) {
        utility::LogError("Scaled point coordinates must be finite.");
    }
    const Eigen::Vector3d rounded = scaled.array().round();
    if (!rounded.allFinite()) {
        utility::LogError("Rounded voxel coordinates must be finite.");
    }

    const double min_key = std::nextafter(
            static_cast<double>(std::numeric_limits<std::int64_t>::min()),
            std::numeric_limits<double>::infinity());
    const double max_key = std::nextafter(
            static_cast<double>(std::numeric_limits<std::int64_t>::max()),
            -std::numeric_limits<double>::infinity());
    if ((rounded.array() < min_key).any() ||
        (rounded.array() > max_key).any()) {
        utility::LogError("Voxel coordinates exceed the supported range.");
    }

    return VoxelKey{static_cast<std::int64_t>(rounded.x()),
                    static_cast<std::int64_t>(rounded.y()),
                    static_cast<std::int64_t>(rounded.z())};
}

using NeighborOffsets = std::array<VoxelKey, 7>;

NeighborOffsets GetNeighborOffsets(int neighbor_search_type,
                                   std::size_t &offset_count) {
    NeighborOffsets offsets{{{0, 0, 0},
                             {-1, 0, 0},
                             {1, 0, 0},
                             {0, -1, 0},
                             {0, 1, 0},
                             {0, 0, -1},
                             {0, 0, 1}}};
    offset_count = 1;
    if (neighbor_search_type == 1) {
        offset_count = offsets.size();
    }
    return offsets;
}

void ValidateNDTOption(const NormalDistributionsTransformOption &option) {
    if (!std::isfinite(option.voxel_size_) || option.voxel_size_ <= 0.0) {
        utility::LogError("voxel_size must be positive.");
    }
    if (option.min_points_per_voxel_ < 4) {
        utility::LogError("min_points_per_voxel must be at least 4.");
    }
    if (!std::isfinite(option.covariance_regularization_) ||
        option.covariance_regularization_ <= 0.0 ||
        option.covariance_regularization_ >= 1.0) {
        utility::LogError(
                "covariance_regularization must be in the range (0, 1).");
    }
    if (!std::isfinite(option.transformation_epsilon_) ||
        option.transformation_epsilon_ <= 0.0) {
        utility::LogError("transformation_epsilon must be positive.");
    }
    if (!std::isfinite(option.relative_objective_) ||
        option.relative_objective_ <= 0.0) {
        utility::LogError("relative_objective must be positive.");
    }
    if (option.max_iteration_ <= 0) {
        utility::LogError("max_iteration must be positive.");
    }
    if (!std::isfinite(option.outlier_threshold_) ||
        option.outlier_threshold_ <= 0.0) {
        utility::LogError("outlier_threshold must be positive.");
    }
    if (option.neighbor_search_type_ != 0 &&
        option.neighbor_search_type_ != 1) {
        utility::LogError("neighbor_search_type must be 0 or 1.");
    }
}

VoxelMap BuildVoxelGaussians(const geometry::PointCloud &target,
                             const NormalDistributionsTransformOption &option) {
    const double inv_voxel_size = 1.0 / option.voxel_size_;
    VoxelAccumulatorMap voxel_accumulators;
    for (int i = 0; i < static_cast<int>(target.points_.size()); ++i) {
        voxel_accumulators[GetVoxelKey(target.points_[i], inv_voxel_size)]
                .AddPoint(target.points_[i], i);
    }

    VoxelMap voxel_map;
    for (const auto &item : voxel_accumulators) {
        const VoxelAccumulator &accumulator = item.second;
        if (accumulator.count < option.min_points_per_voxel_) {
            continue;
        }

        VoxelGaussian gaussian;
        gaussian.count = accumulator.count;
        gaussian.mean = accumulator.mean;
        const Eigen::Matrix3d covariance =
                accumulator.covariance_accumulator /
                static_cast<double>(accumulator.count - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success) {
            continue;
        }
        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        const double max_eigenvalue = eigenvalues.maxCoeff();
        if (max_eigenvalue <= 0.0) {
            continue;
        }
        const double min_eigenvalue =
                max_eigenvalue * option.covariance_regularization_;
        for (int i = 0; i < 3; ++i) {
            eigenvalues[i] = std::max(eigenvalues[i], min_eigenvalue);
        }
        gaussian.information = solver.eigenvectors() *
                               eigenvalues.cwiseInverse().asDiagonal() *
                               solver.eigenvectors().transpose();
        for (const int point_index : accumulator.point_indices) {
            const double distance2 =
                    (target.points_[point_index] - gaussian.mean).squaredNorm();
            if (gaussian.representative_index < 0 ||
                distance2 < (target.points_[gaussian.representative_index] -
                             gaussian.mean)
                                    .squaredNorm()) {
                gaussian.representative_index = point_index;
            }
        }
        voxel_map.emplace(item.first, gaussian);
    }
    return voxel_map;
}

struct NDTLinearSystemReducer {
    const geometry::PointCloud &source_transformed;
    const geometry::PointCloud &target;
    const VoxelMap &voxel_map;
    const NeighborOffsets &offsets;
    std::size_t offset_count;
    double inv_voxel_size;
    double outlier_threshold;
    NDTLinearSystem system;

    NDTLinearSystemReducer(const geometry::PointCloud &source_transformed_,
                           const geometry::PointCloud &target_,
                           const VoxelMap &voxel_map_,
                           const NeighborOffsets &offsets_,
                           std::size_t offset_count_,
                           double inv_voxel_size_,
                           double outlier_threshold_)
        : source_transformed(source_transformed_),
          target(target_),
          voxel_map(voxel_map_),
          offsets(offsets_),
          offset_count(offset_count_),
          inv_voxel_size(inv_voxel_size_),
          outlier_threshold(outlier_threshold_) {}

    NDTLinearSystemReducer(NDTLinearSystemReducer &other, tbb::split)
        : source_transformed(other.source_transformed),
          target(other.target),
          voxel_map(other.voxel_map),
          offsets(other.offsets),
          offset_count(other.offset_count),
          inv_voxel_size(other.inv_voxel_size),
          outlier_threshold(other.outlier_threshold) {}

    void operator()(const tbb::blocked_range<std::size_t> &range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            const Eigen::Vector3d &point = source_transformed.points_[i];
            const VoxelKey key = GetVoxelKey(point, inv_voxel_size);
            double best_residual2 = outlier_threshold;
            int best_target_index = -1;
            for (std::size_t j = 0; j < offset_count; ++j) {
                const VoxelKey neighbor{key.x + offsets[j].x,
                                        key.y + offsets[j].y,
                                        key.z + offsets[j].z};
                const auto voxel_itr = voxel_map.find(neighbor);
                if (voxel_itr == voxel_map.end()) {
                    continue;
                }

                const Eigen::Vector3d diff = point - voxel_itr->second.mean;
                const Eigen::Matrix3d &information =
                        voxel_itr->second.information;
                const double distance = diff.transpose() * information * diff;
                if (!std::isfinite(distance) || distance > outlier_threshold) {
                    continue;
                }

                if (distance <= best_residual2) {
                    best_residual2 = distance;
                    best_target_index = voxel_itr->second.representative_index;
                }

                Eigen::Matrix<double, 3, 6> jacobian;
                jacobian.block<3, 3>(0, 0) = -utility::SkewMatrix(point);
                jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                system.JTJ.noalias() +=
                        jacobian.transpose() * information * jacobian;
                system.JTr.noalias() +=
                        jacobian.transpose() * information * diff;
                system.residual2 += distance;
                ++system.residual_count;
            }
            if (best_target_index >= 0) {
                system.euclidean_error2 +=
                        (point - target.points_[best_target_index])
                                .squaredNorm();
                ++system.correspondence_count;
            }
        }
    }

    void join(NDTLinearSystemReducer &other) {
        system.JTJ += other.system.JTJ;
        system.JTr += other.system.JTr;
        system.residual2 += other.system.residual2;
        system.residual_count += other.system.residual_count;
        system.euclidean_error2 += other.system.euclidean_error2;
        system.correspondence_count += other.system.correspondence_count;
    }
};

NDTLinearSystem ComputeNDTLinearSystem(
        const geometry::PointCloud &source_transformed,
        const geometry::PointCloud &target,
        const VoxelMap &voxel_map,
        const NormalDistributionsTransformOption &option) {
    const double inv_voxel_size = 1.0 / option.voxel_size_;
    std::size_t offset_count;
    const auto offsets =
            GetNeighborOffsets(option.neighbor_search_type_, offset_count);
    NDTLinearSystemReducer reducer(source_transformed, target, voxel_map,
                                   offsets, offset_count, inv_voxel_size,
                                   option.outlier_threshold_);
    tbb::parallel_reduce(tbb::blocked_range<std::size_t>(
                                 0, source_transformed.points_.size(),
                                 utility::DefaultGrainSizeTBB()),
                         reducer);
    return std::move(reducer.system);
}

struct NDTResultReducer {
    const geometry::PointCloud &source_transformed;
    const geometry::PointCloud &target;
    const VoxelMap &voxel_map;
    const NeighborOffsets &offsets;
    std::size_t offset_count;
    double inv_voxel_size;
    double outlier_threshold;
    CorrespondenceSet correspondences;
    double euclidean_error2 = 0.0;

    NDTResultReducer(const geometry::PointCloud &source_transformed_,
                     const geometry::PointCloud &target_,
                     const VoxelMap &voxel_map_,
                     const NeighborOffsets &offsets_,
                     std::size_t offset_count_,
                     double inv_voxel_size_,
                     double outlier_threshold_)
        : source_transformed(source_transformed_),
          target(target_),
          voxel_map(voxel_map_),
          offsets(offsets_),
          offset_count(offset_count_),
          inv_voxel_size(inv_voxel_size_),
          outlier_threshold(outlier_threshold_) {}

    NDTResultReducer(NDTResultReducer &other, tbb::split)
        : source_transformed(other.source_transformed),
          target(other.target),
          voxel_map(other.voxel_map),
          offsets(other.offsets),
          offset_count(other.offset_count),
          inv_voxel_size(other.inv_voxel_size),
          outlier_threshold(other.outlier_threshold) {}

    void operator()(const tbb::blocked_range<std::size_t> &range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            const Eigen::Vector3d &point = source_transformed.points_[i];
            const VoxelKey key = GetVoxelKey(point, inv_voxel_size);
            bool has_inlier = false;
            double best_residual2 = outlier_threshold;
            double best_euclidean_error2 = 0.0;
            int best_target_index = -1;
            for (std::size_t j = 0; j < offset_count; ++j) {
                const VoxelKey neighbor{key.x + offsets[j].x,
                                        key.y + offsets[j].y,
                                        key.z + offsets[j].z};
                const auto voxel_itr = voxel_map.find(neighbor);
                if (voxel_itr == voxel_map.end()) {
                    continue;
                }
                const Eigen::Vector3d diff = point - voxel_itr->second.mean;
                const double distance =
                        diff.transpose() * voxel_itr->second.information * diff;
                if (std::isfinite(distance) && distance <= best_residual2) {
                    has_inlier = true;
                    best_residual2 = distance;
                    best_target_index = voxel_itr->second.representative_index;
                    best_euclidean_error2 =
                            (point - target.points_[best_target_index])
                                    .squaredNorm();
                }
            }

            if (has_inlier) {
                correspondences.emplace_back(static_cast<int>(i),
                                             best_target_index);
                euclidean_error2 += best_euclidean_error2;
            }
        }
    }

    void join(NDTResultReducer &other) {
        correspondences.insert(correspondences.end(),
                               other.correspondences.begin(),
                               other.correspondences.end());
        euclidean_error2 += other.euclidean_error2;
    }
};

RegistrationResult EvaluateNDTResult(
        const geometry::PointCloud &source_transformed,
        const geometry::PointCloud &target,
        const Eigen::Matrix4d &transformation,
        const VoxelMap &voxel_map,
        const NormalDistributionsTransformOption &option) {
    RegistrationResult result(transformation);
    const double inv_voxel_size = 1.0 / option.voxel_size_;
    std::size_t offset_count;
    const auto offsets =
            GetNeighborOffsets(option.neighbor_search_type_, offset_count);
    NDTResultReducer reducer(source_transformed, target, voxel_map, offsets,
                             offset_count, inv_voxel_size,
                             option.outlier_threshold_);
    tbb::parallel_reduce(tbb::blocked_range<std::size_t>(
                                 0, source_transformed.points_.size(),
                                 utility::DefaultGrainSizeTBB()),
                         reducer);
    result.correspondence_set_ = std::move(reducer.correspondences);
    if (!result.correspondence_set_.empty()) {
        const double correspondence_count =
                static_cast<double>(result.correspondence_set_.size());
        result.fitness_ =
                correspondence_count /
                static_cast<double>(source_transformed.points_.size());
        result.inlier_rmse_ =
                std::sqrt(reducer.euclidean_error2 / correspondence_count);
    }
    return result;
}

}  // namespace

NormalDistributionsTransformOption::NormalDistributionsTransformOption(
        double voxel_size,
        int min_points_per_voxel,
        double covariance_regularization,
        double transformation_epsilon,
        double relative_objective,
        int max_iteration,
        double outlier_threshold,
        int neighbor_search_type)
    : voxel_size_(voxel_size),
      min_points_per_voxel_(min_points_per_voxel),
      covariance_regularization_(covariance_regularization),
      transformation_epsilon_(transformation_epsilon),
      relative_objective_(relative_objective),
      max_iteration_(max_iteration),
      outlier_threshold_(outlier_threshold),
      neighbor_search_type_(neighbor_search_type) {
    ValidateNDTOption(*this);
}

RegistrationResult RegistrationNDT(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const NormalDistributionsTransformOption &option,
        const Eigen::Matrix4d &init) {
    ValidateNDTOption(option);
    if (source.IsEmpty() || target.IsEmpty()) {
        return RegistrationResult(init);
    }

    const VoxelMap voxel_map = BuildVoxelGaussians(target, option);
    if (voxel_map.empty()) {
        utility::LogError(
                "No target NDT voxels were created. Increase voxel_size or "
                "decrease min_points_per_voxel.");
    }

    Eigen::Matrix4d transformation = init;
    geometry::PointCloud pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }

    double previous_objective = std::numeric_limits<double>::infinity();

    for (int i = 0; i < option.max_iteration_; ++i) {
        const NDTLinearSystem system =
                ComputeNDTLinearSystem(pcd, target, voxel_map, option);

        if (system.residual_count < 6) {
            utility::LogWarning(
                    "NDT iteration {:d}: too few effective residuals ({:d}).",
                    i, system.residual_count);
            break;
        }

        const double objective = system.MeanObjective();
        utility::LogDebug(
                "NDT Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}, "
                "mean Mahalanobis objective {:.4f}",
                i, system.Fitness(pcd.points_.size()), system.InlierRMSE(),
                objective);
        if (i > 0) {
            const double relative_objective_change =
                    std::abs(previous_objective - objective) /
                    std::max(std::abs(previous_objective),
                             std::numeric_limits<double>::epsilon());
            if (relative_objective_change < option.relative_objective_) {
                break;
            }
        }
        previous_objective = objective;

        if (!system.JTJ.allFinite() || !system.JTr.allFinite()) {
            utility::LogWarning(
                    "NDT iteration {:d}: linear system is non-finite.", i);
            break;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix6d> hessian_solver(
                system.JTJ, Eigen::EigenvaluesOnly);
        if (hessian_solver.info() != Eigen::Success ||
            !hessian_solver.eigenvalues().allFinite()) {
            utility::LogWarning(
                    "NDT iteration {:d}: Hessian eigenvalue decomposition "
                    "failed.",
                    i);
            break;
        }
        constexpr double kMaxHessianConditionNumber = 1e12;
        const double min_eigenvalue = hessian_solver.eigenvalues().minCoeff();
        const double max_eigenvalue = hessian_solver.eigenvalues().maxCoeff();
        if (max_eigenvalue <= 0.0 ||
            min_eigenvalue <= max_eigenvalue / kMaxHessianConditionNumber) {
            utility::LogWarning(
                    "NDT iteration {:d}: Hessian is rank-deficient or "
                    "ill-conditioned.",
                    i);
            break;
        }

        bool is_success = false;
        Eigen::Vector6d update_vector;
        std::tie(is_success, update_vector) =
                utility::SolveLinearSystemPSD(system.JTJ, -system.JTr);
        if (!is_success || !update_vector.allFinite()) {
            utility::LogWarning(
                    "NDT iteration {:d}: linear solve failed or produced a "
                    "non-finite update.",
                    i);
            break;
        }
        const Eigen::Matrix4d update =
                utility::TransformVector6dToMatrix4d(update_vector);
        const Eigen::Matrix4d candidate_transformation =
                update * transformation;
        if (!update.allFinite() || !candidate_transformation.allFinite()) {
            utility::LogWarning(
                    "NDT iteration {:d}: transformation update is "
                    "non-finite.",
                    i);
            break;
        }

        transformation = candidate_transformation;
        pcd.Transform(update);

        if (update_vector.norm() < option.transformation_epsilon_) {
            break;
        }
    }

    return EvaluateNDTResult(pcd, target, transformation, voxel_map, option);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
