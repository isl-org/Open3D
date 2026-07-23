// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/NormalDistributionsTransform.h"

// This implementation follows the same 3D NDT registration formulation used in
// https://github.com/gaoxiang12/slam_in_autonomous_driving/blob/master/src/ch7/ndt_3d.cc:
// target voxel Gaussian modeling, center/six-neighbor voxel residuals,
// covariance eigenvalue regularization, Mahalanobis outlier rejection, and
// Gauss-Newton SE(3) updates adapted to Open3D's registration API.

#include <Eigen/Dense>
#include <algorithm>
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

struct NDTLinearSystem {
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
    double residual2 = 0.0;
    int residual_count = 0;

    double MeanObjective() const {
        return residual2 / static_cast<double>(residual_count);
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

std::vector<VoxelKey> GetNeighborOffsets(int neighbor_search_type) {
    std::vector<VoxelKey> offsets{{0, 0, 0}};
    if (neighbor_search_type == 1) {
        offsets.push_back({-1, 0, 0});
        offsets.push_back({1, 0, 0});
        offsets.push_back({0, -1, 0});
        offsets.push_back({0, 1, 0});
        offsets.push_back({0, 0, -1});
        offsets.push_back({0, 0, 1});
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
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_indices;
    for (int i = 0; i < static_cast<int>(target.points_.size()); ++i) {
        voxel_indices[GetVoxelKey(target.points_[i], inv_voxel_size)].push_back(
                i);
    }

    VoxelMap voxel_map;
    for (const auto &item : voxel_indices) {
        const auto &indices = item.second;
        if (static_cast<int>(indices.size()) < option.min_points_per_voxel_) {
            continue;
        }

        VoxelGaussian gaussian;
        gaussian.count = static_cast<int>(indices.size());
        for (const int idx : indices) {
            gaussian.mean += target.points_[idx];
        }
        gaussian.mean /= static_cast<double>(indices.size());

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        double representative_distance2 = std::numeric_limits<double>::max();
        for (const int idx : indices) {
            const Eigen::Vector3d centered =
                    target.points_[idx] - gaussian.mean;
            covariance += centered * centered.transpose();
            const double distance2 = centered.squaredNorm();
            if (distance2 < representative_distance2) {
                representative_distance2 = distance2;
                gaussian.representative_index = idx;
            }
        }
        covariance /= static_cast<double>(indices.size() - 1);

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
        voxel_map.emplace(item.first, gaussian);
    }
    return voxel_map;
}

NDTLinearSystem ComputeNDTLinearSystem(
        const geometry::PointCloud &source_transformed,
        const VoxelMap &voxel_map,
        const NormalDistributionsTransformOption &option) {
    NDTLinearSystem system;
    const double inv_voxel_size = 1.0 / option.voxel_size_;
    const auto offsets = GetNeighborOffsets(option.neighbor_search_type_);
    for (const Eigen::Vector3d &point : source_transformed.points_) {
        const VoxelKey key = GetVoxelKey(point, inv_voxel_size);
        for (const auto &offset : offsets) {
            const VoxelKey neighbor{key.x + offset.x, key.y + offset.y,
                                    key.z + offset.z};
            const auto voxel_itr = voxel_map.find(neighbor);
            if (voxel_itr == voxel_map.end()) {
                continue;
            }

            const Eigen::Vector3d diff = point - voxel_itr->second.mean;
            const Eigen::Matrix3d &information = voxel_itr->second.information;
            const double distance = diff.transpose() * information * diff;
            if (!std::isfinite(distance) ||
                distance > option.outlier_threshold_) {
                continue;
            }

            Eigen::Matrix<double, 3, 6> jacobian;
            jacobian.block<3, 3>(0, 0) = -utility::SkewMatrix(point);
            jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

            system.JTJ += jacobian.transpose() * information * jacobian;
            system.JTr += jacobian.transpose() * information * diff;
            system.residual2 += distance;
            ++system.residual_count;
        }
    }
    return system;
}

RegistrationResult EvaluateNDTResult(
        const geometry::PointCloud &source_transformed,
        const geometry::PointCloud &target,
        const Eigen::Matrix4d &transformation,
        const VoxelMap &voxel_map,
        const NormalDistributionsTransformOption &option) {
    RegistrationResult result(transformation);
    if (source_transformed.points_.empty()) {
        return result;
    }

    const double inv_voxel_size = 1.0 / option.voxel_size_;
    const auto offsets = GetNeighborOffsets(option.neighbor_search_type_);
    double euclidean_error2 = 0.0;
    for (int i = 0; i < static_cast<int>(source_transformed.points_.size());
         ++i) {
        const Eigen::Vector3d &point = source_transformed.points_[i];
        const VoxelKey key = GetVoxelKey(point, inv_voxel_size);

        bool has_inlier = false;
        double best_residual2 = option.outlier_threshold_;
        double best_euclidean_error2 = 0.0;
        int best_target_index = -1;
        for (int j = 0; j < static_cast<int>(offsets.size()); ++j) {
            const VoxelKey neighbor{key.x + offsets[j].x, key.y + offsets[j].y,
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
            result.correspondence_set_.push_back(
                    Eigen::Vector2i(i, best_target_index));
            euclidean_error2 += best_euclidean_error2;
        }
    }

    if (!result.correspondence_set_.empty()) {
        const double correspondence_count =
                static_cast<double>(result.correspondence_set_.size());
        result.fitness_ =
                correspondence_count /
                static_cast<double>(source_transformed.points_.size());
        result.inlier_rmse_ =
                std::sqrt(euclidean_error2 / correspondence_count);
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

    RegistrationResult result =
            EvaluateNDTResult(pcd, target, transformation, voxel_map, option);
    double previous_objective = std::numeric_limits<double>::infinity();

    for (int i = 0; i < option.max_iteration_; ++i) {
        const NDTLinearSystem system =
                ComputeNDTLinearSystem(pcd, voxel_map, option);

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
                i, result.fitness_, result.inlier_rmse_, objective);
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

        result = EvaluateNDTResult(pcd, target, transformation, voxel_map,
                                   option);

        if (update_vector.norm() < option.transformation_epsilon_) {
            break;
        }
    }

    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
