// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {
namespace {

constexpr double kMinEigenvalue = 1e-9;
constexpr double kMinPivot = 1e-12;
constexpr int kMinCorrespondences = 10;

struct DCRegSystem {
    Eigen::Matrix6d hessian = Eigen::Matrix6d::Zero();
    Eigen::Vector6d rhs = Eigen::Vector6d::Zero();
};

struct DegeneracyDetection {
    bool factorization_ok = false;
    Eigen::Vector3d lambda_schur_rot =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d lambda_schur_trans =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix3d raw_rot_basis = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d raw_trans_basis = Eigen::Matrix3d::Identity();
};

struct DegeneracyCharacterization {
    bool factorization_ok = false;
    Eigen::Matrix6d preconditioner = Eigen::Matrix6d::Identity();
    double condition_number_rot = std::numeric_limits<double>::quiet_NaN();
    double condition_number_trans = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d aligned_lambda_rot =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d aligned_lambda_trans =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d clamped_lambda_rot =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d clamped_lambda_trans =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3i weak_rot_axes = Eigen::Vector3i::Zero();
    Eigen::Vector3i weak_trans_axes = Eigen::Vector3i::Zero();
    bool is_degenerate = false;
};

struct SolverResult {
    Eigen::Vector6d update = Eigen::Vector6d::Zero();
    std::string solver_type = "dense";
    bool pcg_converged = false;
    int pcg_iteration = 0;
};

struct LocalPlaneCorrespondence {
    int source_index = -1;
    int target_index = -1;
    Eigen::Vector3d point_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_world = Eigen::Vector3d::Zero();
    double residual = 0.0;
    double weight = 0.0;
    double weight_derivative = 0.0;
};

struct LocalPlaneCorrespondenceSet {
    std::vector<LocalPlaneCorrespondence> correspondences;
    double residual_square_sum = 0.0;
};

DCRegSystem BuildPointToPlaneSystem(const geometry::PointCloud &source,
                                    const geometry::PointCloud &target,
                                    const CorrespondenceSet &corres,
                                    const RobustKernel &kernel) {
    // Keep the same residual and Jacobian convention as
    // TransformationEstimationPointToPlane; only the linear solver changes.
    // In RegistrationICP, `source` is already transformed by the current pose,
    // and the returned SE(3) increment is left-multiplied in the target/world
    // frame.
    auto compute_jacobian_and_residual = [&](int i, Eigen::Vector6d &J_r,
                                             double &r, double &w) {
        const Eigen::Vector3d &vs = source.points_[corres[i][0]];
        const Eigen::Vector3d &vt = target.points_[corres[i][1]];
        const Eigen::Vector3d &nt = target.normals_[corres[i][1]];
        r = (vs - vt).dot(nt);
        w = kernel.Weight(r);
        J_r.block<3, 1>(0, 0) = vs.cross(nt);
        J_r.block<3, 1>(3, 0) = nt;
    };

    DCRegSystem system;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(system.hessian, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());
    system.rhs = -JTr;
    return system;
}

Eigen::Matrix4d MakeTransform(const Eigen::Matrix3d &rotation,
                              const Eigen::Vector3d &translation) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    return transform;
}

Eigen::Matrix3d So3Exp(const Eigen::Vector3d &omega) {
    const double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity() + utility::SkewMatrix(omega);
    }

    const Eigen::Vector3d axis = omega / theta;
    const Eigen::Matrix3d axis_hat = utility::SkewMatrix(axis);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * axis_hat +
           (1.0 - std::cos(theta)) * axis_hat * axis_hat;
}

bool FitLocalPlaneFromNeighbors(const geometry::PointCloud &target,
                                const std::vector<int> &neighbor_indices,
                                double max_plane_thickness,
                                Eigen::Vector3d &normal_world,
                                double &plane_offset) {
    Eigen::Matrix<double, Eigen::Dynamic, 3> a_matrix(neighbor_indices.size(),
                                                      3);
    const Eigen::VectorXd b_vector =
            Eigen::VectorXd::Constant(neighbor_indices.size(), -1.0);
    for (int row = 0; row < static_cast<int>(neighbor_indices.size()); ++row) {
        a_matrix.row(row) = target.points_[neighbor_indices[row]].transpose();
    }

    const Eigen::Vector3d plane_coefficients =
            a_matrix.colPivHouseholderQr().solve(b_vector);
    const double coeff_norm = plane_coefficients.norm();
    if (coeff_norm < 1e-6) {
        return false;
    }

    normal_world = plane_coefficients / coeff_norm;
    plane_offset = 1.0 / coeff_norm;

    double max_residual = 0.0;
    for (const int index : neighbor_indices) {
        max_residual = std::max(
                max_residual, std::abs(normal_world.dot(target.points_[index]) +
                                       plane_offset));
    }
    return max_residual < max_plane_thickness;
}

LocalPlaneCorrespondenceSet CollectLocalPlaneCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transform,
        const DCRegOption &option) {
    LocalPlaneCorrespondenceSet result;
    if (max_correspondence_distance <= 0.0 || option.local_plane_knn_ <= 0) {
        return result;
    }

    const double radius2 =
            max_correspondence_distance * max_correspondence_distance;
    const int knn = option.local_plane_knn_;
    result.correspondences.reserve(source.points_.size());
    for (int i = 0; i < static_cast<int>(source.points_.size()); ++i) {
        const Eigen::Vector3d point_body = source.points_[i];
        const Eigen::Vector3d point_world =
                transform.block<3, 3>(0, 0) * point_body +
                transform.block<3, 1>(0, 3);
        const Eigen::Vector3d search_point(static_cast<float>(point_world.x()),
                                           static_cast<float>(point_world.y()),
                                           static_cast<float>(point_world.z()));

        std::vector<int> neighbor_indices(knn);
        std::vector<double> neighbor_distances(knn);
        if (target_kdtree.SearchKNN(search_point, knn, neighbor_indices,
                                    neighbor_distances) != knn ||
            neighbor_distances.back() >= radius2) {
            continue;
        }

        Eigen::Vector3d normal_world;
        double plane_offset = 0.0;
        if (!FitLocalPlaneFromNeighbors(target, neighbor_indices,
                                        option.local_plane_max_thickness_,
                                        normal_world, plane_offset)) {
            continue;
        }

        const double residual = normal_world.dot(point_world) + plane_offset;
        const double weight =
                std::max(0.0, 1.0 - option.local_plane_weight_slope_ *
                                              std::abs(residual));
        if (weight <= option.local_plane_min_weight_) {
            continue;
        }

        LocalPlaneCorrespondence correspondence;
        correspondence.source_index = i;
        correspondence.target_index = neighbor_indices.front();
        correspondence.point_body = point_body;
        correspondence.normal_world = normal_world;
        correspondence.residual = residual;
        correspondence.weight = weight;
        if (option.local_plane_use_weight_derivative_ && weight < 1.0 &&
            weight > 0.0) {
            correspondence.weight_derivative =
                    -option.local_plane_weight_slope_ *
                    (residual >= 0.0 ? 1.0 : -1.0);
        }
        result.residual_square_sum += residual * residual;
        result.correspondences.push_back(correspondence);
    }
    return result;
}

Eigen::Matrix<double, 1, 6> ComputeLocalFrameSo3Jacobian(
        const LocalPlaneCorrespondence &correspondence,
        const Eigen::Matrix3d &rotation) {
    Eigen::Matrix<double, 1, 6> jacobian;
    jacobian.block<1, 3>(0, 0) = -correspondence.normal_world.transpose() *
                                 rotation *
                                 utility::SkewMatrix(correspondence.point_body);
    jacobian.block<1, 3>(0, 3) =
            correspondence.normal_world.transpose() * rotation;
    return jacobian;
}

DCRegSystem BuildLocalFrameSystem(
        const std::vector<LocalPlaneCorrespondence> &correspondences,
        const Eigen::Matrix3d &rotation) {
    DCRegSystem system;
    for (const LocalPlaneCorrespondence &correspondence : correspondences) {
        const Eigen::Matrix<double, 1, 6> residual_jacobian =
                ComputeLocalFrameSo3Jacobian(correspondence, rotation);
        const Eigen::Matrix<double, 1, 6> full_jacobian =
                correspondence.weight * residual_jacobian +
                correspondence.residual * correspondence.weight_derivative *
                        residual_jacobian;
        const double weighted_residual =
                -correspondence.weight * correspondence.residual;
        system.hessian.noalias() += full_jacobian.transpose() * full_jacobian;
        system.rhs.noalias() += full_jacobian.transpose() * weighted_residual;
    }
    return system;
}

SolverResult SolveDefaultUpdate(const DCRegSystem &system) {
    SolverResult result;
    bool is_success;
    Eigen::VectorXd update;
    std::tie(is_success, update) =
            utility::SolveLinearSystemPSD(system.hessian, system.rhs);
    if (is_success && update.size() == 6 && update.allFinite()) {
        result.update = update;
        return result;
    }
    result.solver_type = "qr_fallback";
    result.update = system.hessian.colPivHouseholderQr().solve(system.rhs);
    return result;
}

SolverResult SolveRawQRUpdate(const DCRegSystem &system) {
    SolverResult result;
    result.solver_type = "qr_fallback";
    result.update = system.hessian.colPivHouseholderQr().solve(system.rhs);
    return result;
}

std::tuple<bool, Eigen::Vector6d> SolveRankDeficientMinimumNormUpdate(
        const DCRegSystem &system) {
    // Exact geometric null spaces, such as cylinder-axis translation, should
    // not receive arbitrary updates from an LDLT solve. Use the minimum-norm
    // solution when the full Hessian is rank deficient.
    const Eigen::Matrix6d hessian =
            0.5 * (system.hessian + system.hessian.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix6d> solver(hessian);
    if (solver.info() != Eigen::Success) {
        return std::make_tuple(false, Eigen::Vector6d::Zero());
    }

    const Eigen::Vector6d eigenvalues = solver.eigenvalues();
    const double lambda_max = eigenvalues.cwiseAbs().maxCoeff();
    if (lambda_max < kMinPivot) {
        return std::make_tuple(true, Eigen::Vector6d::Zero());
    }

    const double nullspace_threshold = std::max(lambda_max * 1e-10, kMinPivot);
    bool rank_deficient = false;
    Eigen::Vector6d inverse_eigenvalues = Eigen::Vector6d::Zero();
    for (int i = 0; i < 6; ++i) {
        if (eigenvalues(i) <= nullspace_threshold) {
            rank_deficient = true;
            continue;
        }
        inverse_eigenvalues(i) = 1.0 / eigenvalues(i);
    }
    if (!rank_deficient) {
        return std::make_tuple(false, Eigen::Vector6d::Zero());
    }

    const Eigen::Vector6d update =
            solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
            solver.eigenvectors().transpose() * system.rhs;
    if (!update.allFinite()) {
        return std::make_tuple(false, Eigen::Vector6d::Zero());
    }
    return std::make_tuple(true, update);
}

double ComputeConditionNumber(const Eigen::Vector3d &lambda) {
    const double lambda_max = lambda.maxCoeff();
    return lambda_max / std::max(lambda.minCoeff(), kMinPivot);
}

double ComputeFullConditionNumber(const Eigen::Matrix6d &hessian) {
    const Eigen::JacobiSVD<Eigen::Matrix6d> svd(hessian);
    const Eigen::Vector6d singular_values = svd.singularValues();
    return singular_values(0) / std::max(singular_values(5), kMinPivot);
}

bool ComputeSymmetricEigenDecomposition(const Eigen::Matrix3d &matrix,
                                        Eigen::Vector3d &eigenvalues,
                                        Eigen::Matrix3d &eigenvectors) {
    const Eigen::Matrix3d symmetric = 0.5 * (matrix + matrix.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    eigenvalues = solver.eigenvalues();
    eigenvectors = solver.eigenvectors();
    return eigenvalues.allFinite() && eigenvectors.allFinite();
}

DegeneracyDetection DetectDegeneracy(const Eigen::Matrix6d &hessian) {
    DegeneracyDetection detection;

    const Eigen::Matrix3d h_rr = hessian.block<3, 3>(0, 0);
    const Eigen::Matrix3d h_rt = hessian.block<3, 3>(0, 3);
    const Eigen::Matrix3d h_tr = hessian.block<3, 3>(3, 0);
    const Eigen::Matrix3d h_tt = hessian.block<3, 3>(3, 3);

    const Eigen::FullPivLU<Eigen::Matrix3d> lu_rr(h_rr);
    const Eigen::FullPivLU<Eigen::Matrix3d> lu_tt(h_tt);
    if (!lu_rr.isInvertible() || !lu_tt.isInvertible()) {
        return detection;
    }

    // DCReg analyzes clean rotational and translational subspaces via Schur
    // complements: S_R = H_RR - H_Rt H_tt^{-1} H_tR and
    // S_t = H_tt - H_tR H_RR^{-1} H_Rt.
    const Eigen::Matrix3d schur_rot = h_rr - h_rt * lu_tt.solve(h_tr);
    const Eigen::Matrix3d schur_trans = h_tt - h_tr * lu_rr.solve(h_rt);

    Eigen::Vector3d lambda_rot;
    Eigen::Vector3d lambda_trans;
    Eigen::Matrix3d raw_rot_basis;
    Eigen::Matrix3d raw_trans_basis;
    if (!ComputeSymmetricEigenDecomposition(schur_rot, lambda_rot,
                                            raw_rot_basis) ||
        !ComputeSymmetricEigenDecomposition(schur_trans, lambda_trans,
                                            raw_trans_basis)) {
        return detection;
    }

    detection.factorization_ok = true;
    detection.lambda_schur_rot = lambda_rot;
    detection.lambda_schur_trans = lambda_trans;
    detection.raw_rot_basis = raw_rot_basis;
    detection.raw_trans_basis = raw_trans_basis;
    return detection;
}

DegeneracyDetection DetectBlockEigenvalueFallback(
        const Eigen::Matrix6d &hessian) {
    DegeneracyDetection detection;

    const Eigen::Matrix3d h_rr = hessian.block<3, 3>(0, 0);
    const Eigen::Matrix3d h_tt = hessian.block<3, 3>(3, 3);
    Eigen::Vector3d lambda_rot;
    Eigen::Vector3d lambda_trans;
    Eigen::Matrix3d raw_rot_basis;
    Eigen::Matrix3d raw_trans_basis;
    if (!ComputeSymmetricEigenDecomposition(h_rr, lambda_rot, raw_rot_basis) ||
        !ComputeSymmetricEigenDecomposition(h_tt, lambda_trans,
                                            raw_trans_basis)) {
        return detection;
    }

    // When Schur complements cannot be formed because a block is singular, the
    // block spectra still give interpretable x/y/z weak-axis diagnostics. This
    // fallback is diagnostic-only; the ICP update continues to use the
    // minimum-norm rank-deficient solve.
    detection.factorization_ok = true;
    detection.lambda_schur_rot = lambda_rot;
    detection.lambda_schur_trans = lambda_trans;
    detection.raw_rot_basis = raw_rot_basis;
    detection.raw_trans_basis = raw_trans_basis;
    return detection;
}

bool AlignEigenBasisToAxes(const Eigen::Matrix3d &raw_basis,
                           Eigen::Matrix3d &aligned_basis,
                           std::array<int, 3> &original_indices) {
    // Eigenvectors are sign- and order-ambiguous. Align them to the physical
    // x/y/z axes before deciding which weak directions should be clamped.
    const std::array<Eigen::Vector3d, 3> refs = {
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitY(),
            Eigen::Vector3d::UnitZ(),
    };
    aligned_basis.setZero();
    original_indices.fill(-1);

    std::array<bool, 3> used = {false, false, false};
    for (int axis = 0; axis < 3; ++axis) {
        double best_score = -1.0;
        int best_index = -1;
        for (int candidate = 0; candidate < 3; ++candidate) {
            if (used[candidate]) {
                continue;
            }
            const double score =
                    std::abs(refs[axis].dot(raw_basis.col(candidate)));
            if (score > best_score) {
                best_score = score;
                best_index = candidate;
            }
        }
        if (best_index < 0) {
            return false;
        }
        used[best_index] = true;
        original_indices[axis] = best_index;
        Eigen::Vector3d aligned_column = raw_basis.col(best_index);
        if (refs[axis].dot(aligned_column) < 0.0) {
            aligned_column = -aligned_column;
        }
        aligned_basis.col(axis) = aligned_column;
    }
    return true;
}

Eigen::Vector3i ComputeWeakAxes(const Eigen::Vector3d &aligned_lambda,
                                const DCRegOption &option) {
    Eigen::Vector3i weak_axes = Eigen::Vector3i::Zero();
    const double lambda_max =
            std::max(aligned_lambda.maxCoeff(), kMinEigenvalue);
    const double threshold =
            std::max(option.degeneracy_condition_threshold_, 1.0);
    for (int axis = 0; axis < 3; ++axis) {
        const double condition =
                lambda_max / std::max(aligned_lambda(axis), kMinPivot);
        if (condition > threshold) {
            weak_axes(axis) = 1;
        }
    }
    return weak_axes;
}

Eigen::Vector3d ClampWeakEigenvalues(const Eigen::Vector3d &aligned_lambda,
                                     const Eigen::Vector3i &weak_axes,
                                     const DCRegOption &option) {
    Eigen::Vector3d clamped = aligned_lambda;
    const double lambda_max =
            std::max(aligned_lambda.maxCoeff(), kMinEigenvalue);
    const double kappa_target = std::max(option.kappa_target_, 1.0);
    const double weak_lambda =
            std::max(lambda_max / kappa_target, kMinEigenvalue);

    for (int axis = 0; axis < 3; ++axis) {
        if (weak_axes(axis)) {
            clamped(axis) = weak_lambda;
        }
    }
    return clamped;
}

DegeneracyCharacterization CharacterizeDegeneracy(
        const DegeneracyDetection &detection, const DCRegOption &option) {
    DegeneracyCharacterization characterization;
    characterization.factorization_ok = detection.factorization_ok;
    if (!detection.factorization_ok) {
        characterization.is_degenerate = true;
        characterization.weak_rot_axes = Eigen::Vector3i::Ones();
        characterization.weak_trans_axes = Eigen::Vector3i::Ones();
        return characterization;
    }

    Eigen::Matrix3d aligned_rot_basis;
    Eigen::Matrix3d aligned_trans_basis;
    std::array<int, 3> rot_indices;
    std::array<int, 3> trans_indices;
    if (!AlignEigenBasisToAxes(detection.raw_rot_basis, aligned_rot_basis,
                               rot_indices) ||
        !AlignEigenBasisToAxes(detection.raw_trans_basis, aligned_trans_basis,
                               trans_indices)) {
        characterization.factorization_ok = false;
        return characterization;
    }

    for (int axis = 0; axis < 3; ++axis) {
        characterization.aligned_lambda_rot(axis) =
                detection.lambda_schur_rot(rot_indices[axis]);
        characterization.aligned_lambda_trans(axis) =
                detection.lambda_schur_trans(trans_indices[axis]);
    }

    characterization.condition_number_rot =
            ComputeConditionNumber(characterization.aligned_lambda_rot);
    characterization.condition_number_trans =
            ComputeConditionNumber(characterization.aligned_lambda_trans);
    characterization.weak_rot_axes =
            ComputeWeakAxes(characterization.aligned_lambda_rot, option);
    characterization.weak_trans_axes =
            ComputeWeakAxes(characterization.aligned_lambda_trans, option);
    characterization.is_degenerate = characterization.weak_rot_axes.any() ||
                                     characterization.weak_trans_axes.any();
    characterization.clamped_lambda_rot =
            ClampWeakEigenvalues(characterization.aligned_lambda_rot,
                                 characterization.weak_rot_axes, option);
    characterization.clamped_lambda_trans =
            ClampWeakEigenvalues(characterization.aligned_lambda_trans,
                                 characterization.weak_trans_axes, option);

    characterization.preconditioner.setZero();
    characterization.preconditioner.block<3, 3>(0, 0) =
            aligned_rot_basis *
            characterization.clamped_lambda_rot.cwiseMax(kMinEigenvalue)
                    .cwiseInverse()
                    .asDiagonal() *
            aligned_rot_basis.transpose();
    characterization.preconditioner.block<3, 3>(3, 3) =
            aligned_trans_basis *
            characterization.clamped_lambda_trans.cwiseMax(kMinEigenvalue)
                    .cwiseInverse()
                    .asDiagonal() *
            aligned_trans_basis.transpose();
    return characterization;
}

SolverResult SolvePreconditionedUpdate(const DCRegSystem &system,
                                       const DCRegOption &option) {
    bool is_rank_deficient = false;
    Eigen::Vector6d rank_deficient_update;
    std::tie(is_rank_deficient, rank_deficient_update) =
            SolveRankDeficientMinimumNormUpdate(system);
    if (is_rank_deficient) {
        SolverResult result;
        result.update = rank_deficient_update;
        result.solver_type = "minimum_norm";
        return result;
    }

    const DegeneracyDetection detection = DetectDegeneracy(system.hessian);
    const DegeneracyCharacterization characterization =
            CharacterizeDegeneracy(detection, option);
    if (!characterization.factorization_ok ||
        !characterization.preconditioner.allFinite()) {
        return SolveDefaultUpdate(system);
    }

    // The original normal equation is preserved; weak eigenvalues are clamped
    // only in the preconditioner, not in the Hessian itself.
    const Eigen::Matrix6d hessian =
            0.5 * (system.hessian + system.hessian.transpose());
    const double rhs_norm = system.rhs.norm();
    if (rhs_norm < kMinPivot) {
        SolverResult result;
        result.solver_type = "zero_rhs";
        return result;
    }

    Eigen::Vector6d delta = Eigen::Vector6d::Zero();
    Eigen::Vector6d residual = system.rhs;
    const Eigen::Vector6d preconditioned_residual =
            characterization.preconditioner * residual;
    Eigen::Vector6d direction = preconditioned_residual;
    double rz_old = residual.dot(preconditioned_residual);
    if (!preconditioned_residual.allFinite() || !std::isfinite(rz_old) ||
        std::abs(rz_old) < kMinPivot) {
        return SolveDefaultUpdate(system);
    }

    const double tolerance = option.pcg_tolerance_ > 0.0
                                     ? option.pcg_tolerance_
                                     : DCRegOption().pcg_tolerance_;
    const double target_residual = tolerance * std::max(1.0, rhs_norm);
    const int max_iteration = std::max(1, option.pcg_max_iteration_);
    int pcg_iteration = 0;
    for (int iteration = 0; iteration < max_iteration; ++iteration) {
        pcg_iteration = iteration + 1;
        const Eigen::Vector6d hessian_direction = hessian * direction;
        const double denom = direction.dot(hessian_direction);
        if (!std::isfinite(denom) || std::abs(denom) < kMinPivot) {
            break;
        }

        const double alpha = rz_old / denom;
        if (!std::isfinite(alpha)) {
            break;
        }

        delta += alpha * direction;
        residual -= alpha * hessian_direction;
        if (!delta.allFinite() || !residual.allFinite()) {
            break;
        }
        if (residual.norm() <= target_residual) {
            SolverResult result;
            result.update = delta;
            result.solver_type = "pcg";
            result.pcg_converged = true;
            result.pcg_iteration = pcg_iteration;
            return result;
        }

        const Eigen::Vector6d z_next =
                characterization.preconditioner * residual;
        const double rz_new = residual.dot(z_next);
        if (!z_next.allFinite() || !std::isfinite(rz_new) ||
            std::abs(rz_old) < kMinPivot) {
            break;
        }

        const double beta = rz_new / rz_old;
        if (!std::isfinite(beta)) {
            break;
        }
        direction = z_next + beta * direction;
        rz_old = rz_new;
    }

    SolverResult result = SolveDefaultUpdate(system);
    result.pcg_iteration = pcg_iteration;
    return result;
}

SolverResult SolvePreconditionedUpdateDCRegCompatible(
        const DCRegSystem &system, const DCRegOption &option) {
    const DegeneracyDetection detection = DetectDegeneracy(system.hessian);
    const DegeneracyCharacterization characterization =
            CharacterizeDegeneracy(detection, option);
    if (!characterization.factorization_ok ||
        !characterization.preconditioner.allFinite()) {
        return SolveRawQRUpdate(system);
    }

    const Eigen::Matrix6d hessian =
            0.5 * (system.hessian + system.hessian.transpose());
    const double rhs_norm = system.rhs.norm();
    if (rhs_norm < kMinPivot) {
        SolverResult result;
        result.solver_type = "zero_rhs";
        result.pcg_converged = true;
        return result;
    }

    Eigen::Vector6d delta = Eigen::Vector6d::Zero();
    Eigen::Vector6d residual = system.rhs;
    Eigen::Vector6d preconditioned_residual =
            characterization.preconditioner * residual;
    if (!preconditioned_residual.allFinite()) {
        return SolveRawQRUpdate(system);
    }
    Eigen::Vector6d direction = preconditioned_residual;
    double rz_old = residual.dot(preconditioned_residual);
    if (!std::isfinite(rz_old) || std::abs(rz_old) < 1e-20) {
        return SolveRawQRUpdate(system);
    }

    const double tolerance = option.pcg_tolerance_ > 0.0
                                     ? option.pcg_tolerance_
                                     : DCRegOption().pcg_tolerance_;
    const double target_residual = tolerance * std::max(1.0, rhs_norm);
    const int max_iteration = std::max(option.pcg_max_iteration_, 6);
    int pcg_iteration = 0;
    for (int iteration = 0; iteration < max_iteration; ++iteration) {
        pcg_iteration = iteration + 1;
        const Eigen::Vector6d hessian_direction = hessian * direction;
        const double denom = direction.dot(hessian_direction);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-20) {
            break;
        }

        const double alpha = rz_old / denom;
        if (!std::isfinite(alpha)) {
            break;
        }

        delta += alpha * direction;
        residual -= alpha * hessian_direction;
        if (!delta.allFinite() || !residual.allFinite()) {
            break;
        }
        if (residual.norm() <= target_residual) {
            SolverResult result;
            result.update = delta;
            result.solver_type = "pcg";
            result.pcg_converged = true;
            result.pcg_iteration = pcg_iteration;
            return result;
        }

        const Eigen::Vector6d z_next =
                characterization.preconditioner * residual;
        const double rz_new = residual.dot(z_next);
        if (!z_next.allFinite() || !std::isfinite(rz_new) ||
            std::abs(rz_old) < 1e-20) {
            break;
        }

        const double beta = rz_new / rz_old;
        if (!std::isfinite(beta)) {
            break;
        }
        direction = z_next + beta * direction;
        preconditioned_residual = z_next;
        rz_old = rz_new;
    }

    SolverResult result = SolveRawQRUpdate(system);
    result.pcg_iteration = pcg_iteration;
    return result;
}

std::string FormatAxisList(const Eigen::Vector3i &weak_axes) {
    constexpr std::array<const char *, 3> kAxisLabels = {"x", "y", "z"};
    std::string text;
    for (int axis = 0; axis < 3; ++axis) {
        if (!weak_axes(axis)) {
            continue;
        }
        if (!text.empty()) {
            text += ", ";
        }
        text += kAxisLabels[axis];
    }
    return text.empty() ? "none" : text;
}

std::string FormatDouble(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream stream;
    stream << std::setprecision(6) << value;
    return stream.str();
}

void FillDegeneracyDescription(DCRegDegeneracyAnalysis &analysis) {
    analysis.weak_rotation_axes_description_ =
            FormatAxisList(analysis.weak_rotation_axes_);
    analysis.weak_translation_axes_description_ =
            FormatAxisList(analysis.weak_translation_axes_);

    std::ostringstream stream;
    if (!analysis.has_correspondence_) {
        stream << "No correspondences; DCReg degeneracy was not evaluated.";
    } else if (!analysis.has_target_normals_) {
        stream << "Target point cloud has no normals; point-to-plane DCReg "
                  "degeneracy was not evaluated.";
    } else if (!analysis.is_degenerate_) {
        stream << "No degeneracy detected in the " << analysis.coordinate_frame_
               << " normal equation.";
    } else {
        stream << "Degenerate in the " << analysis.coordinate_frame_
               << " normal equation: weak rotation axes = "
               << analysis.weak_rotation_axes_description_
               << ", weak translation axes = "
               << analysis.weak_translation_axes_description_
               << ". condition_number_full = "
               << FormatDouble(analysis.condition_number_full_)
               << ", condition_number_rotation = "
               << FormatDouble(analysis.condition_number_rotation_)
               << ", condition_number_translation = "
               << FormatDouble(analysis.condition_number_translation_) << ". ";
        if (analysis.coordinate_frame_.find("local body frame") !=
            std::string::npos) {
            stream << "This diagnostic uses the standalone-compatible kNN "
                      "local-plane residual and SO(3) local-frame Jacobian.";
        } else {
            stream << "This Open3D diagnostic uses the target/world-frame "
                      "left-multiplied SE(3) point-to-plane Jacobian; it is "
                      "not the standalone DCReg SO(3) local-frame parking-lot "
                      "diagnostic.";
        }
    }
    analysis.degeneracy_description_ = stream.str();
}

void PopulateAnalysisFromSystem(DCRegDegeneracyAnalysis &analysis,
                                const DCRegSystem &system,
                                const DCRegOption &option,
                                bool use_open3d_rank_deficient_fallback) {
    analysis.condition_number_full_ =
            ComputeFullConditionNumber(system.hessian);

    Eigen::Vector6d rank_deficient_update;
    std::tie(analysis.is_rank_deficient_, rank_deficient_update) =
            SolveRankDeficientMinimumNormUpdate(system);

    const DegeneracyDetection schur_detection =
            DetectDegeneracy(system.hessian);
    DegeneracyDetection diagnostic_detection = schur_detection;
    if (use_open3d_rank_deficient_fallback &&
        !diagnostic_detection.factorization_ok && analysis.is_rank_deficient_) {
        diagnostic_detection = DetectBlockEigenvalueFallback(system.hessian);
    }

    const DegeneracyCharacterization characterization =
            CharacterizeDegeneracy(diagnostic_detection, option);
    analysis.schur_factorization_ok_ = schur_detection.factorization_ok;
    analysis.condition_number_rotation_ = characterization.condition_number_rot;
    analysis.condition_number_translation_ =
            characterization.condition_number_trans;
    analysis.schur_eigenvalues_rotation_ =
            diagnostic_detection.lambda_schur_rot;
    analysis.schur_eigenvalues_translation_ =
            diagnostic_detection.lambda_schur_trans;
    analysis.axis_aligned_eigenvalues_rotation_ =
            characterization.aligned_lambda_rot;
    analysis.axis_aligned_eigenvalues_translation_ =
            characterization.aligned_lambda_trans;
    analysis.clamped_eigenvalues_rotation_ =
            characterization.clamped_lambda_rot;
    analysis.clamped_eigenvalues_translation_ =
            characterization.clamped_lambda_trans;
    analysis.weak_rotation_axes_ = characterization.weak_rot_axes;
    analysis.weak_translation_axes_ = characterization.weak_trans_axes;
    analysis.is_degenerate_ =
            analysis.is_rank_deficient_ || characterization.is_degenerate;

    const SolverResult solver_result =
            use_open3d_rank_deficient_fallback
                    ? SolvePreconditionedUpdate(system, option)
                    : SolvePreconditionedUpdateDCRegCompatible(system, option);
    analysis.solver_type_ = solver_result.solver_type;
    analysis.pcg_converged_ = solver_result.pcg_converged;
    analysis.pcg_iteration_ = solver_result.pcg_iteration;
}

}  // namespace

DCRegDegeneracyAnalysis ComputeDCRegDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const DCRegOption &option,
        const RobustKernel &kernel) {
    DCRegDegeneracyAnalysis analysis;
    analysis.has_correspondence_ = !corres.empty();
    analysis.has_target_normals_ = target.HasNormals();
    if (corres.empty() || !target.HasNormals()) {
        FillDegeneracyDescription(analysis);
        return analysis;
    }

    const DCRegSystem system =
            BuildPointToPlaneSystem(source, target, corres, kernel);
    PopulateAnalysisFromSystem(analysis, system, option, true);
    FillDegeneracyDescription(analysis);
    return analysis;
}

DCRegDegeneracyAnalysis ComputeDCRegDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const DCRegOption &option) {
    const L2Loss kernel;
    return ComputeDCRegDegeneracyAnalysis(source, target, corres, option,
                                          kernel);
}

DCRegDegeneracyAnalysis ComputeDCRegLocalDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation,
        const DCRegOption &option) {
    DCRegDegeneracyAnalysis analysis;
    analysis.coordinate_frame_ =
            "local body frame (standalone DCReg SO(3) update)";
    analysis.has_target_normals_ = true;

    geometry::KDTreeFlann target_kdtree;
    target_kdtree.SetGeometry(target);
    const LocalPlaneCorrespondenceSet local_correspondences =
            CollectLocalPlaneCorrespondences(source, target, target_kdtree,
                                             max_correspondence_distance,
                                             transformation, option);
    analysis.has_correspondence_ =
            !local_correspondences.correspondences.empty();
    if (!analysis.has_correspondence_) {
        FillDegeneracyDescription(analysis);
        return analysis;
    }

    const DCRegSystem system =
            BuildLocalFrameSystem(local_correspondences.correspondences,
                                  transformation.block<3, 3>(0, 0));
    PopulateAnalysisFromSystem(analysis, system, option, false);
    FillDegeneracyDescription(analysis);
    return analysis;
}

RegistrationResult RegistrationICPDCRegLocal(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init,
        const DCRegOption &option,
        const ICPConvergenceCriteria &criteria) {
    RegistrationResult result(init);
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }

    geometry::KDTreeFlann target_kdtree;
    target_kdtree.SetGeometry(target);

    Eigen::Matrix3d rotation = init.block<3, 3>(0, 0);
    Eigen::Vector3d translation = init.block<3, 1>(0, 3);
    for (int iteration = 0; iteration < criteria.max_iteration_; ++iteration) {
        const Eigen::Matrix4d current_transform =
                MakeTransform(rotation, translation);
        const LocalPlaneCorrespondenceSet local_correspondences =
                CollectLocalPlaneCorrespondences(source, target, target_kdtree,
                                                 max_correspondence_distance,
                                                 current_transform, option);
        if (static_cast<int>(local_correspondences.correspondences.size()) <
            kMinCorrespondences) {
            result.transformation_ = current_transform;
            result.correspondence_set_.clear();
            result.fitness_ = 0.0;
            result.inlier_rmse_ = 0.0;
            return result;
        }

        const DCRegSystem system = BuildLocalFrameSystem(
                local_correspondences.correspondences, rotation);
        const SolverResult solver_result =
                SolvePreconditionedUpdateDCRegCompatible(system, option);
        const Eigen::Vector6d &delta = solver_result.update;
        if (!delta.allFinite()) {
            result.transformation_ = current_transform;
            return result;
        }

        const Eigen::Matrix3d previous_rotation = rotation;
        rotation = rotation * So3Exp(delta.head<3>());
        translation = translation + previous_rotation * delta.tail<3>();

        result.transformation_ = MakeTransform(rotation, translation);
        result.correspondence_set_.clear();
        result.correspondence_set_.reserve(
                local_correspondences.correspondences.size());
        for (const LocalPlaneCorrespondence &correspondence :
             local_correspondences.correspondences) {
            result.correspondence_set_.emplace_back(
                    correspondence.source_index, correspondence.target_index);
        }
        result.fitness_ =
                static_cast<double>(
                        local_correspondences.correspondences.size()) /
                static_cast<double>(source.points_.size());
        result.inlier_rmse_ = std::sqrt(
                local_correspondences.residual_square_sum /
                static_cast<double>(
                        local_correspondences.correspondences.size()));

        if (delta.head<3>().norm() < option.local_frame_convergence_rotation_ &&
            delta.tail<3>().norm() <
                    option.local_frame_convergence_translation_) {
            return result;
        }
    }

    return result;
}

Eigen::Matrix4d
TransformationEstimationPointToPlaneDCReg::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) {
        return Eigen::Matrix4d::Identity();
    }

    const DCRegSystem system =
            BuildPointToPlaneSystem(source, target, corres, *kernel_);
    const Eigen::Vector6d update =
            SolvePreconditionedUpdate(system, option_).update;
    if (!update.allFinite()) {
        return Eigen::Matrix4d::Identity();
    }
    // RegistrationICP applies this matrix as update * current_pose, matching
    // the legacy point-to-plane estimator's left-multiplied SE(3) convention.
    return utility::TransformVector6dToMatrix4d(update);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
