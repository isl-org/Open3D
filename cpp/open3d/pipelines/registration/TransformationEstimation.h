// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open3d/pipelines/registration/RobustKernel.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

typedef std::vector<Eigen::Vector2i> CorrespondenceSet;

enum class TransformationEstimationType {
    Unspecified = 0,
    PointToPoint = 1,
    PointToPlane = 2,
    ColoredICP = 3,
    GeneralizedICP = 4,
};

/// \class TransformationEstimation
///
/// Base class that estimates a transformation between two point clouds
/// The virtual function ComputeTransformation() must be implemented in
/// subclasses.
class TransformationEstimation {
public:
    /// \brief Default Constructor.
    TransformationEstimation() {}
    virtual ~TransformationEstimation() {}

public:
    virtual TransformationEstimationType GetTransformationEstimationType()
            const = 0;
    /// Compute RMSE between source and target points cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    virtual double ComputeRMSE(const geometry::PointCloud &source,
                               const geometry::PointCloud &target,
                               const CorrespondenceSet &corres) const = 0;
    /// Compute transformation from source to target point cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    virtual Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const = 0;

    /// Initialize the source and target point cloud for the transformation
    /// estimation.
    ///
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param max_correspondence_distance Maximum correspondence distance.
    virtual std::tuple<std::shared_ptr<const geometry::PointCloud>,
                       std::shared_ptr<const geometry::PointCloud>>
    InitializePointCloudsForTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            double max_correspondence_distance) const = 0;
};

/// \class TransformationEstimationPointToPoint
///
/// Estimate a transformation for point to point distance.
class TransformationEstimationPointToPoint : public TransformationEstimation {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param with_scaling Set to True to estimate scaling, False to force
    /// scaling to be 1.
    TransformationEstimationPointToPoint(bool with_scaling = false)
        : with_scaling_(with_scaling) {}
    ~TransformationEstimationPointToPoint() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    std::tuple<std::shared_ptr<const geometry::PointCloud>,
               std::shared_ptr<const geometry::PointCloud>>
    InitializePointCloudsForTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            double max_correspondence_distance) const override;

public:
    /// \brief Set to True to estimate scaling, False to force scaling to be 1.
    ///
    /// The homogeneous transformation is given by\n
    /// T = [ cR t]\n
    ///    [0   1]\n
    /// Sets 𝑐=1 if with_scaling is False.
    bool with_scaling_ = false;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// \class TransformationEstimationPointToPlane
///
/// Class to estimate a transformation for point to plane distance.
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    /// \brief Default Constructor.
    TransformationEstimationPointToPlane() {}
    ~TransformationEstimationPointToPlane() override {}

    /// \brief Constructor that takes as input a RobustKernel.
    /// \param kernel Any of the implemented statistical robust kernel for
    /// outlier rejection.
    explicit TransformationEstimationPointToPlane(
            std::shared_ptr<RobustKernel> kernel)
        : kernel_(std::move(kernel)) {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    std::tuple<std::shared_ptr<const geometry::PointCloud>,
               std::shared_ptr<const geometry::PointCloud>>
    InitializePointCloudsForTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            double max_correspondence_distance) const override;

public:
    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};
/// \class DCRegOption
///
/// Options for degeneracy-aware point-to-plane ICP.
///
/// This estimator adapts the DCReg formulation from Hu et al.,
/// "DCReg: Decoupled Characterization for Efficient Degenerate LiDAR
/// Registration", arXiv:2509.06285, https://arxiv.org/abs/2509.06285 .
/// DCReg detects weak rotational and translational directions from the Schur
/// complements of the ICP normal equation and uses the result as a
/// preconditioner.
class DCRegOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param degeneracy_condition_threshold Directions whose Schur condition
    /// number exceeds this threshold are treated as weak.
    /// \param kappa_target Target condition number after clamping weak Schur
    /// eigenvalues.
    /// \param pcg_tolerance Relative residual threshold for the PCG solve.
    /// \param pcg_max_iteration Maximum number of PCG iterations before
    /// falling back to the default dense solve.
    DCRegOption(double degeneracy_condition_threshold = 10.0,
                double kappa_target = 10.0,
                double pcg_tolerance = 1e-6,
                int pcg_max_iteration = 10,
                int local_plane_knn = 5,
                double local_plane_max_thickness = 0.2,
                double local_plane_weight_slope = 0.9,
                double local_plane_min_weight = 0.1,
                bool local_plane_use_weight_derivative = true,
                double local_frame_convergence_rotation = 1e-5,
                double local_frame_convergence_translation = 1e-3)
        : degeneracy_condition_threshold_(degeneracy_condition_threshold),
          kappa_target_(kappa_target),
          pcg_tolerance_(pcg_tolerance),
          pcg_max_iteration_(pcg_max_iteration),
          local_plane_knn_(local_plane_knn),
          local_plane_max_thickness_(local_plane_max_thickness),
          local_plane_weight_slope_(local_plane_weight_slope),
          local_plane_min_weight_(local_plane_min_weight),
          local_plane_use_weight_derivative_(local_plane_use_weight_derivative),
          local_frame_convergence_rotation_(local_frame_convergence_rotation),
          local_frame_convergence_translation_(
                  local_frame_convergence_translation) {}

public:
    /// Schur condition threshold for weak directions.
    double degeneracy_condition_threshold_ = 10.0;
    /// Target condition number used when clamping weak Schur eigenvalues.
    double kappa_target_ = 10.0;
    /// Relative residual threshold for the PCG solve.
    double pcg_tolerance_ = 1e-6;
    /// Maximum number of PCG iterations.
    int pcg_max_iteration_ = 10;
    /// Number of target neighbors used by the DCReg-local plane fit.
    int local_plane_knn_ = 5;
    /// Maximum accepted local plane residual on its supporting neighbors.
    double local_plane_max_thickness_ = 0.2;
    /// Slope for the original DCReg piecewise-linear robust weight.
    double local_plane_weight_slope_ = 0.9;
    /// Minimum accepted original DCReg robust weight.
    double local_plane_min_weight_ = 0.1;
    /// Whether to include the original robust-weight derivative term.
    bool local_plane_use_weight_derivative_ = true;
    /// Local-frame SO(3) rotation-step convergence threshold.
    double local_frame_convergence_rotation_ = 1e-5;
    /// Local-frame SO(3) translation-step convergence threshold.
    double local_frame_convergence_translation_ = 1e-3;
};

/// \class DCRegDegeneracyAnalysis
///
/// Diagnostic summary for the DCReg normal equation at one ICP linearization.
/// Eigenvalue fields with the ``axis_aligned`` prefix follow the physical
/// x/y/z axis order. Weak-axis fields use 1 for weak and 0 for not weak.
/// For the Open3D legacy ICP estimator, these axes are the target/world-frame
/// incremental axes used by the Open3D left-multiplied SE(3) point-to-plane
/// linearization, not the local body-frame axes used by the standalone DCReg
/// SO(3) examples.
/// If a rank-deficient system prevents Schur-complement factorization, the
/// eigenvalue and weak-axis fields are populated by a block-Hessian eigensolver
/// fallback while schur_factorization_ok_ remains false.
class DCRegDegeneracyAnalysis {
public:
    /// True if at least one correspondence is available.
    bool has_correspondence_ = false;
    /// True if the target point cloud has normals.
    bool has_target_normals_ = false;
    /// True if the full 6x6 normal equation has a numerical null space.
    bool is_rank_deficient_ = false;
    /// True if rotational and translational Schur complements were computed.
    bool schur_factorization_ok_ = false;
    /// True if any Schur condition number exceeds the configured threshold or
    /// the full normal equation is rank deficient.
    bool is_degenerate_ = false;
    /// Condition number of the full 6x6 normal equation.
    double condition_number_full_ = std::numeric_limits<double>::quiet_NaN();
    /// Condition number of the rotational Schur complement.
    double condition_number_rotation_ =
            std::numeric_limits<double>::quiet_NaN();
    /// Condition number of the translational Schur complement.
    double condition_number_translation_ =
            std::numeric_limits<double>::quiet_NaN();
    /// Raw rotational Schur eigenvalues in ascending order, or block-Hessian
    /// fallback eigenvalues if schur_factorization_ok_ is false.
    Eigen::Vector3d schur_eigenvalues_rotation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Raw translational Schur eigenvalues in ascending order, or block-Hessian
    /// fallback eigenvalues if schur_factorization_ok_ is false.
    Eigen::Vector3d schur_eigenvalues_translation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Rotational diagnostic eigenvalues aligned to x/y/z axes.
    Eigen::Vector3d axis_aligned_eigenvalues_rotation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Translational diagnostic eigenvalues aligned to x/y/z axes.
    Eigen::Vector3d axis_aligned_eigenvalues_translation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Clamped rotational eigenvalues used by the preconditioner.
    Eigen::Vector3d clamped_eigenvalues_rotation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Clamped translational eigenvalues used by the preconditioner.
    Eigen::Vector3d clamped_eigenvalues_translation_ =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    /// Weak rotational x/y/z axes.
    Eigen::Vector3i weak_rotation_axes_ = Eigen::Vector3i::Zero();
    /// Weak translational x/y/z axes.
    Eigen::Vector3i weak_translation_axes_ = Eigen::Vector3i::Zero();
    /// Coordinate frame used by weak-axis diagnostics.
    std::string coordinate_frame_ =
            "target/world frame (Open3D left-multiplied SE(3) update)";
    /// Human-readable weak rotational axis list.
    std::string weak_rotation_axes_description_ = "none";
    /// Human-readable weak translational axis list.
    std::string weak_translation_axes_description_ = "none";
    /// Human-readable degeneracy summary.
    std::string degeneracy_description_ = "Degeneracy was not evaluated.";
    /// Solver path selected for the normal equation.
    std::string solver_type_ = "invalid";
    /// True if the PCG path converged before fallback.
    bool pcg_converged_ = false;
    /// Number of PCG iterations executed.
    int pcg_iteration_ = 0;
};

/// \class TransformationEstimationPointToPlaneDCReg
///
/// Degeneracy-aware point-to-plane transformation estimation for ICP.
///
/// This is an Open3D-native legacy CPU implementation of the DCReg linear-solve
/// idea. It uses Open3D's point-to-plane residual model and correspondence
/// pipeline, but solves the 6D normal equation with a Schur-based
/// degeneracy-aware preconditioner. The returned increment follows Open3D's
/// legacy ICP convention and is left-multiplied onto the current pose by
/// RegistrationICP. It does not include the standalone DCReg
/// repository's dataset runners, PCL plane fitting path, or experiment logging.
class TransformationEstimationPointToPlaneDCReg
    : public TransformationEstimationPointToPlane {
public:
    /// \brief Default Constructor.
    TransformationEstimationPointToPlaneDCReg() {}
    ~TransformationEstimationPointToPlaneDCReg() override {}

    /// \brief Constructor that takes DCReg options.
    /// \param option DCReg solver options.
    explicit TransformationEstimationPointToPlaneDCReg(DCRegOption option)
        : option_(std::move(option)) {}

    /// \brief Constructor that takes a RobustKernel.
    /// \param kernel Any of the implemented statistical robust kernels for
    /// outlier rejection.
    explicit TransformationEstimationPointToPlaneDCReg(
            std::shared_ptr<RobustKernel> kernel)
        : TransformationEstimationPointToPlane(std::move(kernel)) {}

    /// \brief Constructor that takes DCReg options and a RobustKernel.
    /// \param option DCReg solver options.
    /// \param kernel Any of the implemented statistical robust kernels for
    /// outlier rejection.
    TransformationEstimationPointToPlaneDCReg(
            DCRegOption option, std::shared_ptr<RobustKernel> kernel)
        : TransformationEstimationPointToPlane(std::move(kernel)),
          option_(std::move(option)) {}

public:
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    /// DCReg solver options.
    DCRegOption option_;
};

/// \brief Compute DCReg degeneracy diagnostics for one point-to-plane ICP
/// linearization.
///
/// \param source Source point cloud.
/// \param target Target point cloud. It must have normals for a valid analysis.
/// \param corres Correspondence set between source and target point cloud.
/// \param option DCReg solver options.
/// \return DCReg degeneracy and solver diagnostics.
DCRegDegeneracyAnalysis ComputeDCRegDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const DCRegOption &option = DCRegOption());

/// \brief Compute DCReg degeneracy diagnostics with a robust kernel.
///
/// \param source Source point cloud.
/// \param target Target point cloud. It must have normals for a valid analysis.
/// \param corres Correspondence set between source and target point cloud.
/// \param option DCReg solver options.
/// \param kernel Robust kernel used to weight point-to-plane residuals.
/// \return DCReg degeneracy and solver diagnostics.
DCRegDegeneracyAnalysis ComputeDCRegDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const DCRegOption &option,
        const RobustKernel &kernel);

/// \brief Compute DCReg diagnostics with the standalone-compatible local-plane
/// residual and local body-frame SO(3) Jacobian.
///
/// This helper does not require target normals. It fits a local plane from the
/// target k-nearest neighbors of each transformed source point, matching the
/// standalone DCReg examples.
DCRegDegeneracyAnalysis ComputeDCRegLocalDegeneracyAnalysis(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation,
        const DCRegOption &option = DCRegOption());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
