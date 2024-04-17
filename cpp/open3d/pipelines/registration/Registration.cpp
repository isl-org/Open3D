// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/Registration.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace pipelines {
namespace registration {

struct RegistrationReduction {
    // Globals:
    const geometry::KDTreeFlann& target_kdtree;
    const geometry::PointCloud& source;
    double max_distance;
    // Locals:
    CorrespondenceSet correspondences;
    double error2;

    RegistrationReduction(const geometry::KDTreeFlann& target_kdtree_,
                          const geometry::PointCloud& source_,
                          double max_correspondence_distance_)
        : target_kdtree(target_kdtree_),
          source(source_),
          max_distance(max_correspondence_distance_),
          correspondences(),
          error2(0.0) {}

    RegistrationReduction(RegistrationReduction& other, tbb::split)
        : target_kdtree(other.target_kdtree),
          source(other.source),
          max_distance(other.max_distance),
          correspondences(),
          error2(0.0) {}

    void operator()(const tbb::blocked_range<std::size_t>& range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto& point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_distance, 1, indices,
                                           dists) > 0) {
                error2 += dists[0];
                correspondences.emplace_back(i, indices[0]);
            }
        }
    }

    void join(RegistrationReduction& other) {
        correspondences.insert(correspondences.end(),
                               other.correspondences.begin(),
                               other.correspondences.end());
        error2 += other.error2;
    }

    std::tuple<CorrespondenceSet, double> as_tuple() && {
        return {std::move(correspondences), error2};
    }
};

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const geometry::KDTreeFlann& target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d& transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    RegistrationReduction reducer(target_kdtree, source,
                                  max_correspondence_distance);
    tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, source.points_.size(),
                                            utility::DefaultGrainSizeTBB()),
            reducer);

    double error2;
    std::tie(result.correspondence_set_, error2) =
            std::move(reducer).as_tuple();

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double)corres_number / (double)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return result;
}

static double EvaluateInlierCorrespondenceRatio(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const CorrespondenceSet& corres,
        double max_correspondence_distance,
        const Eigen::Matrix4d& transformation) {
    RegistrationResult result(transformation);

    int inlier_corres = 0;
    double max_dis2 = max_correspondence_distance * max_correspondence_distance;
    for (const auto& c : corres) {
        double dis2 =
                (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
        if (dis2 < max_dis2) {
            inlier_corres++;
        }
    }

    return double(inlier_corres) / double(corres.size());
}

RegistrationResult EvaluateRegistration(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        double max_correspondence_distance,
        const Eigen::Matrix4d&
                transformation /* = Eigen::Matrix4d::Identity()*/) {
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!transformation.isIdentity()) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
}

RegistrationResult RegistrationICP(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        double max_correspondence_distance,
        const Eigen::Matrix4d& init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation& estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria&
                criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }
    if ((estimation.GetTransformationEstimationType() ==
                 TransformationEstimationType::PointToPlane ||
         estimation.GetTransformationEstimationType() ==
                 TransformationEstimationType::ColoredICP) &&
        (!target.HasNormals())) {
        utility::LogError(
                "TransformationEstimationPointToPlane and "
                "TransformationEstimationColoredICP "
                "require pre-computed normal vectors for target PointCloud.");
    }
    if ((estimation.GetTransformationEstimationType() ==
         TransformationEstimationType::GeneralizedICP) &&
        (!target.HasCovariances() || !source.HasCovariances())) {
        utility::LogError(
                "TransformationEstimationForGeneralizedICP require "
                "pre-computed per point covariances matrices for source and "
                "target PointCloud.");
    }

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}

template <typename T>
void atomic_min(std::atomic<T>& min_val, const T& val) noexcept {
    T prev_val = min_val;
    while (prev_val > val && min_val.compare_exchange_weak(prev_val, val))
        ;
}

struct RANSACCorrespondenceReduction {
    // Globals
    const geometry::PointCloud& source;
    const geometry::PointCloud& target;
    const CorrespondenceSet& corres;
    const TransformationEstimation& estimation;
    using CheckerType =
            std::vector<std::reference_wrapper<const CorrespondenceChecker>>;
    const CheckerType& checkers;
    const geometry::KDTreeFlann& kdtree;
    std::atomic<int>& est_k_global;
    std::atomic<int>& total_validation;
    using RandomIntGen = utility::random::UniformIntGenerator<int>;
    RandomIntGen& rand_gen;
    // Constants
    const double max_distance;
    const int ransac_n;
    const double log_confidence;
    // Locals
    CorrespondenceSet ransac_corres;
    RegistrationResult best_result;

    RANSACCorrespondenceReduction(const geometry::PointCloud& source_,
                                  const geometry::PointCloud& target_,
                                  const CorrespondenceSet& corres_,
                                  const TransformationEstimation& estimation_,
                                  const CheckerType& checkers_,
                                  const geometry::KDTreeFlann& kdtree_,
                                  std::atomic<int>& est_k_global_,
                                  std::atomic<int>& total_validation_,
                                  RandomIntGen& rand_gen_,
                                  double max_dist_,
                                  int ransac_n_,
                                  double confidence)
        : source(source_),
          target(target_),
          corres(corres_),
          estimation(estimation_),
          checkers(checkers_),
          kdtree(kdtree_),
          est_k_global(est_k_global_),
          total_validation(total_validation_),
          rand_gen(rand_gen_),
          max_distance(max_dist_),
          ransac_n(ransac_n_),
          log_confidence(std::log(1.0 - confidence)),
          ransac_corres(ransac_n) {}

    RANSACCorrespondenceReduction(RANSACCorrespondenceReduction& o, tbb::split)
        : source(o.source),
          target(o.target),
          corres(o.corres),
          estimation(o.estimation),
          checkers(o.checkers),
          kdtree(o.kdtree),
          est_k_global(o.est_k_global),
          total_validation(o.total_validation),
          rand_gen(o.rand_gen),
          max_distance(o.max_distance),
          ransac_n(o.ransac_n),
          log_confidence(o.log_confidence) {}

    void operator()(const tbb::blocked_range<int>& range) {
        int est_k_local = est_k_global;
        for (int i = range.begin(); i < range.end(); ++i) {
            if (i < est_k_global) {
                for (int j = 0; j < ransac_n; j++) {
                    ransac_corres[j] = corres[rand_gen()];
                }

                Eigen::Matrix4d transformation =
                        estimation.ComputeTransformation(source, target,
                                                         ransac_corres);

                // Check transformation: inexpensive
                if (!std::all_of(checkers.begin(), checkers.end(),
                                 [&](const auto& checker) {
                                     return checker.get().Check(source, target,
                                                                ransac_corres,
                                                                transformation);
                                 })) {
                    continue;
                }

                // Expensive validation
                geometry::PointCloud pcd = source;
                pcd.Transform(transformation);
                auto result = GetRegistrationResultAndCorrespondences(
                        pcd, target, kdtree, max_distance, transformation);

                if (result.IsBetterRANSACThan(best_result)) {
                    best_result = std::move(result);

                    double corres_inlier_ratio =
                            EvaluateInlierCorrespondenceRatio(
                                    pcd, target, corres, max_distance,
                                    transformation);

                    // Update exit condition if necessary.
                    // If confidence is 1.0, then it is safely inf, we always
                    // consume all the iterations.
                    double est_k_local_d = log_confidence /
                        std::log(1.0 - std::pow(corres_inlier_ratio, ransac_n));
                    if (est_k_local_d < 0) {
                        est_k_local_d = est_k_local;
                    }
                    if (est_k_local_d < est_k_global) {
                        est_k_local = std::ceil(est_k_local_d);
                    }
                    utility::LogDebug(
                            "Thread {:06d}: registration fitness={:.3f}, "
                            "corres inlier ratio={:.3f}, Est. max k = {}",
                            i, best_result.fitness_, corres_inlier_ratio,
                            est_k_local_d);
                }
                total_validation += 1;
                atomic_min(est_k_global, est_k_local);
            }
        }
    }

    void join(RANSACCorrespondenceReduction& other) {
        if (!best_result.IsBetterRANSACThan(other.best_result)) {
            best_result = std::move(other.best_result);
        }
    }
};

RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const CorrespondenceSet& corres,
        double max_correspondence_distance,
        const TransformationEstimation& estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 3*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>&
                checkers /* = {}*/,
        const RANSACConvergenceCriteria& criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || (int)corres.size() < ransac_n ||
        max_correspondence_distance <= 0.0) {
        return {};
    }

    geometry::KDTreeFlann kdtree(target);
    std::atomic<int> est_k_global = criteria.max_iteration_;
    std::atomic<int> total_validation = 0;
    utility::random::UniformIntGenerator<int> rand_gen(0, corres.size() - 1);
    RANSACCorrespondenceReduction reducer(
            source, target, corres, estimation, checkers, kdtree, est_k_global,
            total_validation, rand_gen, max_correspondence_distance, ransac_n,
            criteria.confidence_);
    tbb::parallel_reduce(
            tbb::blocked_range<int>(0, criteria.max_iteration_,
                                    utility::DefaultGrainSizeTBB()),
            reducer);
    auto best_result = std::move(reducer.best_result);
    utility::LogDebug(
            "RANSAC exits after {:d} validations. Best inlier ratio {:e}, "
            "RMSE {:e}",
            total_validation.load(), best_result.fitness_,
            best_result.inlier_rmse_);
    return best_result;
}

RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const Feature& source_features,
        const Feature& target_features,
        bool mutual_filter,
        double max_correspondence_distance,
        const TransformationEstimation&
                estimation /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 3*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>&
                checkers /* = {}*/,
        const RANSACConvergenceCriteria&
                criteria /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    CorrespondenceSet corres = CorrespondencesFromFeatures(
            source_features, target_features, mutual_filter);

    return RegistrationRANSACBasedOnCorrespondence(
            source, target, corres, max_correspondence_distance, estimation,
            ransac_n, checkers, criteria);
}

struct InformationMatrixReducer {
    // Globals
    const CorrespondenceSet& corres;
    const geometry::PointCloud& target;
    // Locals
    Eigen::Matrix6d GTG;

    InformationMatrixReducer(const CorrespondenceSet& corres_,
                             const geometry::PointCloud& target_)
        : corres(corres_), target(target_), GTG(Eigen::Matrix6d::Zero()) {}

    InformationMatrixReducer(InformationMatrixReducer& o, tbb::split)
        : corres(o.corres), target(o.target), GTG(Eigen::Matrix6d::Zero()) {}

    void operator()(const tbb::blocked_range<std::size_t>& range) {
        // write q^*
        // see http://redwood-data.org/indoor/registration.html
        // note: I comes first in this implementation
        Eigen::Vector6d G_r;
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            int t = corres[i](1);
            double x = target.points_[t](0);
            double y = target.points_[t](1);
            double z = target.points_[t](2);
            G_r.setZero();
            G_r(1) = z;
            G_r(2) = -y;
            G_r(3) = 1.0;
            GTG.noalias() += G_r * G_r.transpose();
            G_r.setZero();
            G_r(0) = -z;
            G_r(2) = x;
            G_r(4) = 1.0;
            GTG.noalias() += G_r * G_r.transpose();
            G_r.setZero();
            G_r(0) = y;
            G_r(1) = -x;
            G_r(5) = 1.0;
            GTG.noalias() += G_r * G_r.transpose();
        }
    }

    void join(InformationMatrixReducer& other) { GTG += other.GTG; }
};

Eigen::Matrix6d GetInformationMatrixFromPointClouds(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        double max_correspondence_distance,
        const Eigen::Matrix4d& transformation) {
    geometry::PointCloud pcd = source;
    if (!transformation.isIdentity()) {
        pcd.Transform(transformation);
    }
    RegistrationResult result;
    geometry::KDTreeFlann target_kdtree(target);
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, target_kdtree, max_correspondence_distance,
            transformation);

    InformationMatrixReducer reducer(result.correspondence_set_, target);
    tbb::parallel_reduce(tbb::blocked_range<std::size_t>(
                                 0, result.correspondence_set_.size(),
                                 utility::DefaultGrainSizeTBB()),
                         reducer);
    return std::move(reducer.GTG);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
