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

#include "Python/registration/registration.h"

#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Registration/Feature.h>
#include <Open3D/Registration/CorrespondenceChecker.h>
#include <Open3D/Registration/TransformationEstimation.h>
#include <Open3D/Registration/Registration.h>
#include <Open3D/Registration/FastGlobalRegistration.h>
#include <Open3D/Registration/ColoredICP.h>

using namespace open3d;

template <class TransformationEstimationBase =
                  registration::TransformationEstimation>
class PyTransformationEstimation : public TransformationEstimationBase {
public:
    using TransformationEstimationBase::TransformationEstimationBase;
    registration::TransformationEstimationType GetTransformationEstimationType()
            const override {
        PYBIND11_OVERLOAD_PURE(registration::TransformationEstimationType,
                               TransformationEstimationBase, void);
    }
    double ComputeRMSE(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const registration::CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase, source,
                               target, corres);
    }
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const registration::CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Matrix4d, TransformationEstimationBase,
                               source, target, corres);
    }
};

template <class CorrespondenceCheckerBase = registration::CorrespondenceChecker>
class PyCorrespondenceChecker : public CorrespondenceCheckerBase {
public:
    using CorrespondenceCheckerBase::CorrespondenceCheckerBase;
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const registration::CorrespondenceSet &corres,
               const Eigen::Matrix4d &transformation) const override {
        PYBIND11_OVERLOAD_PURE(bool, CorrespondenceCheckerBase, source, target,
                               corres, transformation);
    }
};

void pybind_registration_classes(py::module &m) {
    py::class_<registration::ICPConvergenceCriteria> convergence_criteria(
            m, "ICPConvergenceCriteria", "ICPConvergenceCriteria");
    py::detail::bind_copy_functions<registration::ICPConvergenceCriteria>(
            convergence_criteria);
    convergence_criteria
            .def(py::init([](double fitness, double rmse, int itr) {
                     return new registration::ICPConvergenceCriteria(fitness,
                                                                     rmse, itr);
                 }),
                 "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
                 "max_iteration"_a = 30)
            .def_readwrite(
                    "relative_fitness",
                    &registration::ICPConvergenceCriteria::relative_fitness_)
            .def_readwrite(
                    "relative_rmse",
                    &registration::ICPConvergenceCriteria::relative_rmse_)
            .def_readwrite(
                    "max_iteration",
                    &registration::ICPConvergenceCriteria::max_iteration_)
            .def("__repr__", [](const registration::ICPConvergenceCriteria &c) {
                return std::string(
                               "registration::ICPConvergenceCriteria class "
                               "with ") +
                       std::string("relative_fitness = ") +
                       std::to_string(c.relative_fitness_) +
                       std::string(", relative_rmse = ") +
                       std::to_string(c.relative_rmse_) +
                       std::string(", and max_iteration = " +
                                   std::to_string(c.max_iteration_));
            });

    py::class_<registration::RANSACConvergenceCriteria> ransac_criteria(
            m, "RANSACConvergenceCriteria", "RANSACConvergenceCriteria");
    py::detail::bind_copy_functions<registration::RANSACConvergenceCriteria>(
            ransac_criteria);
    ransac_criteria
            .def(py::init([](int max_iteration, int max_validation) {
                     return new registration::RANSACConvergenceCriteria(
                             max_iteration, max_validation);
                 }),
                 "max_iteration"_a = 1000, "max_validation"_a = 1000)
            .def_readwrite(
                    "max_iteration",
                    &registration::RANSACConvergenceCriteria::max_iteration_)
            .def_readwrite(
                    "max_validation",
                    &registration::RANSACConvergenceCriteria::max_validation_)
            .def("__repr__",
                 [](const registration::RANSACConvergenceCriteria &c) {
                     return std::string(
                                    "registration::RANSACConvergenceCriteria "
                                    "class with ") +
                            std::string("max_iteration = ") +
                            std::to_string(c.max_iteration_) +
                            std::string(", and max_validation = " +
                                        std::to_string(c.max_validation_));
                 });

    py::class_<
            registration::TransformationEstimation,
            PyTransformationEstimation<registration::TransformationEstimation>>
            te(m, "TransformationEstimation", "TransformationEstimation");
    te.def("compute_rmse", &registration::TransformationEstimation::ComputeRMSE)
            .def("compute_transformation",
                 &registration::TransformationEstimation::
                         ComputeTransformation);

    py::class_<registration::TransformationEstimationPointToPoint,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPoint>,
               registration::TransformationEstimation>
            te_p2p(m, "TransformationEstimationPointToPoint",
                   "TransformationEstimationPointToPoint");
    py::detail::bind_copy_functions<
            registration::TransformationEstimationPointToPoint>(te_p2p);
    te_p2p.def(py::init([](bool with_scaling) {
                   return new registration::
                           TransformationEstimationPointToPoint(with_scaling);
               }),
               "with_scaling"_a = false)
            .def("__repr__",
                 [](const registration::TransformationEstimationPointToPoint
                            &te) {
                     return std::string(
                                    "registration::"
                                    "TransformationEstimationPointToPoint ") +
                            (te.with_scaling_
                                     ? std::string("with scaling.")
                                     : std::string("without scaling."));
                 })
            .def_readwrite("with_scaling",
                           &registration::TransformationEstimationPointToPoint::
                                   with_scaling_);

    py::class_<registration::TransformationEstimationPointToPlane,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPlane>,
               registration::TransformationEstimation>
            te_p2l(m, "TransformationEstimationPointToPlane",
                   "TransformationEstimationPointToPlane");
    py::detail::bind_default_constructor<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    py::detail::bind_copy_functions<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    te_p2l.def(
            "__repr__",
            [](const registration::TransformationEstimationPointToPlane &te) {
                return std::string("TransformationEstimationPointToPlane");
            });

    py::class_<registration::CorrespondenceChecker,
               PyCorrespondenceChecker<registration::CorrespondenceChecker>>
            cc(m, "CorrespondenceChecker", "CorrespondenceChecker");
    cc.def("Check", &registration::CorrespondenceChecker::Check);

    py::class_<registration::CorrespondenceCheckerBasedOnEdgeLength,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnEdgeLength>,
               registration::CorrespondenceChecker>
            cc_el(m, "CorrespondenceCheckerBasedOnEdgeLength",
                  "CorrespondenceCheckerBasedOnEdgeLength");
    py::detail::bind_copy_functions<
            registration::CorrespondenceCheckerBasedOnEdgeLength>(cc_el);
    cc_el.def(py::init([](double similarity_threshold) {
                  return new registration::
                          CorrespondenceCheckerBasedOnEdgeLength(
                                  similarity_threshold);
              }),
              "similarity_threshold"_a = 0.9)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnEdgeLength
                            &c) {
                     return std::string(
                                    "registration::"
                                    "CorrespondenceCheckerBasedOnEdgeLength "
                                    "with similarity threshold ") +
                            std::to_string(c.similarity_threshold_);
                 })
            .def_readwrite(
                    "similarity_threshold",
                    &registration::CorrespondenceCheckerBasedOnEdgeLength::
                            similarity_threshold_);

    py::class_<registration::CorrespondenceCheckerBasedOnDistance,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnDistance>,
               registration::CorrespondenceChecker>
            cc_d(m, "CorrespondenceCheckerBasedOnDistance",
                 "CorrespondenceCheckerBasedOnDistance");
    py::detail::bind_copy_functions<
            registration::CorrespondenceCheckerBasedOnDistance>(cc_d);
    cc_d.def(py::init([](double distance_threshold) {
                 return new registration::CorrespondenceCheckerBasedOnDistance(
                         distance_threshold);
             }),
             "distance_threshold"_a)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnDistance
                            &c) {
                     return std::string(
                                    "registration::"
                                    "CorrespondenceCheckerBasedOnDistance with "
                                    "distance threshold ") +
                            std::to_string(c.distance_threshold_);
                 })
            .def_readwrite("distance_threshold",
                           &registration::CorrespondenceCheckerBasedOnDistance::
                                   distance_threshold_);

    py::class_<registration::CorrespondenceCheckerBasedOnNormal,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnNormal>,
               registration::CorrespondenceChecker>
            cc_n(m, "CorrespondenceCheckerBasedOnNormal",
                 "CorrespondenceCheckerBasedOnNormal");
    py::detail::bind_copy_functions<
            registration::CorrespondenceCheckerBasedOnNormal>(cc_n);
    cc_n.def(py::init([](double normal_angle_threshold) {
                 return new registration::CorrespondenceCheckerBasedOnNormal(
                         normal_angle_threshold);
             }),
             "normal_angle_threshold"_a)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnNormal &c) {
                     return std::string(
                                    "registration::"
                                    "CorrespondenceCheckerBasedOnNormal with "
                                    "normal threshold ") +
                            std::to_string(c.normal_angle_threshold_);
                 })
            .def_readwrite("normal_angle_threshold",
                           &registration::CorrespondenceCheckerBasedOnNormal::
                                   normal_angle_threshold_);

    py::class_<registration::FastGlobalRegistrationOption> fgr_option(
            m, "FastGlobalRegistrationOption", "FastGlobalRegistrationOption");
    py::detail::bind_copy_functions<registration::FastGlobalRegistrationOption>(
            fgr_option);
    fgr_option
            .def(py::init([](double division_factor, bool use_absolute_scale,
                             bool decrease_mu,
                             double maximum_correspondence_distance,
                             int iteration_number, double tuple_scale,
                             int maximum_tuple_count) {
                     return new registration::FastGlobalRegistrationOption(
                             division_factor, use_absolute_scale, decrease_mu,
                             maximum_correspondence_distance, iteration_number,
                             tuple_scale, maximum_tuple_count);
                 }),
                 "division_factor"_a = 1.4, "use_absolute_scale"_a = false,
                 "decrease_mu"_a = false,
                 "maximum_correspondence_distance"_a = 0.025,
                 "iteration_number"_a = 64, "tuple_scale"_a = 0.95,
                 "maximum_tuple_count"_a = 1000)
            .def_readwrite("division_factor",
                           &registration::FastGlobalRegistrationOption::
                                   division_factor_)
            .def_readwrite("use_absolute_scale",
                           &registration::FastGlobalRegistrationOption::
                                   use_absolute_scale_)
            .def_readwrite(
                    "decrease_mu",
                    &registration::FastGlobalRegistrationOption::decrease_mu_)
            .def_readwrite("maximum_correspondence_distance",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_correspondence_distance_)
            .def_readwrite("iteration_number",
                           &registration::FastGlobalRegistrationOption::
                                   iteration_number_)
            .def_readwrite(
                    "tuple_scale",
                    &registration::FastGlobalRegistrationOption::tuple_scale_)
            .def_readwrite("maximum_tuple_count",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_tuple_count_)
            .def("__repr__",
                 [](const registration::FastGlobalRegistrationOption &c) {
                     return std::string(
                                    "registration::"
                                    "FastGlobalRegistrationOption class "
                                    "with ") +
                            std::string("\ndivision_factor = ") +
                            std::to_string(c.division_factor_) +
                            std::string("\nuse_absolute_scale = ") +
                            std::to_string(c.use_absolute_scale_) +
                            std::string("\ndecrease_mu = ") +
                            std::to_string(c.decrease_mu_) +
                            std::string(
                                    "\nmaximum_correspondence_distance = ") +
                            std::to_string(c.maximum_correspondence_distance_) +
                            std::string("\niteration_number = ") +
                            std::to_string(c.iteration_number_) +
                            std::string("\ntuple_scale = ") +
                            std::to_string(c.tuple_scale_) +
                            std::string("\nmaximum_tuple_count = ") +
                            std::to_string(c.maximum_tuple_count_);
                 });

    py::class_<registration::RegistrationResult> registration_result(
            m, "RegistrationResult", "RegistrationResult");
    py::detail::bind_default_constructor<registration::RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<registration::RegistrationResult>(
            registration_result);
    registration_result
            .def_readwrite("transformation",
                           &registration::RegistrationResult::transformation_)
            .def_readwrite(
                    "correspondence_set",
                    &registration::RegistrationResult::correspondence_set_)
            .def_readwrite("inlier_rmse",
                           &registration::RegistrationResult::inlier_rmse_)
            .def_readwrite("fitness",
                           &registration::RegistrationResult::fitness_)
            .def("__repr__", [](const registration::RegistrationResult &rr) {
                return std::string(
                               "registration::RegistrationResult with fitness "
                               "= ") +
                       std::to_string(rr.fitness_) +
                       std::string(", inlier_rmse = ") +
                       std::to_string(rr.inlier_rmse_) +
                       std::string(", and correspondence_set size of ") +
                       std::to_string(rr.correspondence_set_.size()) +
                       std::string("\nAccess transformation to get result.");
            });
}

void pybind_registration_methods(py::module &m) {
    m.def("evaluate_registration", &registration::EvaluateRegistration,
          "Function for evaluating registration between point clouds",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a = Eigen::Matrix4d::Identity(),
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false));
    m.def("registration_icp", &registration::RegistrationICP,
          "Function for ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4d::Identity(),
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false),
          "criteria"_a = registration::ICPConvergenceCriteria());
    m.def("registration_colored_icp", &registration::RegistrationColoredICP,
          "Function for Colored ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4d::Identity(),
          "criteria"_a = registration::ICPConvergenceCriteria(),
          "lambda_geometric"_a = 0.968);
    m.def("registration_ransac_based_on_correspondence",
          &registration::RegistrationRANSACBasedOnCorrespondence,
          "Function for global RANSAC registration based on a set of "
          "correspondences",
          "source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false),
          "ransac_n"_a = 6,
          "criteria"_a = registration::RANSACConvergenceCriteria());
    m.def("registration_ransac_based_on_feature_matching",
          &registration::RegistrationRANSACBasedOnFeatureMatching,
          "Function for global RANSAC registration based on feature matching",
          "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
          "max_correspondence_distance"_a,
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false),
          "ransac_n"_a = 4,
          "checkers"_a = std::vector<std::reference_wrapper<
                  const registration::CorrespondenceChecker>>(),
          "criteria"_a = registration::RANSACConvergenceCriteria(100000, 100));
    m.def("registration_fast_based_on_feature_matching",
          &registration::FastGlobalRegistration,
          "Function for fast global registration based on feature matching",
          "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
          "option"_a = registration::FastGlobalRegistrationOption());
    m.def("get_information_matrix_from_point_clouds",
          &registration::GetInformationMatrixFromPointClouds,
          "Function for computing information matrix from "
          "registration::RegistrationResult",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation_result"_a,
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false));
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule("registration");
    pybind_registration_classes(m_submodule);
    pybind_registration_methods(m_submodule);

    pybind_feature(m_submodule);
    pybind_feature_methods(m_submodule);
    pybind_global_optimization(m_submodule);
    pybind_global_optimization_methods(m_submodule);
}
