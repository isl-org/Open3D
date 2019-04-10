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

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Registration/ColoredICP.h"
#include "Open3D/Registration/CorrespondenceChecker.h"
#include "Open3D/Registration/FastGlobalRegistration.h"
#include "Open3D/Registration/Feature.h"
#include "Open3D/Registration/Registration.h"
#include "Open3D/Registration/TransformationEstimation.h"
#include "Python/docstring.h"

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
    // ope3dn.registration.ICPConvergenceCriteria
    py::class_<registration::ICPConvergenceCriteria> convergence_criteria(
            m, "ICPConvergenceCriteria",
            "Class that defines the convergence criteria of ICP. ICP algorithm "
            "stops if the relative change of fitness and rmse hit "
            "``relative_fitness`` and ``relative_rmse`` individually, or the "
            "iteration number exceeds ``max_iteration``.");
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
                    &registration::ICPConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def_readwrite(
                    "relative_rmse",
                    &registration::ICPConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inliner RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def_readwrite(
                    "max_iteration",
                    &registration::ICPConvergenceCriteria::max_iteration_,
                    "Maximum iteration before iteration stops.")
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

    // ope3dn.registration.RANSACConvergenceCriteria
    py::class_<registration::RANSACConvergenceCriteria> ransac_criteria(
            m, "RANSACConvergenceCriteria",
            "Class that defines the convergence criteria of RANSAC. RANSAC "
            "algorithm stops if the iteration number hits ``max_iteration``, "
            "or the validation has been run for ``max_validation`` times. Note "
            "that the validation is the most computational expensive operator "
            "in an iteration. Most iterations do not do full validation. It is "
            "crucial to control ``max_validation`` so that the computation "
            "time is acceptable.");
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
                    &registration::RANSACConvergenceCriteria::max_iteration_,
                    "Maximum iteration before iteration stops.")
            .def_readwrite(
                    "max_validation",
                    &registration::RANSACConvergenceCriteria::max_validation_,
                    "Maximum times the validation has been run before the "
                    "iteration stops.")
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

    // ope3dn.registration.TransformationEstimation
    py::class_<
            registration::TransformationEstimation,
            PyTransformationEstimation<registration::TransformationEstimation>>
            te(m, "TransformationEstimation",
               "Base class that estimates a transformation between two point "
               "clouds. The virtual function ComputeTransformation() must be "
               "implemented in subclasses.");
    te.def("compute_rmse", &registration::TransformationEstimation::ComputeRMSE,
           "source"_a, "target"_a, "corres"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &registration::TransformationEstimation::ComputeTransformation,
           "source"_a, "target"_a, "corres"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");
    docstring::ClassMethodDocInject(
            m, "TransformationEstimation", "compute_rmse",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."}});
    docstring::ClassMethodDocInject(
            m, "TransformationEstimation", "compute_transformation",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."}});

    // ope3dn.registration.TransformationEstimationPointToPoint:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationPointToPoint,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPoint>,
               registration::TransformationEstimation>
            te_p2p(m, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to point "
                   "distance.");
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
            .def_readwrite(
                    "with_scaling",
                    &registration::TransformationEstimationPointToPoint::
                            with_scaling_,
                    R"(Set to ``True`` to estimate scaling, ``False`` to force
scaling to be ``1``.

The homogeneous transformation is given by

:math:`T = \begin{bmatrix} c\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}`

Sets :math:`c = 1` if ``with_scaling`` is ``False``.
)");

    // ope3dn.registration.TransformationEstimationPointToPlane:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationPointToPlane,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPlane>,
               registration::TransformationEstimation>
            te_p2l(m, "TransformationEstimationPointToPlane",
                   "Class to estimate a transformation for point to plane "
                   "distance.");
    py::detail::bind_default_constructor<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    py::detail::bind_copy_functions<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    te_p2l.def(
            "__repr__",
            [](const registration::TransformationEstimationPointToPlane &te) {
                return std::string("TransformationEstimationPointToPlane");
            });

    // ope3dn.registration.CorrespondenceChecker
    py::class_<registration::CorrespondenceChecker,
               PyCorrespondenceChecker<registration::CorrespondenceChecker>>
            cc(m, "CorrespondenceChecker",
               "Base class that checks if two (small) point clouds can be "
               "aligned. This class is used in feature based matching "
               "algorithms (such as RANSAC and FastGlobalRegistration) to "
               "prune out outlier correspondences. The virtual function "
               "Check() must be implemented in subclasses.");
    cc.def("Check", &registration::CorrespondenceChecker::Check, "source"_a,
           "target"_a, "corres"_a, "transformation"_a,
           "Function to check if two points can be aligned. The two input "
           "point clouds must have exact the same number of points.");
    docstring::ClassMethodDocInject(
            m, "CorrespondenceChecker", "Check",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."},
             {"transformation", "The estimated transformation (inplace)."}});

    // ope3dn.registration.CorrespondenceCheckerBasedOnEdgeLength:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnEdgeLength,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnEdgeLength>,
               registration::CorrespondenceChecker>
            cc_el(m, "CorrespondenceCheckerBasedOnEdgeLength",
                  "Check if two point clouds build the polygons with similar "
                  "edge lengths. That is, checks if the lengths of any two "
                  "arbitrary edges (line formed by two vertices) individually "
                  "drawn withinin source point cloud and within the target "
                  "point cloud with correspondences are similar. The only "
                  "parameter similarity_threshold is a number between 0 "
                  "(loose) and 1 (strict)");
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
                            similarity_threshold_,
                    R"(float value between 0 (loose) and 1 (strict): For the
check to be true,

:math:`||\text{edge}_{\text{source}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{target}}||` and

:math:`||\text{edge}_{\text{target}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{source}}||`

must hold true for all edges.)");

    // ope3dn.registration.CorrespondenceCheckerBasedOnDistance:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnDistance,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnDistance>,
               registration::CorrespondenceChecker>
            cc_d(m, "CorrespondenceCheckerBasedOnDistance",
                 "Class to check if aligned point clouds are close (less than "
                 "specified threshold).");
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
                                   distance_threshold_,
                           "Distance threashold for the check.");

    // ope3dn.registration.CorrespondenceCheckerBasedOnNormal:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnNormal,
               PyCorrespondenceChecker<
                       registration::CorrespondenceCheckerBasedOnNormal>,
               registration::CorrespondenceChecker>
            cc_n(m, "CorrespondenceCheckerBasedOnNormal",
                 "Class to check if two aligned point clouds have similar "
                 "normals. It considers vertex normal affinity of any "
                 "correspondences. It computes dot product of two normal "
                 "vectors. It takes radian value for the threshold.");
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
                                   normal_angle_threshold_,
                           "Radian value for angle threshold.");

    // ope3dn.registration.FastGlobalRegistrationOption:
    py::class_<registration::FastGlobalRegistrationOption> fgr_option(
            m, "FastGlobalRegistrationOption",
            "Options for FastGlobalRegistration.");
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
            .def_readwrite(
                    "division_factor",
                    &registration::FastGlobalRegistrationOption::
                            division_factor_,
                    "float: Division factor used for graduated non-convexity.")
            .def_readwrite(
                    "use_absolute_scale",
                    &registration::FastGlobalRegistrationOption::
                            use_absolute_scale_,
                    "bool: Measure distance in absolute scale (1) or in scale "
                    "relative to the diameter of the model (0).")
            .def_readwrite(
                    "decrease_mu",
                    &registration::FastGlobalRegistrationOption::decrease_mu_,
                    "bool: Set to ``True`` to decrease scale mu by "
                    "``division_factor`` for graduated non-convexity.")
            .def_readwrite("maximum_correspondence_distance",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_correspondence_distance_,
                           "float: Maximum correspondence distance.")
            .def_readwrite("iteration_number",
                           &registration::FastGlobalRegistrationOption::
                                   iteration_number_,
                           "int: Maximum number of iterations.")
            .def_readwrite(
                    "tuple_scale",
                    &registration::FastGlobalRegistrationOption::tuple_scale_,
                    "float: Similarity measure used for tuples of feature "
                    "points.")
            .def_readwrite("maximum_tuple_count",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_tuple_count_,
                           "float: Maximum tuple numbers.")
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

    // ope3dn.registration.RegistrationResult
    py::class_<registration::RegistrationResult> registration_result(
            m, "RegistrationResult",
            "Class that contains the registration results.");
    py::detail::bind_default_constructor<registration::RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<registration::RegistrationResult>(
            registration_result);
    registration_result
            .def_readwrite("transformation",
                           &registration::RegistrationResult::transformation_,
                           "``4 x 4`` float64 numpy array: The estimated "
                           "transformation matrix.")
            .def_readwrite(
                    "correspondence_set",
                    &registration::RegistrationResult::correspondence_set_,
                    "``n x 2`` int numpy array: Correspondence set between "
                    "source and target point cloud.")
            .def_readwrite("inlier_rmse",
                           &registration::RegistrationResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &registration::RegistrationResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in target). Higher is better.")
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

// Registration functions have similar arguments, sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"checkers", "checkers"},
                {"corres",
                 "Checker class to check if two point clouds can be "
                 "aligned. "
                 "One of "
                 "(``registration::CorrespondenceCheckerBasedOnEdgeLength``, "
                 "``registration::CorrespondenceCheckerBasedOnDistance``, "
                 "``registration::CorrespondenceCheckerBasedOnNormal``)"},
                {"criteria", "Convergence criteria"},
                {"estimation_method",
                 "Estimation method. One of "
                 "(``registration::TransformationEstimationPointToPoint``, "
                 "``registration::TransformationEstimationPointToPlane``)"},
                {"init", "Initial transformation estimation"},
                {"lambda_geometric", "lambda_geometric value"},
                {"max_correspondence_distance",
                 "Maximum correspondence points-pair distance."},
                {"option", "Registration option"},
                {"ransac_n", "Fit ransac with ``ransac_n`` correspondences"},
                {"source_feature", "Source point cloud feature."},
                {"source", "The source point cloud."},
                {"target_feature", "Target point cloud feature."},
                {"target", "The target point cloud."},
                {"transformation",
                 "The 4x4 transformation matrix to transform ``source`` to "
                 "``target``"}};

void pybind_registration_methods(py::module &m) {
    m.def("evaluate_registration", &registration::EvaluateRegistration,
          "Function for evaluating registration between point clouds",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a = Eigen::Matrix4d::Identity());
    docstring::FunctionDocInject(m, "evaluate_registration",
                                 map_shared_argument_docstrings);

    m.def("registration_icp", &registration::RegistrationICP,
          "Function for ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4d::Identity(),
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false),
          "criteria"_a = registration::ICPConvergenceCriteria());
    docstring::FunctionDocInject(m, "registration_icp",
                                 map_shared_argument_docstrings);

    m.def("registration_colored_icp", &registration::RegistrationColoredICP,
          "Function for Colored ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4d::Identity(),
          "criteria"_a = registration::ICPConvergenceCriteria(),
          "lambda_geometric"_a = 0.968);
    docstring::FunctionDocInject(m, "registration_colored_icp",
                                 map_shared_argument_docstrings);

    m.def("registration_ransac_based_on_correspondence",
          &registration::RegistrationRANSACBasedOnCorrespondence,
          "Function for global RANSAC registration based on a set of "
          "correspondences",
          "source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(false),
          "ransac_n"_a = 6,
          "criteria"_a = registration::RANSACConvergenceCriteria());
    docstring::FunctionDocInject(m,
                                 "registration_ransac_based_on_correspondence",
                                 map_shared_argument_docstrings);

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
    docstring::FunctionDocInject(
            m, "registration_ransac_based_on_feature_matching",
            map_shared_argument_docstrings);

    m.def("registration_fast_based_on_feature_matching",
          &registration::FastGlobalRegistration,
          "Function for fast global registration based on feature matching",
          "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
          "option"_a = registration::FastGlobalRegistrationOption());
    docstring::FunctionDocInject(m,
                                 "registration_fast_based_on_feature_matching",
                                 map_shared_argument_docstrings);

    m.def("get_information_matrix_from_point_clouds",
          &registration::GetInformationMatrixFromPointClouds,
          "Function for computing information matrix from transformation "
          "matrix",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a);
    docstring::FunctionDocInject(m, "get_information_matrix_from_point_clouds",
                                 map_shared_argument_docstrings);
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
