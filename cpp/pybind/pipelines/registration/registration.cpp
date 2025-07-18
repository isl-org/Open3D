// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/Registration.h"

#include <memory>
#include <utility>

#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/ColoredICP.h"
#include "open3d/pipelines/registration/CorrespondenceChecker.h"
#include "open3d/pipelines/registration/FastGlobalRegistration.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/pipelines/registration/GeneralizedICP.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/pipelines/registration/registration.h"

namespace open3d {
namespace pipelines {
namespace registration {

template <class TransformationEstimationBase = TransformationEstimation>
class PyTransformationEstimation : public TransformationEstimationBase {
public:
    using TransformationEstimationBase::TransformationEstimationBase;
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        PYBIND11_OVERLOAD_PURE(TransformationEstimationType,
                               TransformationEstimationBase, void);
    }
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase, source,
                               target, corres);
    }
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Matrix4d, TransformationEstimationBase,
                               source, target, corres);
    }
};

template <class CorrespondenceCheckerBase = CorrespondenceChecker>
class PyCorrespondenceChecker : public CorrespondenceCheckerBase {
public:
    using CorrespondenceCheckerBase::CorrespondenceCheckerBase;
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const CorrespondenceSet &corres,
               const Eigen::Matrix4d &transformation) const override {
        PYBIND11_OVERLOAD_PURE(bool, CorrespondenceCheckerBase, source, target,
                               corres, transformation);
    }
};

void pybind_registration_declarations(py::module &m) {
    py::module m_registration =
            m.def_submodule("registration", "Registration pipeline.");
    py::class_<ICPConvergenceCriteria> convergence_criteria(
            m_registration, "ICPConvergenceCriteria",
            "Class that defines the convergence criteria of ICP. ICP "
            "algorithm "
            "stops if the relative change of fitness and rmse hit "
            "``relative_fitness`` and ``relative_rmse`` individually, "
            "or the "
            "iteration number exceeds ``max_iteration``.");
    py::class_<RANSACConvergenceCriteria> ransac_criteria(
            m_registration, "RANSACConvergenceCriteria",
            "Class that defines the convergence criteria of "
            "RANSAC. RANSAC algorithm stops if the iteration "
            "number hits ``max_iteration``, or the fitness "
            "measured during validation suggests that the "
            "algorithm can be terminated early with some "
            "``confidence``. Early termination takes place "
            "when the number of iterations reaches ``k = "
            "log(1 - confidence)/log(1 - fitness^{ransac_n})``, "
            "where ``ransac_n`` is the number of points used "
            "during a ransac iteration. Use confidence=1.0 "
            "to avoid early termination.");
    py::class_<TransformationEstimation,
               PyTransformationEstimation<TransformationEstimation>>
            te(m_registration, "TransformationEstimation",
               "Base class that estimates a transformation between two point "
               "clouds. The virtual function ComputeTransformation() must be "
               "implemented in subclasses.");
    py::class_<TransformationEstimationPointToPoint,
               PyTransformationEstimation<TransformationEstimationPointToPoint>,
               TransformationEstimation>
            te_p2p(m_registration, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to point "
                   "distance.");
    py::class_<TransformationEstimationPointToPlane,
               PyTransformationEstimation<TransformationEstimationPointToPlane>,
               TransformationEstimation>
            te_p2l(m_registration, "TransformationEstimationPointToPlane",
                   "Class to estimate a transformation for point to plane "
                   "distance.");
    py::class_<
            TransformationEstimationForColoredICP,
            PyTransformationEstimation<TransformationEstimationForColoredICP>,
            TransformationEstimation>
            te_col(m_registration, "TransformationEstimationForColoredICP",
                   "Class to estimate a transformation between two point "
                   "clouds using color information");
    py::class_<TransformationEstimationForGeneralizedICP,
               PyTransformationEstimation<
                       TransformationEstimationForGeneralizedICP>,
               TransformationEstimation>
            te_gicp(m_registration, "TransformationEstimationForGeneralizedICP",
                    "Class to estimate a transformation for Generalized ICP.");
    py::class_<CorrespondenceChecker,
               PyCorrespondenceChecker<CorrespondenceChecker>>
            cc(m_registration, "CorrespondenceChecker",
               "Base class that checks if two (small) point clouds can be "
               "aligned. This class is used in feature based matching "
               "algorithms (such as RANSAC and FastGlobalRegistration) to "
               "prune out outlier correspondences. The virtual function "
               "Check() must be implemented in subclasses.");
    py::class_<CorrespondenceCheckerBasedOnEdgeLength,
               PyCorrespondenceChecker<CorrespondenceCheckerBasedOnEdgeLength>,
               CorrespondenceChecker>
            cc_el(m_registration, "CorrespondenceCheckerBasedOnEdgeLength",
                  "Check if two point clouds build the polygons with similar "
                  "edge lengths. That is, checks if the lengths of any two "
                  "arbitrary edges (line formed by two vertices) individually "
                  "drawn within the source point cloud and within the target "
                  "point cloud with correspondences are similar. The only "
                  "parameter similarity_threshold is a number between 0 "
                  "(loose) and 1 (strict)");
    py::class_<CorrespondenceCheckerBasedOnDistance,
               PyCorrespondenceChecker<CorrespondenceCheckerBasedOnDistance>,
               CorrespondenceChecker>
            cc_d(m_registration, "CorrespondenceCheckerBasedOnDistance",
                 "Class to check if aligned point clouds are close (less than "
                 "specified threshold).");
    py::class_<CorrespondenceCheckerBasedOnNormal,
               PyCorrespondenceChecker<CorrespondenceCheckerBasedOnNormal>,
               CorrespondenceChecker>
            cc_n(m_registration, "CorrespondenceCheckerBasedOnNormal",
                 "Class to check if two aligned point clouds have similar "
                 "normals. It considers vertex normal affinity of any "
                 "correspondences. It computes dot product of two normal "
                 "vectors. It takes radian value for the threshold.");
    py::class_<FastGlobalRegistrationOption> fgr_option(
            m_registration, "FastGlobalRegistrationOption",
            "Options for FastGlobalRegistration.");
    py::class_<RegistrationResult> registration_result(
            m_registration, "RegistrationResult",
            "Class that contains the registration results.");
    pybind_feature_declarations(m_registration);
    pybind_global_optimization_declarations(m_registration);
    pybind_robust_kernels_declarations(m_registration);
}
void pybind_registration_definitions(py::module &m) {
    auto m_registration = static_cast<py::module>(m.attr("registration"));
    // open3d.registration.ICPConvergenceCriteria
    auto convergence_criteria = static_cast<py::class_<ICPConvergenceCriteria>>(
            m_registration.attr("ICPConvergenceCriteria"));
    py::detail::bind_copy_functions<ICPConvergenceCriteria>(
            convergence_criteria);
    convergence_criteria
            .def(py::init([](double fitness, double rmse, int itr) {
                     return new ICPConvergenceCriteria(fitness, rmse, itr);
                 }),
                 "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
                 "max_iteration"_a = 30)
            .def_readwrite(
                    "relative_fitness",
                    &ICPConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def_readwrite(
                    "relative_rmse", &ICPConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inliner RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def_readwrite("max_iteration",
                           &ICPConvergenceCriteria::max_iteration_,
                           "Maximum iteration before iteration stops.")
            .def("__repr__", [](const ICPConvergenceCriteria &c) {
                return fmt::format(
                        "ICPConvergenceCriteria("
                        "relative_fitness={:e}, "
                        "relative_rmse={:e}, "
                        "max_iteration={:d})",
                        c.relative_fitness_, c.relative_rmse_,
                        c.max_iteration_);
            });

    // open3d.registration.RANSACConvergenceCriteria
    auto ransac_criteria = static_cast<py::class_<RANSACConvergenceCriteria>>(
            m_registration.attr("RANSACConvergenceCriteria"));
    py::detail::bind_copy_functions<RANSACConvergenceCriteria>(ransac_criteria);
    ransac_criteria
            .def(py::init([](int max_iteration, double confidence) {
                     return new RANSACConvergenceCriteria(max_iteration,
                                                          confidence);
                 }),
                 "max_iteration"_a = 100000, "confidence"_a = 0.999)
            .def_readwrite("max_iteration",
                           &RANSACConvergenceCriteria::max_iteration_,
                           "Maximum iteration before iteration stops.")
            .def_readwrite(
                    "confidence", &RANSACConvergenceCriteria::confidence_,
                    "Desired probability of success. Used for estimating early "
                    "termination. Use 1.0 to avoid early termination.")
            .def("__repr__", [](const RANSACConvergenceCriteria &c) {
                return fmt::format(
                        "RANSACConvergenceCriteria("
                        "max_iteration={:d}, "
                        "confidence={:e})",
                        c.max_iteration_, c.confidence_);
            });

    // open3d.registration.TransformationEstimation
    auto te = static_cast<
            py::class_<TransformationEstimation,
                       PyTransformationEstimation<TransformationEstimation>>>(
            m_registration.attr("TransformationEstimation"));
    te.def("compute_rmse", &TransformationEstimation::ComputeRMSE, "source"_a,
           "target"_a, "corres"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &TransformationEstimation::ComputeTransformation, "source"_a,
           "target"_a, "corres"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");
    docstring::ClassMethodDocInject(
            m_registration, "TransformationEstimation", "compute_rmse",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."}});
    docstring::ClassMethodDocInject(
            m_registration, "TransformationEstimation",
            "compute_transformation",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."}});

    // open3d.registration.TransformationEstimationPointToPoint:
    // TransformationEstimation
    auto te_p2p = static_cast<py::class_<
            TransformationEstimationPointToPoint,
            PyTransformationEstimation<TransformationEstimationPointToPoint>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationPointToPoint"));
    py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
            te_p2p);
    te_p2p.def(py::init([](bool with_scaling) {
                   return new TransformationEstimationPointToPoint(
                           with_scaling);
               }),
               "with_scaling"_a = false)
            .def("__repr__",
                 [](const TransformationEstimationPointToPoint &te) {
                     return fmt::format(
                             "TransformationEstimationPointToPoint("
                             "with_scaling={})",
                             te.with_scaling_ ? "True" : "False");
                 })
            .def_readwrite(
                    "with_scaling",
                    &TransformationEstimationPointToPoint::with_scaling_,
                    R"(Set to ``True`` to estimate scaling, ``False`` to force
scaling to be ``1``.

The homogeneous transformation is given by

:math:`T = \begin{bmatrix} c\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}`

Sets :math:`c = 1` if ``with_scaling`` is ``False``.
)");

    // open3d.registration.TransformationEstimationPointToPlane:
    // TransformationEstimation
    auto te_p2l = static_cast<py::class_<
            TransformationEstimationPointToPlane,
            PyTransformationEstimation<TransformationEstimationPointToPlane>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationPointToPlane"));
    py::detail::bind_default_constructor<TransformationEstimationPointToPlane>(
            te_p2l);
    py::detail::bind_copy_functions<TransformationEstimationPointToPlane>(
            te_p2l);
    te_p2l.def(py::init([](std::shared_ptr<RobustKernel> kernel) {
                   return new TransformationEstimationPointToPlane(
                           std::move(kernel));
               }),
               "kernel"_a)
            .def("__repr__",
                 [](const TransformationEstimationPointToPlane &te) {
                     return std::string("TransformationEstimationPointToPlane");
                 })
            .def_readwrite("kernel",
                           &TransformationEstimationPointToPlane::kernel_,
                           "Robust Kernel used in the Optimization");

    // open3d.registration.TransformationEstimationForColoredICP :
    auto te_col = static_cast<py::class_<
            TransformationEstimationForColoredICP,
            PyTransformationEstimation<TransformationEstimationForColoredICP>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationForColoredICP"));
    py::detail::bind_default_constructor<TransformationEstimationForColoredICP>(
            te_col);
    py::detail::bind_copy_functions<TransformationEstimationForColoredICP>(
            te_col);
    te_col.def(py::init([](double lambda_geometric,
                           std::shared_ptr<RobustKernel> kernel) {
                   return new TransformationEstimationForColoredICP(
                           lambda_geometric, std::move(kernel));
               }),
               "lambda_geometric"_a, "kernel"_a)
            .def(py::init([](double lambda_geometric) {
                     return new TransformationEstimationForColoredICP(
                             lambda_geometric);
                 }),
                 "lambda_geometric"_a)
            .def(py::init([](std::shared_ptr<RobustKernel> kernel) {
                     auto te = TransformationEstimationForColoredICP();
                     te.kernel_ = std::move(kernel);
                     return te;
                 }),
                 "kernel"_a)
            .def("__repr__",
                 [](const TransformationEstimationForColoredICP &te) {
                     // This is missing kernel, but getting kernel name on C++
                     // is hard
                     return fmt::format(
                             "TransformationEstimationForColoredICP("
                             "lambda_geometric={})",
                             te.lambda_geometric_);
                 })
            .def_readwrite(
                    "lambda_geometric",
                    &TransformationEstimationForColoredICP::lambda_geometric_,
                    "lambda_geometric")
            .def_readwrite("kernel",
                           &TransformationEstimationForColoredICP::kernel_,
                           "Robust Kernel used in the Optimization");

    // open3d.registration.TransformationEstimationForGeneralizedICP:
    // TransformationEstimation
    auto te_gicp = static_cast<
            py::class_<TransformationEstimationForGeneralizedICP,
                       PyTransformationEstimation<
                               TransformationEstimationForGeneralizedICP>,
                       TransformationEstimation>>(
            m_registration.attr("TransformationEstimationForGeneralizedICP"));
    py::detail::bind_default_constructor<
            TransformationEstimationForGeneralizedICP>(te_gicp);
    py::detail::bind_copy_functions<TransformationEstimationForGeneralizedICP>(
            te_gicp);
    te_gicp.def(py::init([](double epsilon,
                            std::shared_ptr<RobustKernel> kernel) {
                    return new TransformationEstimationForGeneralizedICP(
                            epsilon, std::move(kernel));
                }),
                "epsilon"_a, "kernel"_a)
            .def(py::init([](double epsilon) {
                     return new TransformationEstimationForGeneralizedICP(
                             epsilon);
                 }),
                 "epsilon"_a)
            .def(py::init([](std::shared_ptr<RobustKernel> kernel) {
                     auto te = TransformationEstimationForGeneralizedICP();
                     te.kernel_ = std::move(kernel);
                     return te;
                 }),
                 "kernel"_a)
            .def("__repr__",
                 [](const TransformationEstimationForGeneralizedICP &te) {
                     return fmt::format(
                             "TransformationEstimationForGeneralizedICP("
                             "epsilon={})",
                             te.epsilon_);
                 })
            .def_readwrite("epsilon",
                           &TransformationEstimationForGeneralizedICP::epsilon_,
                           "epsilon")
            .def_readwrite("kernel",
                           &TransformationEstimationForGeneralizedICP::kernel_,
                           "Robust Kernel used in the Optimization");

    // open3d.registration.CorrespondenceChecker
    auto cc = static_cast<
            py::class_<CorrespondenceChecker,
                       PyCorrespondenceChecker<CorrespondenceChecker>>>(
            m_registration.attr("CorrespondenceChecker"));
    cc.def("Check", &CorrespondenceChecker::Check, "source"_a, "target"_a,
           "corres"_a, "transformation"_a,
           "Function to check if two points can be aligned. The two input "
           "point clouds must have exact the same number of points.");
    cc.def_readwrite(
            "require_pointcloud_alignment_",
            &CorrespondenceChecker::require_pointcloud_alignment_,
            "Some checkers do not require point clouds to be aligned, e.g., "
            "the edge length checker. Some checkers do, e.g., the distance "
            "checker.");
    docstring::ClassMethodDocInject(
            m_registration, "CorrespondenceChecker", "Check",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."},
             {"transformation", "The estimated transformation (inplace)."}});

    // open3d.registration.CorrespondenceCheckerBasedOnEdgeLength:
    // CorrespondenceChecker
    auto cc_el = static_cast<py::class_<
            CorrespondenceCheckerBasedOnEdgeLength,
            PyCorrespondenceChecker<CorrespondenceCheckerBasedOnEdgeLength>,
            CorrespondenceChecker>>(
            m_registration.attr("CorrespondenceCheckerBasedOnEdgeLength"));
    py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnEdgeLength>(
            cc_el);
    cc_el.def(py::init([](double similarity_threshold) {
                  return new CorrespondenceCheckerBasedOnEdgeLength(
                          similarity_threshold);
              }),
              "similarity_threshold"_a = 0.9)
            .def("__repr__",
                 [](const CorrespondenceCheckerBasedOnEdgeLength &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnEdgeLength "
                             "with similarity_threshold={:f}",
                             c.similarity_threshold_);
                 })
            .def_readwrite(
                    "similarity_threshold",
                    &CorrespondenceCheckerBasedOnEdgeLength::
                            similarity_threshold_,
                    R"(float value between 0 (loose) and 1 (strict): For the
check to be true,

:math:`||\text{edge}_{\text{source}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{target}}||` and

:math:`||\text{edge}_{\text{target}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{source}}||`

must hold true for all edges.)");

    // open3d.registration.CorrespondenceCheckerBasedOnDistance:
    // CorrespondenceChecker
    auto cc_d = static_cast<py::class_<
            CorrespondenceCheckerBasedOnDistance,
            PyCorrespondenceChecker<CorrespondenceCheckerBasedOnDistance>,
            CorrespondenceChecker>>(
            m_registration.attr("CorrespondenceCheckerBasedOnDistance"));
    py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnDistance>(cc_d);
    cc_d.def(py::init([](double distance_threshold) {
                 return new CorrespondenceCheckerBasedOnDistance(
                         distance_threshold);
             }),
             "distance_threshold"_a)
            .def("__repr__",
                 [](const CorrespondenceCheckerBasedOnDistance &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnDistance with "
                             "distance_threshold={:f}",
                             c.distance_threshold_);
                 })
            .def_readwrite(
                    "distance_threshold",
                    &CorrespondenceCheckerBasedOnDistance::distance_threshold_,
                    "Distance threshold for the check.");

    // open3d.registration.CorrespondenceCheckerBasedOnNormal:
    // CorrespondenceChecker
    auto cc_n = static_cast<py::class_<
            CorrespondenceCheckerBasedOnNormal,
            PyCorrespondenceChecker<CorrespondenceCheckerBasedOnNormal>,
            CorrespondenceChecker>>(
            m_registration.attr("CorrespondenceCheckerBasedOnNormal"));
    py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnNormal>(cc_n);
    cc_n.def(py::init([](double normal_angle_threshold) {
                 return new CorrespondenceCheckerBasedOnNormal(
                         normal_angle_threshold);
             }),
             "normal_angle_threshold"_a)
            .def("__repr__",
                 [](const CorrespondenceCheckerBasedOnNormal &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnNormal with "
                             "normal_threshold={:f}",
                             c.normal_angle_threshold_);
                 })
            .def_readwrite("normal_angle_threshold",
                           &CorrespondenceCheckerBasedOnNormal::
                                   normal_angle_threshold_,
                           "Radian value for angle threshold.");

    // open3d.registration.FastGlobalRegistrationOption:
    auto fgr_option = static_cast<py::class_<FastGlobalRegistrationOption>>(
            m_registration.attr("FastGlobalRegistrationOption"));
    py::detail::bind_copy_functions<FastGlobalRegistrationOption>(fgr_option);
    fgr_option
            .def(py::init([](double division_factor, bool use_absolute_scale,
                             bool decrease_mu,
                             double maximum_correspondence_distance,
                             int iteration_number, double tuple_scale,
                             int maximum_tuple_count, bool tuple_test) {
                     return new FastGlobalRegistrationOption(
                             division_factor, use_absolute_scale, decrease_mu,
                             maximum_correspondence_distance, iteration_number,
                             tuple_scale, maximum_tuple_count, tuple_test);
                 }),
                 "division_factor"_a = 1.4, "use_absolute_scale"_a = false,
                 "decrease_mu"_a = false,
                 "maximum_correspondence_distance"_a = 0.025,
                 "iteration_number"_a = 64, "tuple_scale"_a = 0.95,
                 "maximum_tuple_count"_a = 1000, "tuple_test"_a = true)
            .def_readwrite(
                    "division_factor",
                    &FastGlobalRegistrationOption::division_factor_,
                    "float: Division factor used for graduated non-convexity.")
            .def_readwrite(
                    "use_absolute_scale",
                    &FastGlobalRegistrationOption::use_absolute_scale_,
                    "bool: Measure distance in absolute scale (1) or in scale "
                    "relative to the diameter of the model (0).")
            .def_readwrite("decrease_mu",
                           &FastGlobalRegistrationOption::decrease_mu_,
                           "bool: Set to ``True`` to decrease scale mu by "
                           "``division_factor`` for graduated non-convexity.")
            .def_readwrite("maximum_correspondence_distance",
                           &FastGlobalRegistrationOption::
                                   maximum_correspondence_distance_,
                           "float: Maximum correspondence distance.")
            .def_readwrite("iteration_number",
                           &FastGlobalRegistrationOption::iteration_number_,
                           "int: Maximum number of iterations.")
            .def_readwrite(
                    "tuple_scale", &FastGlobalRegistrationOption::tuple_scale_,
                    "float: Similarity measure used for tuples of feature "
                    "points.")
            .def_readwrite("maximum_tuple_count",
                           &FastGlobalRegistrationOption::maximum_tuple_count_,
                           "float: Maximum tuple numbers.")
            .def_readwrite(
                    "tuple_test", &FastGlobalRegistrationOption::tuple_test_,
                    "bool: Set to `true` to perform geometric compatibility "
                    "tests on initial set of correspondences.")
            .def("__repr__", [](const FastGlobalRegistrationOption &c) {
                return fmt::format(
                        "FastGlobalRegistrationOption("
                        "\ndivision_factor={},"
                        "\nuse_absolute_scale={},"
                        "\ndecrease_mu={},"
                        "\nmaximum_correspondence_distance={},"
                        "\niteration_number={},"
                        "\ntuple_scale={},"
                        "\nmaximum_tuple_count={},"
                        "\ntuple_test={},"
                        "\n)",
                        c.division_factor_, c.use_absolute_scale_,
                        c.decrease_mu_, c.maximum_correspondence_distance_,
                        c.iteration_number_, c.tuple_scale_,
                        c.maximum_tuple_count_, c.tuple_test_);
            });

    // open3d.registration.RegistrationResult
    auto registration_result = static_cast<py::class_<RegistrationResult>>(
            m_registration.attr("RegistrationResult"));
    py::detail::bind_default_constructor<RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<RegistrationResult>(registration_result);
    registration_result
            .def_readwrite("transformation",
                           &RegistrationResult::transformation_,
                           "``4 x 4`` float64 numpy array: The estimated "
                           "transformation matrix.")
            .def_readwrite(
                    "correspondence_set",
                    &RegistrationResult::correspondence_set_,
                    "``n x 2`` int numpy array: Correspondence set between "
                    "source and target point cloud.")
            .def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &RegistrationResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in source). Higher is better.")
            .def("__repr__", [](const RegistrationResult &rr) {
                return fmt::format(
                        "RegistrationResult with "
                        "fitness={:e}"
                        ", inlier_rmse={:e}"
                        ", and correspondence_set size of {:d}"
                        "\nAccess transformation to get result.",
                        rr.fitness_, rr.inlier_rmse_,
                        rr.correspondence_set_.size());
            });
    // Registration functions have similar arguments, sharing arg docstrings
    static const std::unordered_map<std::string, std::string>
            map_shared_argument_docstrings = {
                    {"checkers",
                     "Vector of Checker class to check if two point "
                     "clouds can be aligned. One of "
                     "(``CorrespondenceCheckerBasedOnEdgeLength``, "
                     "``CorrespondenceCheckerBasedOnDistance``, "
                     "``CorrespondenceCheckerBasedOnNormal``)"},
                    {"confidence",
                     "Desired probability of success for RANSAC. Used for "
                     "estimating early termination by k = log(1 - "
                     "confidence)/log(1 - inlier_ratio^{ransac_n}."},
                    {"corres",
                     "o3d.utility.Vector2iVector that stores indices of "
                     "corresponding point or feature arrays."},
                    {"criteria", "Convergence criteria"},
                    {"estimation_method",
                     "Estimation method. One of "
                     "(``TransformationEstimationPointToPoint``, "
                     "``TransformationEstimationPointToPlane``, "
                     "``TransformationEstimationForGeneralizedICP``, "
                     "``TransformationEstimationForColoredICP``)"},
                    {"init", "Initial transformation estimation"},
                    {"lambda_geometric", "lambda_geometric value"},
                    {"epsilon", "epsilon value"},
                    {"kernel", "Robust Kernel used in the Optimization"},
                    {"max_correspondence_distance",
                     "Maximum correspondence points-pair distance."},
                    {"mutual_filter",
                     "Enables mutual filter such that the correspondence of "
                     "the "
                     "source point's correspondence is itself."},
                    {"option", "Registration option"},
                    {"ransac_n",
                     "Fit ransac with ``ransac_n`` correspondences"},
                    {"source_feature", "Source point cloud feature."},
                    {"source", "The source point cloud."},
                    {"target_feature", "Target point cloud feature."},
                    {"target", "The target point cloud."},
                    {"transformation",
                     "The 4x4 transformation matrix to transform ``source`` to "
                     "``target``"}};
    m_registration.def(
            "evaluate_registration", &EvaluateRegistration,
            py::call_guard<py::gil_scoped_release>(),
            "Function for evaluating registration between point clouds",
            "source"_a, "target"_a, "max_correspondence_distance"_a,
            "transformation"_a = Eigen::Matrix4d::Identity());
    docstring::FunctionDocInject(m_registration, "evaluate_registration",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "registration_icp", &RegistrationICP,
            py::call_guard<py::gil_scoped_release>(),
            "Function for ICP registration", "source"_a, "target"_a,
            "max_correspondence_distance"_a,
            "init"_a = Eigen::Matrix4d::Identity(),
            "estimation_method"_a = TransformationEstimationPointToPoint(false),
            "criteria"_a = ICPConvergenceCriteria());
    docstring::FunctionDocInject(m_registration, "registration_icp",
                                 map_shared_argument_docstrings);

    m_registration.def("registration_colored_icp", &RegistrationColoredICP,
                       py::call_guard<py::gil_scoped_release>(),
                       "Function for Colored ICP registration", "source"_a,
                       "target"_a, "max_correspondence_distance"_a,
                       "init"_a = Eigen::Matrix4d::Identity(),
                       "estimation_method"_a =
                               TransformationEstimationForColoredICP(0.968),
                       "criteria"_a = ICPConvergenceCriteria());
    docstring::FunctionDocInject(m_registration, "registration_colored_icp",
                                 map_shared_argument_docstrings);

    m_registration.def("registration_generalized_icp",
                       &RegistrationGeneralizedICP,
                       py::call_guard<py::gil_scoped_release>(),
                       "Function for Generalized ICP registration", "source"_a,
                       "target"_a, "max_correspondence_distance"_a,
                       "init"_a = Eigen::Matrix4d::Identity(),
                       "estimation_method"_a =
                               TransformationEstimationForGeneralizedICP(1e-3),
                       "criteria"_a = ICPConvergenceCriteria());
    docstring::FunctionDocInject(m_registration, "registration_generalized_icp",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "registration_ransac_based_on_correspondence",
            &RegistrationRANSACBasedOnCorrespondence,
            py::call_guard<py::gil_scoped_release>(),
            "Function for global RANSAC registration based on a set of "
            "correspondences",
            "source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
            "estimation_method"_a = TransformationEstimationPointToPoint(false),
            "ransac_n"_a = 3,
            "checkers"_a = std::vector<
                    std::reference_wrapper<const CorrespondenceChecker>>(),
            "criteria"_a = RANSACConvergenceCriteria(100000, 0.999));
    docstring::FunctionDocInject(m_registration,
                                 "registration_ransac_based_on_correspondence",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "registration_ransac_based_on_feature_matching",
            &RegistrationRANSACBasedOnFeatureMatching,
            py::call_guard<py::gil_scoped_release>(),
            "Function for global RANSAC registration based on feature matching",
            "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
            "mutual_filter"_a, "max_correspondence_distance"_a,
            "estimation_method"_a = TransformationEstimationPointToPoint(false),
            "ransac_n"_a = 3,
            "checkers"_a = std::vector<
                    std::reference_wrapper<const CorrespondenceChecker>>(),
            "criteria"_a = RANSACConvergenceCriteria(100000, 0.999));
    docstring::FunctionDocInject(
            m_registration, "registration_ransac_based_on_feature_matching",
            map_shared_argument_docstrings);

    m_registration.def(
            "registration_fgr_based_on_correspondence",
            &FastGlobalRegistrationBasedOnCorrespondence,
            py::call_guard<py::gil_scoped_release>(),
            "Function for fast global registration based on a set of "
            "correspondences",
            "source"_a, "target"_a, "corres"_a,
            "option"_a = FastGlobalRegistrationOption());
    docstring::FunctionDocInject(m_registration,
                                 "registration_fgr_based_on_correspondence",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "registration_fgr_based_on_feature_matching",
            &FastGlobalRegistrationBasedOnFeatureMatching,
            py::call_guard<py::gil_scoped_release>(),
            "Function for fast global registration based on feature matching",
            "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
            "option"_a = FastGlobalRegistrationOption());
    docstring::FunctionDocInject(m_registration,
                                 "registration_fgr_based_on_feature_matching",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "get_information_matrix_from_point_clouds",
            &GetInformationMatrixFromPointClouds,
            py::call_guard<py::gil_scoped_release>(),
            "Function for computing information matrix from transformation "
            "matrix",
            "source"_a, "target"_a, "max_correspondence_distance"_a,
            "transformation"_a);
    docstring::FunctionDocInject(m_registration,
                                 "get_information_matrix_from_point_clouds",
                                 map_shared_argument_docstrings);
    pybind_feature_definitions(m_registration);
    pybind_global_optimization_definitions(m_registration);
    pybind_robust_kernels_definitions(m_registration);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
