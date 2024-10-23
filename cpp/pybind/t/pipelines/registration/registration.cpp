// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Registration.h"

#include <memory>
#include <utility>

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/t/pipelines/registration/registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

template <class TransformationEstimationBase = TransformationEstimation>
class PyTransformationEstimation : public TransformationEstimationBase {
public:
    using TransformationEstimationBase::TransformationEstimationBase;
    TransformationEstimationType GetTransformationEstimationType() const {
        PYBIND11_OVERLOAD_PURE(TransformationEstimationType,
                               TransformationEstimationBase, void);
    }
    double ComputeRMSE(const t::geometry::PointCloud &source,
                       const t::geometry::PointCloud &target,
                       const core::Tensor &correspondences) const {
        PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase, source,
                               target, correspondences);
    }
    core::Tensor ComputeTransformation(const t::geometry::PointCloud &source,
                                       const t::geometry::PointCloud &target,
                                       const core::Tensor &correspondences,
                                       const core::Tensor &current_transform,
                                       const std::size_t iteration) const {
        PYBIND11_OVERLOAD_PURE(core::Tensor, TransformationEstimationBase,
                               source, target, correspondences,
                               current_transform, iteration);
    }
};

// Registration functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"correspondences",
                 "Tensor of type Int64 containing indices of corresponding "
                 "target points, where the value is the target index and the "
                 "index of the value itself is the source index. It contains "
                 "-1 as value at index with no correspondence."},
                {"criteria", "Convergence criteria"},
                {"criteria_list",
                 "List of Convergence criteria for each scale of multi-scale "
                 "icp."},
                {"estimation_method",
                 "Estimation method. One of "
                 "(``TransformationEstimationPointToPoint``, "
                 "``TransformationEstimationPointToPlane``, "
                 "``TransformationEstimationForColoredICP``, "
                 "``TransformationEstimationForGeneralizedICP``)"},
                {"init_source_to_target", "Initial transformation estimation"},
                {"max_correspondence_distance",
                 "Maximum correspondence points-pair distance."},
                {"max_correspondence_distances",
                 "o3d.utility.DoubleVector of maximum correspondence "
                 "points-pair distances for multi-scale icp."},
                {"option", "Registration option"},
                {"source", "The source point cloud."},
                {"target", "The target point cloud."},
                {"transformation",
                 "The 4x4 transformation matrix of type Float64 "
                 "to transform ``source`` to ``target``"},
                {"voxel_size",
                 "The input pointclouds will be down-sampled to this "
                 "`voxel_size` scale. If `voxel_size` < 0, original scale will "
                 "be used. However it is highly recommended to down-sample the "
                 "point-cloud for performance. By default original scale of "
                 "the point-cloud will be used."},
                {"voxel_sizes",
                 "o3d.utility.DoubleVector of voxel sizes in strictly "
                 "decreasing order, for multi-scale icp."},
                {"callback_after_iteration",
                 "Optional lambda function, saves string to tensor map of "
                 "attributes such as iteration_index, scale_index, "
                 "scale_iteration_index, inlier_rmse, fitness, transformation, "
                 "on CPU device, updated after each iteration."}};

void pybind_registration_declarations(py::module &m) {
    py::module m_registration = m.def_submodule(
            "registration", "Tensor-based registration pipeline.");
    py::class_<ICPConvergenceCriteria> convergence_criteria(
            m_registration, "ICPConvergenceCriteria",
            "Convergence criteria of ICP. "
            "ICP algorithm stops if the relative change of fitness and rmse "
            "hit ``relative_fitness`` and ``relative_rmse`` individually, "
            "or the iteration number exceeds ``max_iteration``.");
    py::class_<RegistrationResult> registration_result(
            m_registration, "RegistrationResult", "Registration results.");
    py::class_<TransformationEstimation,
               PyTransformationEstimation<TransformationEstimation>>
            te(m_registration, "TransformationEstimation",
               "Base class that estimates a transformation between two "
               "point clouds. The virtual function ComputeTransformation() "
               "must be implemented in subclasses.");
    py::class_<TransformationEstimationPointToPoint,
               PyTransformationEstimation<TransformationEstimationPointToPoint>,
               TransformationEstimation>
            te_p2p(m_registration, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to "
                   "point distance.");
    py::class_<TransformationEstimationPointToPlane,
               PyTransformationEstimation<TransformationEstimationPointToPlane>,
               TransformationEstimation>
            te_p2l(m_registration, "TransformationEstimationPointToPlane",
                   "Class to estimate a transformation for point to "
                   "plane distance.");
    py::class_<
            TransformationEstimationForColoredICP,
            PyTransformationEstimation<TransformationEstimationForColoredICP>,
            TransformationEstimation>
            te_col(m_registration, "TransformationEstimationForColoredICP",
                   "Class to estimate a transformation between two point "
                   "clouds using color information");
    py::class_<
            TransformationEstimationForDopplerICP,
            PyTransformationEstimation<TransformationEstimationForDopplerICP>,
            TransformationEstimation>
            te_dop(m_registration, "TransformationEstimationForDopplerICP",
                   "Class to estimate a transformation between two point "
                   "clouds using color information");
    pybind_robust_kernel_declarations(m_registration);
}
void pybind_registration_definitions(py::module &m) {
    auto m_registration = static_cast<py::module>(m.attr("registration"));
    // open3d.t.pipelines.registration.ICPConvergenceCriteria
    auto convergence_criteria = static_cast<py::class_<ICPConvergenceCriteria>>(
            m_registration.attr("ICPConvergenceCriteria"));
    py::detail::bind_copy_functions<ICPConvergenceCriteria>(
            convergence_criteria);
    convergence_criteria
            .def(py::init<double, double, int>(), "relative_fitness"_a = 1e-6,
                 "relative_rmse"_a = 1e-6, "max_iteration"_a = 30)
            .def_readwrite(
                    "relative_fitness",
                    &ICPConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def_readwrite(
                    "relative_rmse", &ICPConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inlier RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def_readwrite("max_iteration",
                           &ICPConvergenceCriteria::max_iteration_,
                           "Maximum iteration before iteration stops.")
            .def("__repr__", [](const ICPConvergenceCriteria &c) {
                return fmt::format(
                        "ICPConvergenceCriteria[relative_fitness_={:e}, "
                        "relative_rmse={:e}, max_iteration_={:d}].",
                        c.relative_fitness_, c.relative_rmse_,
                        c.max_iteration_);
            });

    // open3d.t.pipelines.registration.RegistrationResult
    auto registration_result = static_cast<py::class_<RegistrationResult>>(
            m_registration.attr("RegistrationResult"));
    py::detail::bind_default_constructor<RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<RegistrationResult>(registration_result);
    registration_result
            .def_readwrite("transformation",
                           &RegistrationResult::transformation_,
                           "``4 x 4`` float64 tensor on CPU: The estimated "
                           "transformation matrix.")
            .def_readwrite("correspondence_set",
                           &RegistrationResult::correspondences_,
                           "Tensor of type Int64 containing indices of "
                           "corresponding target points, where the value is "
                           "the target index and the index of the value itself "
                           "is the source index. It contains -1 as value at "
                           "index with no correspondence.")
            .def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite("fitness", &RegistrationResult::fitness_,
                           "float: The overlapping area (# of inlier "
                           "correspondences "
                           "/ # of points in source). Higher is better.")
            .def_readwrite(
                    "converged", &RegistrationResult::converged_,
                    "bool: Specifies whether the algorithm converged or not.")
            .def_readwrite(
                    "num_iterations", &RegistrationResult::num_iterations_,
                    "int: Number of iterations the algorithm took to converge.")
            .def("__repr__", [](const RegistrationResult &rr) {
                return fmt::format(
                        "RegistrationResult["
                        "converged={}"
                        ", num_iteration={:d}"
                        ", fitness_={:e}"
                        ", inlier_rmse={:e}"
                        ", correspondences={:d}]."
                        "\nAccess transformation to get result.",
                        rr.converged_, rr.num_iterations_, rr.fitness_,
                        rr.inlier_rmse_, rr.correspondences_.GetLength());
            });

    // open3d.t.pipelines.registration.TransformationEstimation
    auto te = static_cast<
            py::class_<TransformationEstimation,
                       PyTransformationEstimation<TransformationEstimation>>>(
            m_registration.attr("TransformationEstimation"));
    te.def("compute_rmse", &TransformationEstimation::ComputeRMSE, "source"_a,
           "target"_a, "correspondences"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &TransformationEstimation::ComputeTransformation, "source"_a,
           "target"_a, "correspondences"_a,
           "current_transform"_a =
                   core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
           "iteration"_a = 0,
           "Compute transformation from source to target point cloud given "
           "correspondences.");
    docstring::ClassMethodDocInject(m_registration, "TransformationEstimation",
                                    "compute_rmse",
                                    {{"source", "Source point cloud."},
                                     {"target", "Target point cloud."},
                                     {"correspondences",
                                      "Tensor of type Int64 containing "
                                      "indices of corresponding target "
                                      "points, where the value is the "
                                      "target index and the index of "
                                      "the value itself is the source "
                                      "index. It contains -1 as value "
                                      "at index with no correspondence."}});
    docstring::ClassMethodDocInject(
            m_registration, "TransformationEstimation",
            "compute_transformation",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"correspondences",
              "Tensor of type Int64 containing indices of corresponding target "
              "points, where the value is the target index and the index of "
              "the value itself is the source index. It contains -1 as value "
              "at index with no correspondence."},
             {"current_transform", "The current pose estimate of ICP."},
             {"iteration",
              "The current iteration number of the ICP algorithm."}});

    // open3d.t.pipelines.registration.TransformationEstimationPointToPoint
    // TransformationEstimation
    auto te_p2p = static_cast<py::class_<
            TransformationEstimationPointToPoint,
            PyTransformationEstimation<TransformationEstimationPointToPoint>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationPointToPoint"));
    py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
            te_p2p);
    te_p2p.def(py::init())
            .def("__repr__",
                 [](const TransformationEstimationPointToPoint &te) {
                     return std::string("TransformationEstimationPointToPoint");
                 });

    // open3d.t.pipelines.registration.TransformationEstimationPointToPlane
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
    te_p2l.def(py::init([](const RobustKernel &kernel) {
                   return new TransformationEstimationPointToPlane(kernel);
               }),
               "kernel"_a)
            .def("__repr__",
                 [](const TransformationEstimationPointToPlane &te) {
                     return std::string("TransformationEstimationPointToPlane");
                 })
            .def_readwrite("kernel",
                           &TransformationEstimationPointToPlane::kernel_,
                           "Robust Kernel used in the Optimization");

    // open3d.t.pipelines.registration.TransformationEstimationForColoredICP
    // TransformationEstimation
    auto te_col = static_cast<py::class_<
            TransformationEstimationForColoredICP,
            PyTransformationEstimation<TransformationEstimationForColoredICP>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationForColoredICP"));
    py::detail::bind_default_constructor<TransformationEstimationForColoredICP>(
            te_col);
    py::detail::bind_copy_functions<TransformationEstimationForColoredICP>(
            te_col);
    te_col.def(py::init([](double lambda_geometric, RobustKernel &kernel) {
                   return new TransformationEstimationForColoredICP(
                           lambda_geometric, kernel);
               }),
               "lambda_geometric"_a, "kernel"_a)
            .def(py::init([](const double lambda_geometric) {
                     return new TransformationEstimationForColoredICP(
                             lambda_geometric);
                 }),
                 "lambda_geometric"_a)
            .def(py::init([](const RobustKernel kernel) {
                     auto te = TransformationEstimationForColoredICP();
                     te.kernel_ = kernel;
                     return te;
                 }),
                 "kernel"_a)
            .def("__repr__",
                 [](const TransformationEstimationForColoredICP &te) {
                     return std::string(
                                    "TransformationEstimationForColoredICP "
                                    "with lambda_geometric: ") +
                            std::to_string(te.lambda_geometric_);
                 })
            .def_readwrite(
                    "lambda_geometric",
                    &TransformationEstimationForColoredICP::lambda_geometric_,
                    "lambda_geometric")
            .def_readwrite("kernel",
                           &TransformationEstimationForColoredICP::kernel_,
                           "Robust Kernel used in the Optimization");

    // open3d.t.pipelines.registration.TransformationEstimationForDopplerICP
    // TransformationEstimation
    auto te_dop = static_cast<py::class_<
            TransformationEstimationForDopplerICP,
            PyTransformationEstimation<TransformationEstimationForDopplerICP>,
            TransformationEstimation>>(
            m_registration.attr("TransformationEstimationForDopplerICP"));
    py::detail::bind_default_constructor<TransformationEstimationForDopplerICP>(
            te_dop);
    py::detail::bind_copy_functions<TransformationEstimationForDopplerICP>(
            te_dop);
    te_dop.def(py::init([](double period, double lambda_doppler,
                           bool reject_dynamic_outliers,
                           double doppler_outlier_threshold,
                           std::size_t outlier_rejection_min_iteration,
                           std::size_t geometric_robust_loss_min_iteration,
                           std::size_t doppler_robust_loss_min_iteration,
                           RobustKernel &geometric_kernel,
                           RobustKernel &doppler_kernel,
                           core::Tensor &transform_vehicle_to_sensor) {
                   return new TransformationEstimationForDopplerICP(
                           period, lambda_doppler, reject_dynamic_outliers,
                           doppler_outlier_threshold,
                           outlier_rejection_min_iteration,
                           geometric_robust_loss_min_iteration,
                           doppler_robust_loss_min_iteration, geometric_kernel,
                           doppler_kernel, transform_vehicle_to_sensor);
               }),
               "period"_a, "lambda_doppler"_a, "reject_dynamic_outliers"_a,
               "doppler_outlier_threshold"_a,
               "outlier_rejection_min_iteration"_a,
               "geometric_robust_loss_min_iteration"_a,
               "doppler_robust_loss_min_iteration"_a, "goemetric_kernel"_a,
               "doppler_kernel"_a, "transform_vehicle_to_sensor"_a)
            .def(py::init([](const double lambda_doppler) {
                     return new TransformationEstimationForDopplerICP(
                             lambda_doppler);
                 }),
                 "lambda_doppler"_a)
            .def("compute_transformation",
                 py::overload_cast<const t::geometry::PointCloud &,
                                   const t::geometry::PointCloud &,
                                   const core::Tensor &, const core::Tensor &,
                                   const std::size_t>(
                         &TransformationEstimationForDopplerICP::
                                 ComputeTransformation,
                         py::const_),
                 "Compute transformation from source to target point cloud "
                 "given correspondences")
            .def("__repr__",
                 [](const TransformationEstimationForDopplerICP &te) {
                     return std::string(
                                    "TransformationEstimationForDopplerICP "
                                    "with lambda_doppler: ") +
                            std::to_string(te.lambda_doppler_);
                 })
            .def_readwrite("period",
                           &TransformationEstimationForDopplerICP::period_,
                           "Time period (in seconds) between the source and "
                           "the target point clouds.")
            .def_readwrite(
                    "lambda_doppler",
                    &TransformationEstimationForDopplerICP::lambda_doppler_,
                    "`λ ∈ [0, 1]` in the overall energy `(1−λ)EG + λED`. Refer "
                    "the documentation of DopplerICP for more information.")
            .def_readwrite("reject_dynamic_outliers",
                           &TransformationEstimationForDopplerICP::
                                   reject_dynamic_outliers_,
                           "Whether or not to reject dynamic point outlier "
                           "correspondences.")
            .def_readwrite("doppler_outlier_threshold",
                           &TransformationEstimationForDopplerICP::
                                   doppler_outlier_threshold_,
                           "Correspondences with Doppler error greater than "
                           "this threshold are rejected from optimization.")
            .def_readwrite("outlier_rejection_min_iteration",
                           &TransformationEstimationForDopplerICP::
                                   outlier_rejection_min_iteration_,
                           "Number of iterations of ICP after which outlier "
                           "rejection is enabled.")
            .def_readwrite("geometric_robust_loss_min_iteration",
                           &TransformationEstimationForDopplerICP::
                                   geometric_robust_loss_min_iteration_,
                           "Minimum iterations after which Robust Kernel is "
                           "used for the Geometric error")
            .def_readwrite("doppler_robust_loss_min_iteration",
                           &TransformationEstimationForDopplerICP::
                                   doppler_robust_loss_min_iteration_,
                           "Minimum iterations after which Robust Kernel is "
                           "used for the Doppler error")
            .def_readwrite(
                    "geometric_kernel",
                    &TransformationEstimationForDopplerICP::geometric_kernel_,
                    "Robust Kernel used in the Geometric Error Optimization")
            .def_readwrite(
                    "doppler_kernel",
                    &TransformationEstimationForDopplerICP::doppler_kernel_,
                    "Robust Kernel used in the Doppler Error Optimization")
            .def_readwrite("transform_vehicle_to_sensor",
                           &TransformationEstimationForDopplerICP::
                                   transform_vehicle_to_sensor_,
                           "The 4x4 extrinsic transformation matrix between "
                           "the vehicle and the sensor frames.");
    m_registration.def(
            "evaluate_registration", &EvaluateRegistration,
            py::call_guard<py::gil_scoped_release>(),
            "Function for evaluating registration between point clouds",
            "source"_a, "target"_a, "max_correspondence_distance"_a,
            "transformation"_a =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")));
    docstring::FunctionDocInject(m_registration, "evaluate_registration",
                                 map_shared_argument_docstrings);
    m_registration.def(
            "icp", &ICP, py::call_guard<py::gil_scoped_release>(),
            "Function for ICP registration", "source"_a, "target"_a,
            "max_correspondence_distance"_a,
            "init_source_to_target"_a =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            "estimation_method"_a = TransformationEstimationPointToPoint(),
            "criteria"_a = ICPConvergenceCriteria(), "voxel_size"_a = -1.0,
            "callback_after_iteration"_a = py::none());
    docstring::FunctionDocInject(m_registration, "icp",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "multi_scale_icp", &MultiScaleICP,
            py::call_guard<py::gil_scoped_release>(),
            "Function for Multi-Scale ICP registration", "source"_a, "target"_a,
            "voxel_sizes"_a, "criteria_list"_a,
            "max_correspondence_distances"_a,
            "init_source_to_target"_a =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            "estimation_method"_a = TransformationEstimationPointToPoint(),
            "callback_after_iteration"_a = py::none());
    docstring::FunctionDocInject(m_registration, "multi_scale_icp",
                                 map_shared_argument_docstrings);

    m_registration.def(
            "get_information_matrix", &GetInformationMatrix,
            py::call_guard<py::gil_scoped_release>(),
            "Function for computing information matrix from transformation "
            "matrix. Information matrix is tensor of shape {6, 6}, dtype "
            "Float64 "
            "on CPU device.",
            "source"_a, "target"_a, "max_correspondence_distance"_a,
            "transformation"_a);
    docstring::FunctionDocInject(m_registration, "get_information_matrix",
                                 map_shared_argument_docstrings);
    pybind_feature_definitions(m_registration);
    pybind_robust_kernel_definitions(m_registration);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
