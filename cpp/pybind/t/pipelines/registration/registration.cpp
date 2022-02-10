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
    core::Tensor ComputeTransformation(
            const t::geometry::PointCloud &source,
            const t::geometry::PointCloud &target,
            const core::Tensor &correspondences) const {
        PYBIND11_OVERLOAD_PURE(core::Tensor, TransformationEstimationBase,
                               source, target, correspondences);
    }
};

void pybind_registration_classes(py::module &m) {
    // open3d.t.pipelines.registration.ICPConvergenceCriteria
    py::class_<ICPConvergenceCriteria> convergence_criteria(
            m, "ICPConvergenceCriteria",
            "Convergence criteria of ICP. "
            "ICP algorithm stops if the relative change of fitness and rmse "
            "hit ``relative_fitness`` and ``relative_rmse`` individually, "
            "or the iteration number exceeds ``max_iteration``.");
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
    py::class_<RegistrationResult> registration_result(m, "RegistrationResult",
                                                       "Registration results.");
    py::detail::bind_default_constructor<RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<RegistrationResult>(registration_result);
    registration_result
            .def_readwrite("transformation",
                           &RegistrationResult::transformation_,
                           "``4 x 4`` float64 tensor on CPU: The estimated "
                           "transformation matrix.")
            .def_readwrite("correspondences_",
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
            .def_readwrite("save_loss_log", &RegistrationResult::save_loss_log_,
                           "To store iteration-wise information in "
                           "`loss_log_`, mark this as `True`.")
            .def_readwrite("loss_log", &RegistrationResult::loss_log_,
                           "tensor_map containing iteration-wise information. "
                           "The tensor_map contains `index` (primary-key), "
                           "`scale`, `iteration`, `inlier_rmse`, `fitness`, "
                           "`transformation`, on CPU device.")
            .def("__repr__", [](const RegistrationResult &rr) {
                return fmt::format(
                        "RegistrationResult[fitness_={:e}, "
                        "inlier_rmse={:e}, correspondences={:d}]."
                        "\nAccess transformation to get result.",
                        rr.fitness_, rr.inlier_rmse_,
                        rr.fitness_ * rr.correspondences_.GetLength());
            });

    // open3d.t.pipelines.registration.TransformationEstimation
    py::class_<TransformationEstimation,
               PyTransformationEstimation<TransformationEstimation>>
            te(m, "TransformationEstimation",
               "Base class that estimates a transformation between two "
               "point clouds. The virtual function ComputeTransformation() "
               "must be implemented in subclasses.");
    te.def("compute_rmse", &TransformationEstimation::ComputeRMSE, "source"_a,
           "target"_a, "correspondences"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &TransformationEstimation::ComputeTransformation, "source"_a,
           "target"_a, "correspondences"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");
    docstring::ClassMethodDocInject(m, "TransformationEstimation",
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
            m, "TransformationEstimation", "compute_transformation",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"correspondences",
              "Tensor of type Int64 containing indices of corresponding target "
              "points, where the value is the target index and the index of "
              "the value itself is the source index. It contains -1 as value "
              "at index with no correspondence."}});

    // open3d.t.pipelines.registration.TransformationEstimationPointToPoint
    // TransformationEstimation
    py::class_<TransformationEstimationPointToPoint,
               PyTransformationEstimation<TransformationEstimationPointToPoint>,
               TransformationEstimation>
            te_p2p(m, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to "
                   "point distance.");
    py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
            te_p2p);
    te_p2p.def(py::init())
            .def("__repr__",
                 [](const TransformationEstimationPointToPoint &te) {
                     return std::string("TransformationEstimationPointToPoint");
                 });

    // open3d.t.pipelines.registration.TransformationEstimationPointToPlane
    // TransformationEstimation
    py::class_<TransformationEstimationPointToPlane,
               PyTransformationEstimation<TransformationEstimationPointToPlane>,
               TransformationEstimation>
            te_p2l(m, "TransformationEstimationPointToPlane",
                   "Class to estimate a transformation for point to "
                   "plane distance.");
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
    py::class_<
            TransformationEstimationForColoredICP,
            PyTransformationEstimation<TransformationEstimationForColoredICP>,
            TransformationEstimation>
            te_col(m, "TransformationEstimationForColoredICP",
                   "Class to estimate a transformation between two point "
                   "clouds using color information");
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
}

// Registration functions have similar arguments, sharing arg
// docstrings.
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
                 "point-cloud for performance. By default origianl scale of "
                 "the point-cloud will be used."},
                {"voxel_sizes",
                 "o3d.utility.DoubleVector of voxel sizes in strictly "
                 "decreasing order, for multi-scale icp."},
                {"save_loss_log",
                 "When `True`, it saves the iteration-wise values of "
                 "`fitness`, `inlier_rmse`, `transformaton`, `scale`, "
                 "`iteration` in `loss_log_` in `regsitration_result`. "
                 "Default: False."}};

void pybind_registration_methods(py::module &m) {
    m.def("evaluate_registration", &EvaluateRegistration,
          py::call_guard<py::gil_scoped_release>(),
          "Function for evaluating registration between point clouds",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a =
                  core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")));
    docstring::FunctionDocInject(m, "evaluate_registration",
                                 map_shared_argument_docstrings);

    m.def("icp", &ICP, py::call_guard<py::gil_scoped_release>(),
          "Function for ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init_source_to_target"_a =
                  core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
          "estimation_method"_a = TransformationEstimationPointToPoint(),
          "criteria"_a = ICPConvergenceCriteria(), "voxel_size"_a = -1.0,
          "save_loss_log"_a = false);
    docstring::FunctionDocInject(m, "icp", map_shared_argument_docstrings);

    m.def("multi_scale_icp", &MultiScaleICP,
          py::call_guard<py::gil_scoped_release>(),
          "Function for Multi-Scale ICP registration", "source"_a, "target"_a,
          "voxel_sizes"_a, "criteria_list"_a, "max_correspondence_distances"_a,
          "init_source_to_target"_a =
                  core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
          "estimation_method"_a = TransformationEstimationPointToPoint(),
          "save_loss_log"_a = false);
    docstring::FunctionDocInject(m, "multi_scale_icp",
                                 map_shared_argument_docstrings);

    m.def("get_information_matrix", &GetInformationMatrix,
          py::call_guard<py::gil_scoped_release>(),
          "Function for computing information matrix from transformation "
          "matrix. Information matrix is tensor of shape {6, 6}, dtype Float64 "
          "on CPU device.",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a);
    docstring::FunctionDocInject(m, "get_information_matrix",
                                 map_shared_argument_docstrings);
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule(
            "registration", "Tensor-based registration pipeline.");
    pybind_registration_classes(m_submodule);
    pybind_registration_methods(m_submodule);

    pybind_robust_kernels(m_submodule);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
