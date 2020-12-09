// //
// ----------------------------------------------------------------------------
// // -                        Open3D: www.open3d.org -
// //
// ----------------------------------------------------------------------------
// // The MIT License (MIT)
// //
// // Copyright (c) 2018 www.open3d.org
// //
// // Permission is hereby granted, free of charge, to any person obtaining a
// copy
// // of this software and associated documentation files (the "Software"), to
// deal
// // in the Software without restriction, including without limitation the
// rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in
// // all copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// // FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS
// // IN THE SOFTWARE.
// //
// ----------------------------------------------------------------------------

// #include "open3d/pipelines/registration/Registration.h"

// #include <memory>
// #include <utility>

// #include "open3d/t/geometry/PointCloud.h"
// // #include "open3d/t/pipelines/registration/ColoredICP.h"
// // #include "open3d/t/pipelines/registration/CorrespondenceChecker.h"
// // #include "open3d/t/pipelines/registration/FastGlobalRegistration.h"
// // #include "open3d/t/pipelines/registration/Feature.h"
// #include "open3d/pipelines/registration/RobustKernel.h"
// #include "open3d/t/pipelines/registration/TransformationEstimation.h"
// #include "open3d/utility/Console.h"
// #include "pybind/docstring.h"
// #include "pybind/t/pipelines/registration/registration.h"

// namespace open3d {
// namespace t {
// namespace pipelines {
// namespace registration {

// template <class TransformationEstimationBase = TransformationEstimation>
// class PyTransformationEstimation : public TransformationEstimationBase {
// public:
//     using TransformationEstimationBase::TransformationEstimationBase;
//     TransformationEstimationType GetTransformationEstimationType()
//             const override {
//         PYBIND11_OVERLOAD_PURE(TransformationEstimationType,
//                                TransformationEstimationBase, void);
//     }
//     double ComputeRMSE(const geometry::PointCloud &source,
//                        const geometry::PointCloud &target,
//                        const CorrespondenceSet &corres) const override {
//         PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase, source,
//                                target, corres);
//     }
//     Eigen::Matrix4d ComputeTransformation(
//             const geometry::PointCloud &source,
//             const geometry::PointCloud &target,
//             const CorrespondenceSet &corres) const override {
//         PYBIND11_OVERLOAD_PURE(Eigen::Matrix4d, TransformationEstimationBase,
//                                source, target, corres);
//     }
// };

// template <class CorrespondenceCheckerBase = CorrespondenceChecker>
// class PyCorrespondenceChecker : public CorrespondenceCheckerBase {
// public:
//     using CorrespondenceCheckerBase::CorrespondenceCheckerBase;
//     bool Check(const geometry::PointCloud &source,
//                const geometry::PointCloud &target,
//                const CorrespondenceSet &corres,
//                const Eigen::Matrix4d &transformation) const override {
//         PYBIND11_OVERLOAD_PURE(bool, CorrespondenceCheckerBase, source,
//         target,
//                                corres, transformation);
//     }
// };

// void pybind_registration_classes(py::module &m) {
//     // open3d.registration.ICPConvergenceCriteria
//     py::class_<ICPConvergenceCriteria> convergence_criteria(
//             m, "ICPConvergenceCriteria",
//             "Class that defines the convergence criteria of ICP. ICP "
//             "algorithm "
//             "stops if the relative change of fitness and rmse hit "
//             "``relative_fitness`` and ``relative_rmse`` individually, "
//             "or the "
//             "iteration number exceeds ``max_iteration``.");
//     py::detail::bind_copy_functions<ICPConvergenceCriteria>(
//             convergence_criteria);
//     convergence_criteria
//             .def(py::init([](double fitness, double rmse, int itr) {
//                      return new ICPConvergenceCriteria(fitness, rmse, itr);
//                  }),
//                  "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
//                  "max_iteration"_a = 30)
//             .def_readwrite(
//                     "relative_fitness",
//                     &ICPConvergenceCriteria::relative_fitness_,
//                     "If relative change (difference) of fitness score is
//                     lower " "than ``relative_fitness``, the iteration
//                     stops.")
//             .def_readwrite(
//                     "relative_rmse", &ICPConvergenceCriteria::relative_rmse_,
//                     "If relative change (difference) of inliner RMSE score is
//                     " "lower than ``relative_rmse``, the iteration stops.")
//             .def_readwrite("max_iteration",
//                            &ICPConvergenceCriteria::max_iteration_,
//                            "Maximum iteration before iteration stops.")
//             .def("__repr__", [](const ICPConvergenceCriteria &c) {
//                 return fmt::format(
//                         "ICPConvergenceCriteria class "
//                         "with relative_fitness={:e}, relative_rmse={:e}, "
//                         "and max_iteration={:d}",
//                         c.relative_fitness_, c.relative_rmse_,
//                         c.max_iteration_);
//             });

//     // open3d.registration.TransformationEstimation
//     py::class_<TransformationEstimation,
//                PyTransformationEstimation<TransformationEstimation>>
//             te(m, "TransformationEstimation",
//                "Base class that estimates a transformation between two point
//                " "clouds. The virtual function ComputeTransformation() must
//                be " "implemented in subclasses.");
//     te.def("compute_rmse", &TransformationEstimation::ComputeRMSE,
//     "source"_a,
//            "target"_a, "corres"_a,
//            "Compute RMSE between source and target points cloud given "
//            "correspondences.");
//     te.def("compute_transformation",
//            &TransformationEstimation::ComputeTransformation, "source"_a,
//            "target"_a, "corres"_a,
//            "Compute transformation from source to target point cloud given "
//            "correspondences.");
//     docstring::ClassMethodDocInject(
//             m, "TransformationEstimation", "compute_rmse",
//             {{"source", "Source point cloud."},
//              {"target", "Target point cloud."},
//              {"corres",
//               "Correspondence set between source and target point cloud."}});
//     docstring::ClassMethodDocInject(
//             m, "TransformationEstimation", "compute_transformation",
//             {{"source", "Source point cloud."},
//              {"target", "Target point cloud."},
//              {"corres",
//               "Correspondence set between source and target point cloud."}});

//     // open3d.registration.TransformationEstimationPointToPoint:
//     // TransformationEstimation
//     py::class_<TransformationEstimationPointToPoint,
//                PyTransformationEstimation<TransformationEstimationPointToPoint>,
//                TransformationEstimation>
//             te_p2p(m, "TransformationEstimationPointToPoint",
//                    "Class to estimate a transformation for point to point "
//                    "distance.");
//     py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
//             te_p2p);
//     te_p2p.def(py::init([](bool with_scaling) {
//                    return new TransformationEstimationPointToPoint(
//                            with_scaling);
//                }),
//                "with_scaling"_a = false)
//             .def("__repr__",
//                  [](const TransformationEstimationPointToPoint &te) {
//                      return std::string(
//                                     ""
//                                     "TransformationEstimationPointToPoint ")
//                                     +
//                             (te.with_scaling_
//                                      ? std::string("with scaling.")
//                                      : std::string("without scaling."));
//                  })
//             .def_readwrite(
//                     "with_scaling",
//                     &TransformationEstimationPointToPoint::with_scaling_,
//                     R"(Set to ``True`` to estimate scaling, ``False`` to
//                     force
// scaling to be ``1``.

// The homogeneous transformation is given by

// :math:`T = \begin{bmatrix} c\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1
// \end{bmatrix}`

// Sets :math:`c = 1` if ``with_scaling`` is ``False``.
// )");

//     // open3d.registration.TransformationEstimationPointToPlane:
//     // TransformationEstimation
//     py::class_<TransformationEstimationPointToPlane,
//                PyTransformationEstimation<TransformationEstimationPointToPlane>,
//                TransformationEstimation>
//             te_p2l(m, "TransformationEstimationPointToPlane",
//                    "Class to estimate a transformation for point to plane "
//                    "distance.");
//     py::detail::bind_default_constructor<TransformationEstimationPointToPlane>(
//             te_p2l);
//     py::detail::bind_copy_functions<TransformationEstimationPointToPlane>(
//             te_p2l);
//     te_p2l.def(py::init([](std::shared_ptr<RobustKernel> kernel) {
//                    return new TransformationEstimationPointToPlane(
//                            std::move(kernel));
//                }),
//                "kernel"_a)
//             .def("__repr__",
//                  [](const TransformationEstimationPointToPlane &te) {
//                      return
//                      std::string("TransformationEstimationPointToPlane");
//                  })
//             .def_readwrite("kernel",
//                            &TransformationEstimationPointToPlane::kernel_,
//                            "Robust Kernel used in the Optimization");

//     // open3d.registration.RegistrationResult
//     py::class_<RegistrationResult> registration_result(
//             m, "RegistrationResult",
//             "Class that contains the registration results.");
//     py::detail::bind_default_constructor<RegistrationResult>(
//             registration_result);
//     py::detail::bind_copy_functions<RegistrationResult>(registration_result);
//     registration_result
//             .def_readwrite("transformation",
//                            &RegistrationResult::transformation_,
//                            "``4 x 4`` float64 numpy array: The estimated "
//                            "transformation matrix.")
//             .def_readwrite(
//                     "correspondence_set",
//                     &RegistrationResult::correspondence_set_,
//                     "``n x 2`` int numpy array: Correspondence set between "
//                     "source and target point cloud.")
//             .def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_,
//                            "float: RMSE of all inlier correspondences. Lower
//                            " "is better.")
//             .def_readwrite(
//                     "fitness", &RegistrationResult::fitness_,
//                     "float: The overlapping area (# of inlier correspondences
//                     "
//                     "/ # of points in target). Higher is better.")
//             .def("__repr__", [](const RegistrationResult &rr) {
//                 return fmt::format(
//                         "RegistrationResult with "
//                         "fitness={:e}"
//                         ", inlier_rmse={:e}"
//                         ", and correspondence_set size of {:d}"
//                         "\nAccess transformation to get result.",
//                         rr.fitness_, rr.inlier_rmse_,
//                         rr.correspondence_set_.size());
//             });
// }

// // Registration functions have similar arguments, sharing arg docstrings
// static const std::unordered_map<std::string, std::string>
//         map_shared_argument_docstrings = {
//                 {"checkers",
//                  "Vector of Checker class to check if two point "
//                  "clouds can be aligned. One of "
//                  "(``"
//                  "CorrespondenceCheckerBasedOnEdgeLength``, "
//                  "``"
//                  "CorrespondenceCheckerBasedOnDistance``, "
//                  "``"
//                  "CorrespondenceCheckerBasedOnNormal``)"},
//                 {"confidence",
//                  "Desired probability of success for RANSAC. Used for "
//                  "estimating early termination by k = log(1 - "
//                  "confidence)/log(1 - inlier_ratio^{ransac_n}."},
//                 {"corres",
//                  "o3d.utility.Vector2iVector that stores indices of "
//                  "corresponding point or feature arrays."},
//                 {"criteria", "Convergence criteria"},
//                 {"estimation_method",
//                  "Estimation method. One of "
//                  "(``"
//                  "TransformationEstimationPointToPoint``, "
//                  "``"
//                  "TransformationEstimationPointToPlane``, "
//                  "``"
//                  "TransformationEstimationForColoredICP``)"},
//                 {"init", "Initial transformation estimation"},
//                 {"lambda_geometric", "lambda_geometric value"},
//                 {"kernel", "Robust Kernel used in the Optimization"},
//                 {"max_correspondence_distance",
//                  "Maximum correspondence points-pair distance."},
//                 {"mutual_filter",
//                  "Enables mutual filter such that the correspondence of the "
//                  "source point's correspondence is itself."},
//                 {"option", "Registration option"},
//                 {"ransac_n", "Fit ransac with ``ransac_n`` correspondences"},
//                 {"source_feature", "Source point cloud feature."},
//                 {"source", "The source point cloud."},
//                 {"target_feature", "Target point cloud feature."},
//                 {"target", "The target point cloud."},
//                 {"transformation",
//                  "The 4x4 transformation matrix to transform ``source`` to "
//                  "``target``"}};

// void pybind_registration_methods(py::module &m) {
//     m.def("evaluate_registration", &EvaluateRegistration,
//           "Function for evaluating registration between point clouds",
//           "source"_a, "target"_a, "max_correspondence_distance"_a,
//           "transformation"_a = Eigen::Matrix4d::Identity());
//     docstring::FunctionDocInject(m, "evaluate_registration",
//                                  map_shared_argument_docstrings);

//     m.def("registration_icp", &RegistrationICP, "Function for ICP
//     registration",
//           "source"_a, "target"_a, "max_correspondence_distance"_a,
//           "init"_a = Eigen::Matrix4d::Identity(),
//           "estimation_method"_a =
//           TransformationEstimationPointToPoint(false), "criteria"_a =
//           ICPConvergenceCriteria());
//     docstring::FunctionDocInject(m, "registration_icp",
//                                  map_shared_argument_docstrings);

// void pybind_registration(py::module &m) {
//     py::module m_submodule =
//             m.def_submodule("registration", "Registration pipeline.");
//     pybind_registration_classes(m_submodule);
//     pybind_registration_methods(m_submodule);

//     // pybind_feature(m_submodule);
//     // pybind_feature_methods(m_submodule);
//     // pybind_global_optimization(m_submodule);
//     // pybind_global_optimization_methods(m_submodule);
//     // pybind_robust_kernels(m_submodule);
// }

// }  // namespace registration
// }  // namespace pipelines
// }  // namespace t
// }  // namespace open3d
