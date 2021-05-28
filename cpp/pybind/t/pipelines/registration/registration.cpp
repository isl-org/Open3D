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

#include "open3d/t/pipelines/registration/Registration.h"

#include <memory>
#include <utility>

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Console.h"
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
                       const CorrespondenceSet &correspondences) const {
        PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase, source,
                               target, correspondences);
    }
    core::Tensor ComputeTransformation(
            const t::geometry::PointCloud &source,
            const t::geometry::PointCloud &target,
            const CorrespondenceSet &correspondences) const {
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
                    "If relative change (difference) of inliner RMSE score is "
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
            .def_readwrite(
                    "correspondence_set",
                    &RegistrationResult::correspondence_set_,
                    "Correspondence set between source and target point cloud. "
                    "It is a pair of ``Int64`` ``{C,}`` tensor, where C is "
                    "the number of good correspondences between source and "
                    "target pointcloud. The first tensor is the source "
                    "indices, and the second tensor is corresponding target "
                    "indices. Such that, source[correspondence.first] and "
                    "target[correspondence.second] is a correspondence pair.")
            .def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &RegistrationResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in target). Higher is better.")
            .def("__repr__", [](const RegistrationResult &rr) {
                return fmt::format(
                        "RegistrationResult[fitness_={:e}, "
                        "inlier_rmse={:e}, correspondences={:d}]."
                        "\nAccess transformation to get result.",
                        rr.fitness_, rr.inlier_rmse_,
                        rr.correspondence_set_.second.GetLength());
            });

    // open3d.t.pipelines.registration.TransformationEstimation
    py::class_<TransformationEstimation,
               PyTransformationEstimation<TransformationEstimation>>
            te(m, "TransformationEstimation",
               "Base class that estimates a transformation between two point "
               "clouds. The virtual function ComputeTransformation() must be "
               "implemented in subclasses.");
    te.def("compute_rmse", &TransformationEstimation::ComputeRMSE, "source"_a,
           "target"_a, "correspondences"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &TransformationEstimation::ComputeTransformation, "source"_a,
           "target"_a, "correspondences"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");
    docstring::ClassMethodDocInject(
            m, "TransformationEstimation", "compute_rmse",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"correspondences",
              "Correspondence set between source and target point cloud."}});
    docstring::ClassMethodDocInject(
            m, "TransformationEstimation", "compute_transformation",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"correspondences",
              "Correspondence set between source and target point cloud."}});

    // open3d.t.pipelines.registration.TransformationEstimationPointToPoint
    // TransformationEstimation
    py::class_<TransformationEstimationPointToPoint,
               PyTransformationEstimation<TransformationEstimationPointToPoint>,
               TransformationEstimation>
            te_p2p(m, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to point "
                   "distance.");
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
                   "plane "
                   "distance.");
    py::detail::bind_default_constructor<TransformationEstimationPointToPlane>(
            te_p2l);
    py::detail::bind_copy_functions<TransformationEstimationPointToPlane>(
            te_p2l);
    te_p2l.def(py::init())
            .def("__repr__",
                 [](const TransformationEstimationPointToPlane &te) {
                     return std::string("TransformationEstimationPointToPlane");
                 });
}

// Registration functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"correspondences",
                 "pair of Tensors that stores indices of "
                 "corresponding point or feature arrays."},
                {"criteria", "Convergence criteria"},
                {"criteria_list",
                 "List of Convergence criteria for each scale of multi-scale "
                 "icp."},
                {"estimation_method",
                 "Estimation method. One of "
                 "(``TransformationEstimationPointToPoint``, "
                 "``TransformationEstimationPointToPlane``)"},
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
                {"voxel_sizes",
                 "o3d.utility.DoubleVector of voxel sizes in strictly "
                 "decreasing order, for multi-scale icp."}};

void pybind_registration_methods(py::module &m) {
    m.def("evaluate_registration", &EvaluateRegistration,
          py::call_guard<py::gil_scoped_release>(),
          "Function for evaluating registration between point clouds",
          "source"_a, "target"_a, "max_correspondence_distance"_a,
          "transformation"_a = core::Tensor::Eye(4, core::Dtype::Float64,
                                                 core::Device("CPU:0")));
    docstring::FunctionDocInject(m, "evaluate_registration",
                                 map_shared_argument_docstrings);

    m.def("registration_icp", &RegistrationICP,
          py::call_guard<py::gil_scoped_release>(),
          "Function for ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init_source_to_target"_a = core::Tensor::Eye(4, core::Dtype::Float64,
                                                        core::Device("CPU:0")),
          "estimation_method"_a = TransformationEstimationPointToPoint(),
          "criteria"_a = ICPConvergenceCriteria());
    docstring::FunctionDocInject(m, "registration_icp",
                                 map_shared_argument_docstrings);

    m.def("registration_multi_scale_icp", &RegistrationMultiScaleICP,
          py::call_guard<py::gil_scoped_release>(),
          "Function for Multi-Scale ICP registration", "source"_a, "target"_a,
          "voxel_sizes"_a, "criteria_list"_a, "max_correspondence_distances"_a,
          "init_source_to_target"_a = core::Tensor::Eye(4, core::Dtype::Float64,
                                                        core::Device("CPU:0")),
          "estimation_method"_a = TransformationEstimationPointToPoint());
    docstring::FunctionDocInject(m, "registration_multi_scale_icp",
                                 map_shared_argument_docstrings);
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule(
            "registration", "Tensor-based registration pipeline.");
    pybind_registration_classes(m_submodule);
    pybind_registration_methods(m_submodule);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
