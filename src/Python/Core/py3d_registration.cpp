// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Registration/TransformationEstimation.h>
#include <Core/Registration/Registration.h>
using namespace three;

template <class TransformationEstimationBase = TransformationEstimation>
class PyTransformationEstimation : public TransformationEstimationBase
{
public:
	using TransformationEstimationBase::TransformationEstimationBase;
	double ComputeRMSE(const PointCloud &source, const PointCloud &target,
			const CorrespondenceSet &corres) const override {
		PYBIND11_OVERLOAD_PURE(double, TransformationEstimationBase,
				source, target, corres);
	}
	Eigen::Matrix4d ComputeTransformation(const PointCloud &source,
			const PointCloud &target,
			const CorrespondenceSet &corres) const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Matrix4d, TransformationEstimationBase,
				source, target, corres);
	}
};

void pybind_registration(py::module &m)
{
	py::class_<ICPConvergenceCriteria> convergence_criteria(m,
			"ICPConvergenceCriteria");
	py::detail::bind_copy_functions<ICPConvergenceCriteria>(
			convergence_criteria);
	convergence_criteria.def("__init__", [](ICPConvergenceCriteria &c,
			double fitness, double rmse, int itr) {
		new (&c)ICPConvergenceCriteria(fitness, rmse, itr);
	}, "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
			"max_iteration"_a = 30);
	convergence_criteria
		.def_readwrite("relative_fitness",
				&ICPConvergenceCriteria::relative_fitness_)
		.def_readwrite("relative_rmse", &ICPConvergenceCriteria::relative_rmse_)
		.def_readwrite("max_iteration", &ICPConvergenceCriteria::max_iteration_)
		.def("__repr__", [](const ICPConvergenceCriteria &c) {
			return std::string("ICPConvergenceCriteria class.\n") +
					std::string("relative_fitness = ") +
					std::to_string(c.relative_fitness_) +
					std::string("relative_rmse = ") +
					std::to_string(c.relative_rmse_) +
					std::string(", max_iteration = " +
					std::to_string(c.max_iteration_));
		});

	py::class_<TransformationEstimation,
			PyTransformationEstimation<TransformationEstimation>>
			te(m, "TransformationEstimation");
	te
		.def("ComputeRMSE", &TransformationEstimation::ComputeRMSE)
		.def("ComputeTransformation",
				&TransformationEstimation::ComputeTransformation);

	py::class_<TransformationEstimationPointToPoint,
			PyTransformationEstimation<TransformationEstimationPointToPoint>,
			TransformationEstimation> te_p2p(m,
			"TransformationEstimationPointToPoint");
	py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
			te_p2p);
	te_p2p.def("__init__", [](TransformationEstimationPointToPoint &c,
			bool with_scaling) {
		new (&c)TransformationEstimationPointToPoint(with_scaling);
	}, "with_scaling"_a = false);
	te_p2p
		.def("__repr__", [](const TransformationEstimationPointToPoint &te) {
			return std::string("TransformationEstimationPointToPoint ") +
					(te.with_scaling_ ? std::string("with scaling.") :
					std::string("without scaling."));
		})
		.def_readwrite("with_scaling",
				&TransformationEstimationPointToPoint::with_scaling_);

	py::class_<TransformationEstimationPointToPlane,
			PyTransformationEstimation<TransformationEstimationPointToPlane>,
			TransformationEstimation> te_p2l(m,
			"TransformationEstimationPointToPlane");
	py::detail::bind_default_constructor<TransformationEstimationPointToPlane>(
			te_p2l);
	py::detail::bind_copy_functions<TransformationEstimationPointToPlane>(
			te_p2l);
	te_p2p
		.def("__repr__", [](const TransformationEstimationPointToPlane &te) {
			return std::string("TransformationEstimationPointToPlane");
		});
	
	py::class_<RegistrationResult> registration_result(m, "RegistrationResult");
	py::detail::bind_default_constructor<RegistrationResult>(
			registration_result);
	py::detail::bind_copy_functions<RegistrationResult>(registration_result);
	registration_result
		.def_readwrite("transformation", &RegistrationResult::transformation_)
		.def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_)
		.def_readwrite("fitness", &RegistrationResult::fitness_)
		.def("__repr__", [](const RegistrationResult &rr) {
			return std::string("RegistrationResult with fitness = ") +
					std::to_string(rr.fitness_) +
					std::string(", and inlier_rmse = ") +
					std::to_string(rr.inlier_rmse_) +
					std::string("\nAccess transformation to get result.");
		});
}

void pybind_registration_methods(py::module &m)
{
	m.def("EvaluateRegistration", &EvaluateRegistration,
			"Function for evaluating registration between point clouds",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"transformation"_a = Eigen::Matrix4d::Identity());
	m.def("RegistrationICP", &RegistrationICP,
			"Function for ICP registration",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"init"_a = Eigen::Matrix4d::Identity(), "estimation"_a =
			TransformationEstimationPointToPoint(false), "criteria"_a =
			ICPConvergenceCriteria());
	m.def("RegistrationRANSACBasedOnCorrespondence",
			&RegistrationRANSACBasedOnCorrespondence,
			"Function for RANSAC registration based on a set of correspondences",
			"source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
			"estimation"_a = TransformationEstimationPointToPoint(false),
			"ransac_n"_a = 6, "max_ransac_iteration"_a = 1000);
}
