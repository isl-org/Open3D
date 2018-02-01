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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Registration/Feature.h>
#include <Core/Registration/CorrespondenceChecker.h>
#include <Core/Registration/TransformationEstimation.h>
#include <Core/Registration/Registration.h>
#include <Core/Registration/ColoredICP.h>

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

template <class CorrespondenceCheckerBase = CorrespondenceChecker>
class PyCorrespondenceChecker : public CorrespondenceCheckerBase
{
public:
	using CorrespondenceCheckerBase::CorrespondenceCheckerBase;
	bool Check(const PointCloud &source, const PointCloud &target,
			const CorrespondenceSet &corres,
			const Eigen::Matrix4d &transformation) const override {
		PYBIND11_OVERLOAD_PURE(bool, CorrespondenceCheckerBase,
				source, target, corres, transformation);
	}
};

void pybind_registration(py::module &m)
{
	py::class_<ICPConvergenceCriteria> convergence_criteria(m,
			"ICPConvergenceCriteria");
	py::detail::bind_copy_functions<ICPConvergenceCriteria>(
			convergence_criteria);
	convergence_criteria
		.def(py::init([](double fitness, double rmse, int itr) {
			return new ICPConvergenceCriteria(fitness, rmse, itr);
		}), "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
				"max_iteration"_a = 30)
		.def_readwrite("relative_fitness",
				&ICPConvergenceCriteria::relative_fitness_)
		.def_readwrite("relative_rmse", &ICPConvergenceCriteria::relative_rmse_)
		.def_readwrite("max_iteration", &ICPConvergenceCriteria::max_iteration_)
		.def("__repr__", [](const ICPConvergenceCriteria &c) {
			return std::string("ICPConvergenceCriteria class with ") +
					std::string("relative_fitness = ") +
					std::to_string(c.relative_fitness_) +
					std::string(", relative_rmse = ") +
					std::to_string(c.relative_rmse_) +
					std::string(", and max_iteration = " +
					std::to_string(c.max_iteration_));
		});

	py::class_<RANSACConvergenceCriteria> ransac_criteria(m,
			"RANSACConvergenceCriteria");
	py::detail::bind_copy_functions<RANSACConvergenceCriteria>(
			ransac_criteria);
	ransac_criteria
		.def(py::init([](int max_iteration, int max_validation) {
			return new RANSACConvergenceCriteria(max_iteration, max_validation);
		}), "max_iteration"_a = 1000, "max_validation"_a = 1000)
		.def_readwrite("max_iteration",
				&RANSACConvergenceCriteria::max_iteration_)
		.def_readwrite("max_validation",
				&RANSACConvergenceCriteria::max_validation_)
		.def("__repr__", [](const RANSACConvergenceCriteria &c) {
			return std::string("RANSACConvergenceCriteria class with ") +
					std::string("max_iteration = ") +
					std::to_string(c.max_iteration_) +
					std::string(", and max_validation = " +
					std::to_string(c.max_validation_));
		});

	py::class_<TransformationEstimation,
			PyTransformationEstimation<TransformationEstimation>>
			te(m, "TransformationEstimation");
	te
		.def("compute_rmse", &TransformationEstimation::ComputeRMSE)
		.def("compute_transformation",
				&TransformationEstimation::ComputeTransformation);

	py::class_<TransformationEstimationPointToPoint,
			PyTransformationEstimation<TransformationEstimationPointToPoint>,
			TransformationEstimation> te_p2p(m,
			"TransformationEstimationPointToPoint");
	py::detail::bind_copy_functions<TransformationEstimationPointToPoint>(
			te_p2p);
	te_p2p
		.def(py::init([](bool with_scaling) {
			return new TransformationEstimationPointToPoint(with_scaling);
		}), "with_scaling"_a = false)
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
	te_p2l
		.def("__repr__", [](const TransformationEstimationPointToPlane &te) {
			return std::string("TransformationEstimationPointToPlane");
		});

	py::class_<CorrespondenceChecker,
			PyCorrespondenceChecker<CorrespondenceChecker>>
			cc(m, "CorrespondenceChecker");
	cc
			.def("Check", &CorrespondenceChecker::Check);

	py::class_<CorrespondenceCheckerBasedOnEdgeLength,
			PyCorrespondenceChecker<CorrespondenceCheckerBasedOnEdgeLength>,
			CorrespondenceChecker> cc_el(m,
			"CorrespondenceCheckerBasedOnEdgeLength");
	py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnEdgeLength>(
			cc_el);
	cc_el
		.def(py::init([](double similarity_threshold) {
			return new CorrespondenceCheckerBasedOnEdgeLength(
				similarity_threshold);
		}), "similarity_threshold"_a = 0.9)
		.def("__repr__", [](const CorrespondenceCheckerBasedOnEdgeLength &c) {
			return std::string("CorrespondenceCheckerBasedOnEdgeLength with similarity threshold ") +
					std::to_string(c.similarity_threshold_);
		})
		.def_readwrite("similarity_threshold",
				&CorrespondenceCheckerBasedOnEdgeLength::similarity_threshold_);

	py::class_<CorrespondenceCheckerBasedOnDistance,
			PyCorrespondenceChecker<CorrespondenceCheckerBasedOnDistance>,
			CorrespondenceChecker> cc_d(m,
			"CorrespondenceCheckerBasedOnDistance");
	py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnDistance>(
			cc_d);
	cc_d
		.def(py::init([](double distance_threshold) {
			return new CorrespondenceCheckerBasedOnDistance(
				distance_threshold);
		}), "distance_threshold"_a)
		.def("__repr__", [](const CorrespondenceCheckerBasedOnDistance &c) {
			return std::string("CorrespondenceCheckerBasedOnDistance with distance threshold ") +
					std::to_string(c.distance_threshold_);
		})
		.def_readwrite("distance_threshold",
				&CorrespondenceCheckerBasedOnDistance::distance_threshold_);

	py::class_<CorrespondenceCheckerBasedOnNormal,
			PyCorrespondenceChecker<CorrespondenceCheckerBasedOnNormal>,
			CorrespondenceChecker> cc_n(m,
			"CorrespondenceCheckerBasedOnNormal");
	py::detail::bind_copy_functions<CorrespondenceCheckerBasedOnNormal>(
			cc_n);
	cc_n
		.def(py::init([](double normal_angle_threshold) {
			return new CorrespondenceCheckerBasedOnNormal(
					normal_angle_threshold);
		}), "normal_angle_threshold"_a)
		.def("__repr__", [](const CorrespondenceCheckerBasedOnNormal &c) {
			return std::string("CorrespondenceCheckerBasedOnNormal with normal threshold ") +
					std::to_string(c.normal_angle_threshold_);
		})
		.def_readwrite("normal_angle_threshold",
				&CorrespondenceCheckerBasedOnNormal::normal_angle_threshold_);

	py::class_<RegistrationResult> registration_result(m, "RegistrationResult");
	py::detail::bind_default_constructor<RegistrationResult>(
			registration_result);
	py::detail::bind_copy_functions<RegistrationResult>(registration_result);
	registration_result
		.def_readwrite("transformation", &RegistrationResult::transformation_)
		.def_readwrite("correspondence_set",
				&RegistrationResult::correspondence_set_)
		.def_readwrite("inlier_rmse", &RegistrationResult::inlier_rmse_)
		.def_readwrite("fitness", &RegistrationResult::fitness_)
		.def("__repr__", [](const RegistrationResult &rr) {
			return std::string("RegistrationResult with fitness = ") +
					std::to_string(rr.fitness_) +
					std::string(", inlier_rmse = ") +
					std::to_string(rr.inlier_rmse_) +
					std::string(", and correspondence_set size of ") +
					std::to_string(rr.correspondence_set_.size()) +
					std::string("\nAccess transformation to get result.");
		});
}

void pybind_registration_methods(py::module &m)
{
	m.def("evaluate_registration", &EvaluateRegistration,
			"Function for evaluating registration between point clouds",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"transformation"_a = Eigen::Matrix4d::Identity());
	m.def("registration_icp", &RegistrationICP,
			"Function for ICP registration",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"init"_a = Eigen::Matrix4d::Identity(), "estimation_method"_a =
			TransformationEstimationPointToPoint(false), "criteria"_a =
			ICPConvergenceCriteria());
	m.def("registration_colored_icp", &RegistrationColoredICP,
			"Function for Colored ICP registration",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"init"_a = Eigen::Matrix4d::Identity(),
			"criteria"_a = ICPConvergenceCriteria());
	m.def("registration_ransac_based_on_correspondence",
			&RegistrationRANSACBasedOnCorrespondence,
			"Function for global RANSAC registration based on a set of correspondences",
			"source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
			"estimation_method"_a = TransformationEstimationPointToPoint(false),
			"ransac_n"_a = 6, "criteria"_a = RANSACConvergenceCriteria());
	m.def("registration_ransac_based_on_feature_matching",
			&RegistrationRANSACBasedOnFeatureMatching,
			"Function for global RANSAC registration based on feature matching",
			"source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
			"max_correspondence_distance"_a, "estimation_method"_a =
			TransformationEstimationPointToPoint(false), "ransac_n"_a = 4,
			"checkers"_a = std::vector<std::reference_wrapper<const
			CorrespondenceChecker>>(), "criteria"_a =
			RANSACConvergenceCriteria(100000, 100));
	m.def("get_information_matrix_from_point_clouds",
			&GetInformationMatrixFromPointClouds,
			"Function for computing information matrix from RegistrationResult",
			"source"_a, "target"_a, "max_correspondence_distance"_a,
			"transformation_result"_a);
}
