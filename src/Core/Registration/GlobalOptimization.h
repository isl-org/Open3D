// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

#pragma once

#include <vector>
#include <memory>

#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>
#include <Core/Utility/Timer.h>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimizationOption.h>
#include <Core/Registration/GlobalOptimizationMethod.h>

namespace three {

namespace {

class OptimizationStatus{
public:
	OptimizationStatus(const PoseGraph& pose_graph,
		const GlobalOptimizationOption option) {
		pose_graph_refined_ = std::make_shared<PoseGraph>();
		*pose_graph_refined_ = pose_graph;
		option_ = option;
		stop_ = false;
	};
	~OptimizationStatus() {};
	void Init();
	void ComputeLinearSystemInClass();
	void SolveLinearSystemInClass();
	void UpdatePoseGraphInClass();
	void UpdateCurrentInClass();
	std::shared_ptr<PoseGraph> UpdatePruneInClass();
	void Checkb() {
		if (b_.maxCoeff() < 1e-6) {
			PrintWarning("[Job finished] b is near zero.\n");
			stop_ = true;
		}
	};
	bool CheckRelative() {
		if (delta_.norm() < 1e-6 * (pose_vector_.norm() + 1e-6)) {
			stop_ = true;
			PrintDebug("[Job finished] delta.norm() < %e * (pose_vector.norm() + %e)\n",
				1e-6, 1e-6);
			return true;
		} else return false;
	}
	void CheckRelativeResidual() {
		if (current_residual_ - new_residual_ < 1e-6 * current_residual_) {
			stop_ = true;
			PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n",
				1e-6);
		}
	}
	void CheckResidual() {
		if (current_residual_ < 1e-6) {
			stop_ = true;
			PrintDebug("[Job finished] current_residual < %e\n",
				1e-6);
		}
	}
	void CheckMaxIteration() {
		if (iter_ == option_.max_iteration_) {
			stop_ = true;
			PrintDebug("[Job finished] reached maximum number of iterations\n",
				1e-6);
		}
	}
	void PrintStatus() {
		PrintDebug("[Iteration %02d] residual : %e, valid edges : %d/%d, time : %.3f sec.\n",
			iter_, current_residual_, valid_edges_num_, n_edges_ - (n_nodes_ - 1),
			timer_iter_.GetDuration() / 1000.0);
	}
	void PrintOverallTime() {
		PrintDebug("[GlobalOptimization] total time : %.3f sec.\n",
			timer_overall_.GetDuration() / 1000.0);
	}
public:
	int iter_;
	Timer timer_overall_;
	Timer timer_iter_;
	Timer timer_lm_;
	std::shared_ptr<PoseGraph> pose_graph_refined_;
	std::shared_ptr<PoseGraph> pose_graph_refined_new_;
	double current_residual_;
	double new_residual_;
	int n_nodes_;
	int n_edges_;
	int valid_edges_num_;
	Eigen::VectorXd pose_vector_;
	Eigen::VectorXd line_process_;
	Eigen::MatrixXd H_;
	Eigen::VectorXd b_;
	Eigen::VectorXd delta_;
	Eigen::VectorXd zeta_;
	Eigen::VectorXd zeta_new_;
	GlobalOptimizationOption option_;
	bool stop_;
};

}

class GraphOptimizationConvergenceCriteria
{
public:
	//GraphOptimizationConvergenceCriteria(double relative_increment = 1e-6,
	//	double relative_residual_increment = 1e-6, int max_iteration = 30) :
	//	relative_increment_(relative_increment), 
	//	relative_residual_increment_(relative_residual_increment),
	//	max_iteration_(max_iteration) {}
	//~GraphOptimizationConvergenceCriteria() {}

public:
	int max_iteration = 100;
	int max_iteration_lm = 20;
	double relative_increment_;
	double relative_residual_increment_;
	double affordable_solver_error_;
	double min_right_term_;
	double min_residual_;
	int max_iteration_;
};

/// Class that defines the convergence criteria of ICP
/// ICP algorithm stops if the relative change of fitness and rmse hit
/// relative_fitness_ and relative_rmse_ individually, or the iteration number
/// exceeds max_iteration_.


/// Function to optimize a PoseGraph 
/// Reference:
/// [Kümmerle et al 2011] 
///    R Kümmerle, G. Grisetti, H. Strasdat, K. Konolige, W. Burgard
///    g2o: A General Framework for Graph Optimization, ICRA 2011
/// [Choi et al 2015]
///    S. Choi, Q.-Y. Zhou, V. Koltun,
///    Robust Reconstruction of Indoor Scenes, CVPR 2015
/// [M. Lourakis 2009] 
///    M. Lourakis,
///    SBA: A Software Package for Generic Sparse Bundle Adjustment, 
///    Transactions on Mathematical Software, 2009
std::shared_ptr<PoseGraph> GlobalOptimization(
		const PoseGraph &pose_graph, 
		const GlobalOptimizationOption &option = GlobalOptimizationOption(),
		const GraphOptimizationMethod &method = 
			GraphOptimizationLevenbergMethodMarquardt());

}	// namespace three
