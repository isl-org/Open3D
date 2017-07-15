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

#include "GlobalOptimization.h"

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <json/json.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>
#include <Core/Utility/Timer.h>

namespace three{

namespace {

/// Definition of linear operators used for computing Jacobian matrix.
/// If the relative transform of the two geometry is reasonably small, 
/// they can be approximated as below linearized form
/// SE(3) \approx = |     1 -gamma   beta     a |
///                 | gamma      1 -alpha     b |
///                 | -beta  alpha      1     c |
///                 |     0      0      0     1 |
/// It is from sin(x) \approx x and cos(x) \approx 1 when x is almost zero.
/// See [Choi et al 2015] for more detail. Reference list in GlobalOptimization.h
const std::vector<Eigen::Matrix4d> jacobian_operator = {
	(Eigen::Matrix4d() << /* for alpha */
	0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for beta */
	0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for gamma */
	0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for a */
	0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for b */
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0).finished(),
	(Eigen::Matrix4d() << /* for c */
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0).finished() };

inline Eigen::Vector6d GetApproximate6DVector(Eigen::Matrix4d input)
{
	Eigen::Vector6d output;
	output(0) = (-input(1, 2) + input(2, 1)) / 2.0;
	output(1) = (-input(2, 0) + input(0, 2)) / 2.0;
	output(2) = (-input(0, 1) + input(1, 0)) / 2.0;
	output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
	return std::move(output);
}

inline Eigen::Vector6d GetMisalignmentVector(const Eigen::Matrix4d &X_inv, 
		const Eigen::Matrix4d &Ts, const Eigen::Matrix4d &Tt_inv)
{
	Eigen::Matrix4d temp;
	temp.noalias() = X_inv * Tt_inv * Ts;
	return GetApproximate6DVector(temp);
}

inline std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, Eigen::Matrix4d>
		GetRelativePoses(const PoseGraph &pose_graph, int edge_id) 
{
	const PoseGraphEdge &te = pose_graph.edges_[edge_id];
	const PoseGraphNode &ts = pose_graph.nodes_[te.source_node_id_];
	const PoseGraphNode &tt = pose_graph.nodes_[te.target_node_id_];
	Eigen::Matrix4d X_inv = te.transformation_.inverse();
	Eigen::Matrix4d Ts = ts.pose_;
	Eigen::Matrix4d Tt_inv = tt.pose_.inverse();
	return std::make_tuple(std::move(X_inv), std::move(Ts), std::move(Tt_inv));
}

std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> GetJacobian(
		const Eigen::Matrix4d &X_inv, const Eigen::Matrix4d &Ts,
		const Eigen::Matrix4d &Tt_inv)
{
	Eigen::Matrix6d Js = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * Tt_inv * 
				jacobian_operator[i] * Ts;
		Js.block<6, 1>(0, i) = GetApproximate6DVector(temp);
	}
	Eigen::Matrix6d Jt = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * Tt_inv * 
				-jacobian_operator[i] * Ts;
		Jt.block<6, 1>(0, i) = GetApproximate6DVector(temp);
	}
	return std::make_tuple(std::move(Js), std::move(Jt));
}

/// Function to update line_process value defined in [Choi et al 2015]
/// See Eq (2). temp2 value in this function is derived from dE/dl = 0 
std::tuple<Eigen::VectorXd, int> ComputeLineprocess(
		const PoseGraph &pose_graph, const Eigen::VectorXd &zeta,
		const GlobalOptimizationOption &option)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();
	int line_process_cnt = 0;
	int valid_edges_num = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		if (t.uncertain_) {
			Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
			double residual_square = e.transpose() * t.information_ * e;
			double temp = option.line_process_weight_ / 
					(option.line_process_weight_ + residual_square);
			double temp2 = temp * temp;
			if (temp2 < option.edge_prune_threshold_) {
				line_process(line_process_cnt++) = 0; 
			} else {
				line_process(line_process_cnt++) = temp2;				
				valid_edges_num++;
			}				
		}
	}
	return std::make_tuple(std::move(line_process), valid_edges_num);
}

double ComputeResidual(const PoseGraph &pose_graph, Eigen::VectorXd zeta,
		Eigen::VectorXd line_process, const GlobalOptimizationOption &option)
{
	int line_process_cnt = 0;
	double residual = 0.0;
	int n_edges = (int)pose_graph.edges_.size();
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		double line_process_iter = 1.0;
		if (te.uncertain_)
			line_process_iter = line_process(line_process_cnt++);
		Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
		residual += line_process_iter * e.transpose() * te.information_ * e +
				option.line_process_weight_ * 
				pow(sqrt(line_process_iter) - 1, 2.0);
	}
	return residual;
}

Eigen::VectorXd ComputeZeta(const PoseGraph &pose_graph) 
{
	int n_edges = (int)pose_graph.edges_.size();	
	Eigen::VectorXd output(n_edges * 6);
	int line_process_cnt = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
		Eigen::Vector6d e = GetMisalignmentVector(X_inv, Ts, Tt_inv);
		output.block<6, 1>(iter_edge * 6, 0) = e;		
	}
	return std::move(output);
}

/// The information matrix used here is consistent with [Choi et al 2015]. 
/// It is [p_x | I]^T[p_x | I]. \zeta is [\alpha \beta \gamma a b c]
/// Another definition of information matrix used for [Kümmerle et al 2011] is 
/// [I | p_x] ^ T[I | p_x]  so \zeta is [a b c \alpha \beta \gamma].
///
/// To see how H can be derived see [Kümmerle et al 2011].
/// Eq (9) for definition of H and b for k-th constraint.
/// To see how the covariance matrix forms H, check g2o technical note:
/// https ://github.com/RainerKuemmerle/g2o/blob/master/doc/g2o.pdf 
/// Eq (20) and Eq (21). (There is a typo in the equation though. B should be J) 
///
/// This function focus the case that every edge has two nodes (not hyper graph) 
/// so we have two Jacobian matrices from one constraint.
std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeLinearSystem(
		const PoseGraph &pose_graph, const Eigen::VectorXd &zeta,
		const Eigen::VectorXd &line_process)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	int line_process_cnt = 0;
	Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b(n_nodes * 6);
	H.setZero();
	b.setZero();

	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
		
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);

		Eigen::Matrix6d Js, Jt;
		std::tie(Js, Jt) = GetJacobian(X_inv, Ts, Tt_inv);	
		Eigen::Matrix6d JsT_Info =
				Js.transpose() * t.information_;
		Eigen::Matrix6d JtT_Info =
				Jt.transpose() * t.information_;
		Eigen::Vector6d eT_Info = e.transpose() * t.information_;

		double line_process_iter = 1.0;
		if (t.uncertain_) {
			line_process_iter = line_process(line_process_cnt++);
		}
		int id_i = t.source_node_id_ * 6;
		int id_j = t.target_node_id_ * 6;
		H.block<6, 6>(id_i, id_i).noalias() +=
				line_process_iter * JsT_Info * Js;
		H.block<6, 6>(id_i, id_j).noalias() +=
				line_process_iter * JsT_Info * Jt;
		H.block<6, 6>(id_j, id_i).noalias() +=
				line_process_iter * JtT_Info * Js;
		H.block<6, 6>(id_j, id_j).noalias() +=
				line_process_iter * JtT_Info * Jt;
		b.block<6, 1>(id_i, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * Js;
		b.block<6, 1>(id_j, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * Jt;
	}
	return std::make_tuple(std::move(H), std::move(b));
}

Eigen::VectorXd UpdatePoseVector(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	Eigen::VectorXd output(n_nodes * 6);	
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d output_iter = TransformMatrix4dToVector6d(
				pose_graph.nodes_[iter_node].pose_);
		output.block<6, 1>(iter_node * 6, 0) = output_iter;
	}
	return std::move(output);
}

std::shared_ptr<PoseGraph> UpdatePoseGraph(const PoseGraph &pose_graph,
		const Eigen::VectorXd delta) 
{
	std::shared_ptr<PoseGraph> pose_graph_updated =
		std::make_shared<PoseGraph>();
	*pose_graph_updated = pose_graph;
	int n_nodes = (int)pose_graph.nodes_.size();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
		pose_graph_updated->nodes_[iter_node].pose_ =
				TransformVector6dToMatrix4d(delta_iter) *
				pose_graph_updated->nodes_[iter_node].pose_;
	}
	return pose_graph_updated;
}

std::shared_ptr<PoseGraph> PruneInvalidEdges(const PoseGraph &pose_graph, 
		Eigen::VectorXd line_process, const GlobalOptimizationOption &option)
{
	std::shared_ptr<PoseGraph> pose_graph_pruned = 
			std::make_shared<PoseGraph>();

	int n_nodes = (int)pose_graph.nodes_.size();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		const PoseGraphNode &t = pose_graph.nodes_[iter_node];
		pose_graph_pruned->nodes_.push_back(t);
	}
	int n_edges = (int)pose_graph.edges_.size();
	int line_process_cnt = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];		
		if (t.uncertain_) {
			if (line_process(line_process_cnt++) > 
					option.edge_prune_threshold_) {
				pose_graph_pruned->edges_.push_back(t);
			}
		} else {
			pose_graph_pruned->edges_.push_back(t);
		}
	}
	return pose_graph_pruned;
}

std::shared_ptr<PoseGraph> GlobalOptimizationLM(const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("[GlobalOptimizationLM] Optimizing PoseGraph having %d nodes and %d edges\n",
			n_nodes, n_edges);

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::VectorXd zeta = ComputeZeta(*pose_graph_refined);
	double new_residual = ComputeResidual(
			*pose_graph_refined, zeta, line_process, option);
	double current_residual = new_residual;

	int valid_edges_num;
	std::tie(line_process, valid_edges_num) = ComputeLineprocess(
			*pose_graph_refined, zeta, option);

	Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	Eigen::VectorXd pose_vector = UpdatePoseVector(*pose_graph_refined);

	std::tie(H, b) = ComputeLinearSystem(
			*pose_graph_refined, zeta, line_process);

	Eigen::VectorXd H_diag = H.diagonal();
	double tau = 1e-5;
	double current_lambda = tau * H_diag.maxCoeff();
	double ni = 2.0;
	double rho = 0.0;

	PrintDebug("[Initial     ] residual : %e, lambda : %e\n",
			current_residual, current_lambda);

	bool stop = false;
	if (b.maxCoeff() < 1e-6) {
		PrintWarning("[Job finished] b is near zero.\n");
		stop = true;
	}

	Timer timer_overall;
	timer_overall.Start();
	int iter;
	for (iter = 0; !stop; iter++) {
		Timer timer_iter;
		timer_iter.Start();
		int lm_count = 0;
		do {
			Eigen::MatrixXd H_LM = H + current_lambda * H_I;
			Eigen::VectorXd delta = H_LM.ldlt().solve(b);
			if (delta.norm() < 1e-6 * (pose_vector.norm() + 1e-6)) {
				stop = true;
				PrintDebug("[Job finished] delta.norm() < %e * (pose_vector.norm() + %e)\n", 
						1e-6, 1e-6);
			} else {
				std::shared_ptr<PoseGraph> pose_graph_refined_new =
						UpdatePoseGraph(*pose_graph_refined, delta);

				Eigen::VectorXd zeta_new;
				zeta_new = ComputeZeta(*pose_graph_refined_new);
				new_residual = ComputeResidual(
						*pose_graph_refined, zeta_new, line_process, option);
				rho = (current_residual - new_residual) / 
						(delta.dot(current_lambda * delta + b) + 1e-3);
				if (rho > 0) {
					if (current_residual - new_residual < 
							1e-6 * current_residual) {
						stop = true;
						PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n", 1e-6);
					}
					double alpha = 1. - pow((2 * rho - 1), 3);
					alpha = (std::min)(alpha, option.upper_scale_factor_);
					double scale_factor = (std::max)
							(option.lower_scale_factor_, alpha);
					current_lambda *= scale_factor;
					ni = 2;
					current_residual = new_residual;

					zeta = zeta_new;
					*pose_graph_refined = *pose_graph_refined_new;
					pose_vector = UpdatePoseVector(*pose_graph_refined);
					std::tie(line_process, valid_edges_num) = ComputeLineprocess(
							*pose_graph_refined, zeta, option);
					std::tie(H, b) = ComputeLinearSystem(
							*pose_graph_refined, zeta, line_process);
					
					if (b.maxCoeff() < 1e-6) {
						stop = true;
						PrintDebug("[Job finished] b.maxCoeff() < %e\n", 1e-6);
					}
				} else {
					current_lambda *= ni;
					ni *= 2;
				}
			}
			lm_count++;
			if (lm_count > option.max_iteration_lm_) {
				stop = true;
				PrintDebug("[Job finished] lm_count > %d\n", 
						option.max_iteration_lm_);
			}
		} while (!((rho > 0) || stop));
		if (!stop) {
			timer_iter.Stop();
			PrintDebug("[Iteration %02d] residual : %e, lambda : %e, valid edges : %d/%d, time : %.3f sec.\n",
					iter, current_residual, current_lambda, 
					valid_edges_num, n_edges - (n_nodes - 1), 
					timer_iter.GetDuration() / 1000.0);
		}
		if (current_residual < 1e-6) {
			stop = true;
			PrintDebug("[Job finished] current_residual < %e\n", 1e-6);
		}			
		if (iter == option.max_iteration_) {
			stop = true;
			PrintDebug("[Job finished] reached maximum number of iterations\n", 1e-6);
		}
	}	// end for
	timer_overall.Stop();
	PrintDebug("[GlobalOptimizationLM] total time : %.3f sec.\n", 
			timer_overall.GetDuration() / 1000.0);

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned =
			PruneInvalidEdges(*pose_graph_refined, line_process, option);
	return pose_graph_refined_pruned;
}

std::shared_ptr<PoseGraph> GlobalOptimizationGN(const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("[GlobalOptimization] Optimizing PoseGraph having %d nodes and %d edges\n",
			n_nodes, n_edges);

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::VectorXd zeta = ComputeZeta(*pose_graph_refined);
	double new_residual = ComputeResidual(
			*pose_graph_refined, zeta, line_process, option);
	double current_residual = new_residual;

	int valid_edges_num;
	std::tie(line_process, valid_edges_num) = ComputeLineprocess(
			*pose_graph_refined, zeta, option);

	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	Eigen::VectorXd pose_vector = UpdatePoseVector(*pose_graph_refined);

	std::tie(H, b) = ComputeLinearSystem(
			*pose_graph_refined, zeta, line_process);

	PrintDebug("[Initial     ] residual : %e\n", current_residual);

	bool stop = false;
	if (b.maxCoeff() < 1e-6) {
		PrintWarning("[Job finished] b is near zero.\n");
		stop = true;
	}

	Timer timer_overall;
	timer_overall.Start();
	int iter;
	for (iter = 0; !stop; iter++) {
		Timer timer_iter;
		timer_iter.Start();
		
		Eigen::VectorXd delta = H.ldlt().solve(b);
		if (delta.norm() < 1e-6 * (pose_vector.norm() + 1e-6)) {
			stop = true;
			PrintDebug("[Job finished] delta.norm() < %e * (pose_vector.norm() + %e)\n",
					1e-6, 1e-6);
		} else {
			std::shared_ptr<PoseGraph> pose_graph_refined_new =
				UpdatePoseGraph(*pose_graph_refined, delta);

			Eigen::VectorXd zeta_new;
			zeta_new = ComputeZeta(*pose_graph_refined_new);
			new_residual = ComputeResidual(
					*pose_graph_refined, zeta_new, line_process, option);
			if (current_residual - new_residual < 1e-6 * current_residual) {
				stop = true;
				PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n", 
						1e-6);
			}
			current_residual = new_residual;

			zeta = zeta_new;
			*pose_graph_refined = *pose_graph_refined_new;
			pose_vector = UpdatePoseVector(*pose_graph_refined);
			std::tie(line_process, valid_edges_num) = ComputeLineprocess(
					*pose_graph_refined, zeta, option);
			std::tie(H, b) = ComputeLinearSystem(
					*pose_graph_refined, zeta, line_process);

			if (b.maxCoeff() < 1e-6) {
				stop = true;
				PrintDebug("[Job finished] b.maxCoeff() < %e\n", 1e-6);
			}
		}
		if (!stop) {
			timer_iter.Stop();
			PrintDebug("[Iteration %02d] residual : %e, valid edges : %d/%d, time : %.3f sec.\n",
				iter, current_residual, valid_edges_num, n_edges - (n_nodes - 1),
				timer_iter.GetDuration() / 1000.0);
		}
		if (current_residual < 1e-6) {
			stop = true;
			PrintDebug("[Job finished] current_residual < %e\n", 
					1e-6);
		}
		if (iter == option.max_iteration_) {
			stop = true;
			PrintDebug("[Job finished] reached maximum number of iterations\n", 
					1e-6);
		}
	}	// end for
	timer_overall.Stop();
	PrintDebug("[GlobalOptimization] total time : %.3f sec.\n",
			timer_overall.GetDuration() / 1000.0);

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned =
			PruneInvalidEdges(*pose_graph_refined, line_process, option);
	return pose_graph_refined_pruned;
}

void OptimizationStatus::Init() {

	n_nodes_ = (int)pose_graph_refined_->nodes_.size();
	n_edges_ = (int)pose_graph_refined_->edges_.size();
	PrintDebug("[GlobalOptimization] Optimizing PoseGraph having %d nodes and %d edges\n",
		n_nodes_, n_edges_);

	line_process_.resize(n_edges_ - (n_nodes_ - 1));
	line_process_.setOnes();

	zeta_ = ComputeZeta(*pose_graph_refined_);
	new_residual_ = ComputeResidual(
			*pose_graph_refined_, zeta_, line_process_, option_);
	current_residual_ = new_residual_;

	std::tie(line_process_, valid_edges_num_) = ComputeLineprocess(
		*pose_graph_refined_, zeta_, option_);

	pose_vector_ = UpdatePoseVector(*pose_graph_refined_);

	ComputeLinearSystemInClass();

	PrintDebug("[Initial     ] residual : %e\n", current_residual_);
}

void OptimizationStatus::ComputeLinearSystemInClass() {
	std::tie(H_, b_) = ComputeLinearSystem(
		*pose_graph_refined_, zeta_, line_process_);
}

void OptimizationStatus::SolveLinearSystemInClass() {
	delta_ = H_.ldlt().solve(b_);
}

void OptimizationStatus::UpdatePoseGraphInClass() {
	pose_graph_refined_new_ =
		UpdatePoseGraph(*pose_graph_refined_, delta_);
	zeta_new_ = ComputeZeta(*pose_graph_refined_new_);
	new_residual_ = ComputeResidual(
		*pose_graph_refined_, zeta_new_, line_process_, option_);
}

void OptimizationStatus::UpdateCurrentInClass() {
	current_residual_ = new_residual_;
	zeta_ = zeta_new_;
	*pose_graph_refined_ = *pose_graph_refined_new_;
	pose_vector_ = UpdatePoseVector(*pose_graph_refined_);
	std::tie(line_process_, valid_edges_num_) = ComputeLineprocess(
		*pose_graph_refined_, zeta_, option_);
}

std::shared_ptr<PoseGraph> OptimizationStatus::UpdatePruneInClass() {
	return PruneInvalidEdges(*pose_graph_refined_, line_process_, option_);
}

std::shared_ptr<PoseGraph> GlobalOptimizationGNNewType(const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option)
{
	OptimizationStatus status(pose_graph, option);

	status.Init();
	status.Checkb();

	status.timer_overall_.Start();
	for (status.iter_ = 0; !status.stop_; status.iter_++) {
		status.timer_iter_.Start();
		
		status.SolveLinearSystemInClass();

		if (!status.CheckRelative()) {
			status.UpdatePoseGraphInClass();			
			status.CheckRelativeResidual();
			status.UpdateCurrentInClass(); 
			status.ComputeLinearSystemInClass();
			status.Checkb();
		} 
		if (!status.stop_) {
			status.timer_iter_.Stop();
			status.PrintStatus();
		}
		status.CheckResidual();
		status.CheckMaxIteration();
	}	// end for
	status.timer_overall_.Stop();
	status.PrintOverallTime();

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned = 
			status.UpdatePruneInClass();
	return pose_graph_refined_pruned;
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(
		const PoseGraph &pose_graph, 
		const GlobalOptimizationOption &option,
		const GraphOptimizationMethod &method)
{
	return GlobalOptimizationGNNewType(pose_graph, option);
}

}	// namespace three
