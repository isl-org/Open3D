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
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimizationOption.h>
#include <Core/Registration/GlobalOptimizationMethod.h>

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

}	// unnamed namespace

void OptimizationStatusGaussNewton::Init(
	const PoseGraph& pose_graph,
	const GlobalOptimizationOption option) {

	pose_graph_current_ = std::make_shared<PoseGraph>();
	*pose_graph_current_ = pose_graph;
	option_ = option;
	stop_ = false;

	n_nodes_ = (int)pose_graph_current_->nodes_.size();
	n_edges_ = (int)pose_graph_current_->edges_.size();
	line_process_.resize(n_edges_ - (n_nodes_ - 1));
	line_process_.setOnes();

	zeta_ = ComputeZeta(*pose_graph_current_);
	new_residual_ = ComputeResidual(
			*pose_graph_current_, zeta_, line_process_, option_);
	current_residual_ = new_residual_;

	std::tie(line_process_, valid_edges_num_) = ComputeLineprocess(
			*pose_graph_current_, zeta_, option_);

	pose_vector_ = UpdatePoseVector(*pose_graph_current_);

	ComputeLinearSystemInClass();
}

void OptimizationStatusGaussNewton::ComputeLinearSystemInClass() {
	std::tie(H_, b_) = ComputeLinearSystem(
			*pose_graph_current_, zeta_, line_process_);
}

void OptimizationStatusGaussNewton::SolveLinearSystemInClass() {
	delta_ = H_.ldlt().solve(b_);
}

void OptimizationStatusGaussNewton::UpdatePoseGraphInClass() {
	pose_graph_new_ = UpdatePoseGraph(*pose_graph_current_, delta_);
	zeta_new_ = ComputeZeta(*pose_graph_new_);
	new_residual_ = ComputeResidual(
			*pose_graph_current_, zeta_new_, line_process_, option_);
}

void OptimizationStatusGaussNewton::UpdateCurrentInClass() {
	current_residual_ = new_residual_;
	zeta_ = zeta_new_;
	*pose_graph_current_ = *pose_graph_new_;
	pose_vector_ = UpdatePoseVector(*pose_graph_current_);
	std::tie(line_process_, valid_edges_num_) = ComputeLineprocess(
			*pose_graph_current_, zeta_, option_);
}

std::shared_ptr<PoseGraph> OptimizationStatusGaussNewton::UpdatePruneInClass() {
	return PruneInvalidEdges(*pose_graph_current_, line_process_, option_);
}

void OptimizationStatusLevenbergMarquardt::InitLM() {
	ni_ = 2.0;
	rho_ = 0.0;
	tau_ = 1e-5;
	H_I_ = Eigen::MatrixXd::Identity(n_nodes_ * 6, n_nodes_ * 6);
	current_lambda_ = tau_ * H_.diagonal().maxCoeff();
}

void OptimizationStatusLevenbergMarquardt::SolveLinearSystemInClass() {
	Eigen::MatrixXd H_LM = H_ + current_lambda_ * H_I_;
	delta_ = H_LM.ldlt().solve(b_);
}

void OptimizationStatusLevenbergMarquardt::ComputeRho() {
	rho_ = (current_residual_ - new_residual_) /
			(delta_.dot(current_lambda_ * delta_ + b_) + 1e-3);
}

void OptimizationStatusLevenbergMarquardt::ComputeGain() {
	double alpha = 1. - pow((2 * rho_ - 1), 3);
	alpha = (std::min)(alpha, option_.upper_scale_factor_);
	double scale_factor = (std::max)(option_.lower_scale_factor_, alpha);
	current_lambda_ *= scale_factor;
	ni_ = 2;
}

void OptimizationStatusLevenbergMarquardt::ResetGain() {
	current_lambda_ *= ni_;
	ni_ *= 2;
}

std::shared_ptr<PoseGraph> GlobalOptimization(
		const PoseGraph &pose_graph, 
		const GraphOptimizationMethod &method,
		/* GraphOptimizationLevenbergMethodMarquardt() */
		const GlobalOptimizationOption &option 
		/* = GlobalOptimizationOption() */)
{
	return method.OptimizePoseGraph(pose_graph, option);
	//std::shared_ptr<PoseGraph> test;
	//return test;
}

}	// namespace three
