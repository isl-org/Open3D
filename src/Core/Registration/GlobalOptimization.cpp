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
#include <json/json.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>

namespace three{

namespace {

const int MAX_ITER = 100;
const int MAX_LM_ITER = 20;
const double MU = 10000;
const double PRUNE = 0.25;
const double NUMERICAL_DELTA = 1e-9;
const double EPS_1 = 1e-6;
const double EPS_2 = 1e-6;
const double EPS_3 = 1e-6;
const double EPS_4 = 1e-6;
const double EPS_5 = 1e-6;
const double _goodStepUpperScale = 2./3.;
const double _goodStepLowerScale = 1./3.;
std::vector<Eigen::Matrix4d> diff;

inline Eigen::Vector6d GetApproximate6DVector(Eigen::Matrix4d input)
{
	Eigen::Vector6d output;
	output(0) = (-input(1, 2) + input(2, 1)) / 2.0;
	output(1) = (-input(2, 0) + input(0, 2)) / 2.0;
	output(2) = (-input(0, 1) + input(1, 0)) / 2.0;
	output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
	return std::move(output);
}

/// todo: we may also do batch inverse for T_j
inline Eigen::Vector6d GetDiffVec(const Eigen::Matrix4d &X_inv, 
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv)
{
	Eigen::Matrix4d temp;
	temp.noalias() = X_inv * T_j_inv * T_i;
	return GetApproximate6DVector(temp);
}

inline Eigen::Matrix4d LinearizedSmallTransform(Eigen::Vector6d delta) 
{
	Eigen::Matrix4d delta_mat;
	delta_mat << 1, -delta(2), delta(1), delta(3),
			delta(2), 1, -delta(0), delta(4),
			-delta(1), delta(0), 1, delta(5),
			0, 0, 0, 1;
	return delta_mat;
}

inline Eigen::Matrix4d GetIncrementForTarget(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv, 
		const Eigen::Vector6d &delta)
{
	Eigen::Matrix4d output = 
			X_inv * T_j_inv * LinearizedSmallTransform(delta).inverse() * T_i;
	return std::move(output);
}

inline Eigen::Matrix4d GetIncrementForSource(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv,
		const Eigen::Vector6d &delta)
{
	Eigen::Matrix4d output = 
			X_inv * T_j_inv * LinearizedSmallTransform(delta) * T_i;
	return std::move(output);
}

typedef std::function<Eigen::Matrix4d(const Eigen::Matrix4d &,
		const Eigen::Matrix4d &, const Eigen::Matrix4d &,
		const Eigen::Vector6d &)> function_type;

inline Eigen::Matrix6d GetSingleNumericalJacobian(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv, 
		function_type &f)
{
	Eigen::Matrix6d output;
	Eigen::Vector6d delta;
	output.setZero();
	for (int i = 0; i < 6; i++) {
		delta.setZero();
		delta(i) = NUMERICAL_DELTA;
		Eigen::Vector6d temp_p = GetApproximate6DVector(
				f(X_inv, T_i, T_j_inv, delta));
		Eigen::Vector6d temp_n = GetApproximate6DVector(
				f(X_inv, T_i, T_j_inv, -delta));
		output.block<6, 1>(0, i) = (temp_p - temp_n) / (2.0 * NUMERICAL_DELTA);
	}
	return std::move(output);
}

inline std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> GetNumericalJacobian(
		const Eigen::Matrix4d &X_inv, const Eigen::Matrix4d &T_i, 
		const Eigen::Matrix4d &T_j_inv)
{
	function_type function_J_source = &GetIncrementForSource;
	function_type function_J_target = &GetIncrementForTarget;
	Eigen::Matrix6d J_source = GetSingleNumericalJacobian(
			X_inv, T_i, T_j_inv, function_J_source);
	Eigen::Matrix6d J_target = GetSingleNumericalJacobian(
			X_inv, T_i, T_j_inv, function_J_target);
	return std::make_tuple(std::move(J_source),std::move(J_target));
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

void InitAnalysticalJacobianOperators()
{
	diff.clear();
	Eigen::Matrix4d diff_alpha, diff_beta, diff_gamma, diff_a, diff_b, diff_c;
	diff_alpha.setZero();
	diff_alpha(1, 2) = -1;
	diff_alpha(2, 1) = 1;
	diff_beta.setZero();
	diff_beta(2, 0) = -1;
	diff_beta(0, 2) = 1;
	diff_gamma.setZero();
	diff_gamma(0, 1) = -1;
	diff_gamma(1, 0) = 1;
	diff_a.setZero();
	diff_a(0, 3) = 1;
	diff_b.setZero();
	diff_b(1, 3) = 1;
	diff_c.setZero();
	diff_c(2, 3) = 1;
	diff.push_back(diff_alpha);
	diff.push_back(diff_beta);
	diff.push_back(diff_gamma);
	diff.push_back(diff_a);
	diff.push_back(diff_b);
	diff.push_back(diff_c);
}

inline Eigen::Matrix6d GetSingleAnalysticalJacobian(
		const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv)
{
	Eigen::Matrix6d output = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * T_j_inv * diff[i] * T_i;
		output.block<6, 1>(0, i) = GetApproximate6DVector(temp);
	}
	return std::move(output);
}

std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> GetAnalysticalJacobian(
		const Eigen::Matrix4d &X_inv, const Eigen::Matrix4d &T_i,
		const Eigen::Matrix4d &T_j_inv)
{
	Eigen::Matrix6d J_source =
			GetSingleAnalysticalJacobian(X_inv, T_i, T_j_inv);
	Eigen::Matrix6d J_target = -J_source;
	return std::make_tuple(std::move(J_source), std::move(J_target));
}

std::tuple<Eigen::VectorXd, int> ComputeLineprocess(
		const PoseGraph &pose_graph, const Eigen::VectorXd &evec) 
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();
	int line_process_cnt = 0;
	int valid_edges = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		if (abs(t.target_node_id_ - t.source_node_id_) != 1) {
			Eigen::Vector6d e = evec.block<6, 1>(iter_edge * 6, 0);
			double residual_square = e.transpose() * t.information_ * e;
			double temp = MU / (MU + residual_square);
			double temp2 = temp * temp;
			if (temp2 < PRUNE) {// prunning
				line_process(line_process_cnt++) = 0.0;
			} else {
				line_process(line_process_cnt++) = temp2;				
				valid_edges++;
			}				
		}
	}
	return std::make_tuple(std::move(line_process), valid_edges);
}

std::tuple<Eigen::VectorXd, double> ComputeE(const PoseGraph &pose_graph, 
		Eigen::VectorXd line_process) 
{
	int n_edges = (int)pose_graph.edges_.size();
	double residual = 0.0;
	Eigen::VectorXd output(n_edges * 6);
	int line_process_cnt = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
		Eigen::Vector6d e = GetDiffVec(X_inv, Ts, Tt_inv);
		output.block<6, 1>(iter_edge * 6, 0) = e;
		const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		double line_process_iter = 1.0;
		if (abs(te.source_node_id_ - te.target_node_id_) != 1)
			line_process_iter = line_process(line_process_cnt++);
		residual += line_process_iter * e.transpose() * te.information_ * e + 
			MU * pow(sqrt(line_process_iter)-1,2.0);
	}
	return std::make_tuple(std::move(output), residual);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeH(
		const PoseGraph &pose_graph, const Eigen::VectorXd &evec,
		const Eigen::VectorXd &line_process)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	int line_process_cnt = 0;
	double total_residual = 0.0;
	Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b(n_nodes * 6);
	H.setZero();
	b.setZero();

	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		Eigen::Vector6d e = evec.block<6, 1>(iter_edge * 6, 0);
		
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);

		Eigen::Matrix6d J_source, J_target;
		//std::tie(J_source, J_target) = GetNumericalJacobian(X_inv, Ts, Tt_inv);
		std::tie(J_source, J_target) = GetAnalysticalJacobian(X_inv, Ts, Tt_inv);		

		Eigen::Matrix6d J_sourceT_Info =
				J_source.transpose() * t.information_;
		Eigen::Matrix6d J_targetT_Info =
				J_target.transpose() * t.information_;
		Eigen::Vector6d eT_Info = e.transpose() * t.information_;

		double line_process_iter = 1.0;
		if (abs(t.target_node_id_ - t.source_node_id_) != 1) {
			line_process_iter = line_process(line_process_cnt++);
		}
		int id_i = t.source_node_id_ * 6;
		int id_j = t.target_node_id_ * 6;
		H.block<6, 6>(id_i, id_i).noalias() +=
				line_process_iter * J_sourceT_Info * J_source;
		H.block<6, 6>(id_i, id_j).noalias() +=
				line_process_iter * J_sourceT_Info * J_target;
		H.block<6, 6>(id_j, id_i).noalias() +=
				line_process_iter * J_targetT_Info * J_source;
		H.block<6, 6>(id_j, id_j).noalias() +=
				line_process_iter * J_targetT_Info * J_target;
		b.block<6, 1>(id_i, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * J_source;
		b.block<6, 1>(id_j, 0).noalias() -=
				line_process_iter * eT_Info.transpose() * J_target;
	}
	return std::make_tuple(std::move(H), std::move(b));
}

Eigen::VectorXd UpdateX(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	Eigen::VectorXd output(n_nodes * 6);	
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d output_iter = GetApproximate6DVector(
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
		Eigen::VectorXd line_process)
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
		if (abs(t.target_node_id_ - t.source_node_id_) != 1) {
			if (line_process(line_process_cnt++) > PRUNE) {
				pose_graph_pruned->edges_.push_back(t);
			}
		} else {
			pose_graph_pruned->edges_.push_back(t);
		}
	}
	return pose_graph_pruned;
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d nodes and %d edges\n", 
			n_nodes, n_edges);

	InitAnalysticalJacobianOperators();

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::VectorXd evec;
	double total_residual;
	std::tie(evec, total_residual) = ComputeE(*pose_graph_refined, line_process);

	for (int iter = 0; iter < MAX_ITER; iter++) {
		
		int line_process_cnt = 0;

		Eigen::MatrixXd H;
		Eigen::VectorXd b;
		std::tie(H, b) = ComputeH(
				*pose_graph_refined, evec, line_process);
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual);

		Eigen::VectorXd delta = H.colPivHouseholderQr().solve(b);
		Eigen::VectorXd err = H*(delta)-b;
		std::cout << "err norm : " << err.norm() << std::endl;
		
		// update pose of nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			pose_graph_refined->nodes_[iter_node].pose_ = 
					TransformVector6dToMatrix4d(delta_iter) * 
					pose_graph_refined->nodes_[iter_node].pose_;
		}
		std::tie(evec, total_residual) = ComputeE(*pose_graph_refined, line_process);
		int valid_edges;
		std::tie(line_process, valid_edges) = 
				ComputeLineprocess(*pose_graph_refined, evec);

		// update line process only for loop edges
	}
	return pose_graph_refined;
}

std::shared_ptr<PoseGraph> GlobalOptimizationLM(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d nodes and %d edges\n",
			n_nodes, n_edges);

	InitAnalysticalJacobianOperators();

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::VectorXd evec;
	double current_residual, new_residual;
	std::tie(evec, new_residual) = ComputeE(*pose_graph_refined, line_process);
	current_residual = new_residual;

	int valid_edges;
	std::tie(line_process, valid_edges) = ComputeLineprocess(*pose_graph_refined, evec);

	Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	Eigen::VectorXd x = UpdateX(*pose_graph_refined);

	std::tie(H, b) = ComputeH(*pose_graph_refined, evec, line_process);

	Eigen::VectorXd H_diag = H.diagonal();
	double tau = 1e-5;
	double current_lambda = tau * H_diag.maxCoeff();
	double ni = 2.0;
	double rho = 0.0;

	PrintDebug("[Initial     ] residual : %e, lambda : %e\n",
			current_residual, current_lambda);

	bool stop = false;
	if (b.maxCoeff() < EPS_1) {
		PrintWarning("[Job finished] b is near zero.\n");
		stop = true;
	}

	int iter;
	for (iter = 0; iter < MAX_ITER && !stop; iter++) {
		int lm_count = 0;
		do {
			Eigen::MatrixXd H_LM = H + current_lambda * H_I;
			Eigen::VectorXd delta = H_LM.colPivHouseholderQr().solve(b);
			
			double solver_error = (H_LM*(delta)-b).norm();
			if (solver_error > EPS_5) {
				PrintWarning("[Job finished] error norm %e is higher than %e\n",
						solver_error, EPS_5);
				stop = true;
			}

			if (delta.norm() < EPS_2 * (x.norm() + EPS_2)) {
				stop = true;
				PrintDebug("[Job finished] delta.norm() < %e * (x.norm() + %e)\n", 
						EPS_2, EPS_2);
			} else {
				std::shared_ptr<PoseGraph> pose_graph_refined_new =
						UpdatePoseGraph(*pose_graph_refined, delta);

				Eigen::VectorXd evec_new;
				std::tie(evec_new, new_residual) = 
						ComputeE(*pose_graph_refined_new, line_process);
				rho = (current_residual - new_residual) / 
						(delta.dot(current_lambda * delta + b) + 1e-3);
				if (rho > 0) {
					if (current_residual - new_residual < EPS_4 * evec.norm()) {
						stop = true;
						PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n", EPS_4);
					}
					double alpha = 1. - pow((2 * rho - 1), 3);
					alpha = (std::min)(alpha, _goodStepUpperScale);
					double scaleFactor = (std::max)(_goodStepLowerScale, alpha);
					current_lambda *= scaleFactor;
					ni = 2;
					current_residual = new_residual;

					evec = evec_new;
					*pose_graph_refined = *pose_graph_refined_new;
					x = UpdateX(*pose_graph_refined);
					std::tie(line_process, valid_edges) = ComputeLineprocess(
							*pose_graph_refined, evec);
					std::tie(H, b) = ComputeH(*pose_graph_refined, 
							evec, line_process);
					
					if (b.maxCoeff() < EPS_1) {
						stop = true;
						PrintDebug("[Job finished] b.maxCoeff() < %e\n", EPS_1);
					}
				} else {
					current_lambda *= ni;
					ni *= 2;
				}
			}
			lm_count++;
			if (lm_count > MAX_LM_ITER) {
				stop = true;
				PrintDebug("[Job finished] lm_count > %d\n", MAX_LM_ITER);
			}
		} while (!((rho > 0) || stop));
		if (!stop) {
			PrintDebug("[Iteration %02d] residual : %e, lambda : %e, valid edges : %d/%d\n",
					iter, current_residual, current_lambda, 
					valid_edges, n_edges - (n_nodes - 1));
		}
		if (current_residual < EPS_3) {
			stop = true;
			PrintDebug("[Job finished] current_residual < %e\n", EPS_3);
		}			
	}	// end for
	if (iter == MAX_ITER) {
		stop = true;
		PrintDebug("[Job finished] reached maximum number of iterations\n", EPS_3);
	}

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned =
			PruneInvalidEdges(*pose_graph_refined, line_process);
	return pose_graph_refined_pruned;
}

}	// namespace three
