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

#include <iostream> // for debugging
#include <fstream> // for debugging
#include <Eigen/Dense>
#include <json/json.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>

namespace three{

namespace {

const int MAX_ITER = 100;
const double MU = 100;
const double PRUNE = 0.25;
const double DELTA = 1e-9;

bool stopping_criterion(/* what could be an input for this function? */) {
	return false;
}

inline Eigen::Vector6d Extract6DVector(Eigen::Matrix4d input)
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
	//return TransformMatrix4dToVector6d(temp);
	return Extract6DVector(temp);
}

inline Eigen::Matrix4d LinearizedSmallTransform(Eigen::Vector6d delta) {
	Eigen::Matrix4d delta_mat;
	//delta_mat << 1, -delta(2), delta(1), delta(3),
	//			delta(2), 1, -delta(0), delta(4),
	//			-delta(1), delta(0), 1, delta(5),
	//			0, 0, 0, 1;
	delta_mat = TransformVector6dToMatrix4d(delta);
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
		delta(i) = DELTA;
		Eigen::Vector6d temp_p = Extract6DVector(
				f(X_inv, T_i, T_j_inv, delta));
		Eigen::Vector6d temp_n = Extract6DVector(
				f(X_inv, T_i, T_j_inv, -delta));
		output.block<6, 1>(0, i) = (temp_p - temp_n) / (2.0 * DELTA);
	}
	return std::move(output);
}

inline std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> GetNumericalJacobian(
	const Eigen::Matrix4d &X_inv, const Eigen::Matrix4d &T_i, 
	const Eigen::Matrix4d &T_j_inv)
{
	function_type function_J_source = &GetIncrementForSource;
	function_type function_J_target = &GetIncrementForTarget;
	Eigen::Matrix6d J_source = 
			GetSingleNumericalJacobian(X_inv, T_i, T_j_inv, function_J_source);
	Eigen::Matrix6d J_target = 
			GetSingleNumericalJacobian(X_inv, T_i, T_j_inv, function_J_target);
	return std::make_tuple(std::move(J_source),std::move(J_target));
}

std::vector<Eigen::Matrix4d> diff;

void InitAnalysticalJacobian()
{
	diff.clear();
	Eigen::Matrix4d diff_alpha = Eigen::Matrix4d::Zero();
	Eigen::Matrix4d diff_beta = Eigen::Matrix4d::Zero();
	Eigen::Matrix4d diff_gamma = Eigen::Matrix4d::Zero();
	Eigen::Matrix4d diff_a = Eigen::Matrix4d::Zero();
	Eigen::Matrix4d diff_b = Eigen::Matrix4d::Zero();
	Eigen::Matrix4d diff_c = Eigen::Matrix4d::Zero();
	diff_alpha(1, 2) = -1;
	diff_alpha(2, 1) = 1;
	diff_beta(2, 0) = -1;
	diff_beta(0, 2) = 1;	
	diff_gamma(0, 1) = -1;
	diff_gamma(1, 0) = 1;
	diff_a(0, 3) = 1;
	diff_b(1, 3) = 1;
	diff_c(2, 3) = 1;
	diff.push_back(diff_alpha);
	diff.push_back(diff_beta);
	diff.push_back(diff_gamma);
	diff.push_back(diff_a);
	diff.push_back(diff_b);
	diff.push_back(diff_c);
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

inline Eigen::Matrix6d GetSingleAnalysticalJacobian(
	const Eigen::Matrix4d &X_inv,
	const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv)
{
	Eigen::Matrix6d output = Eigen::Matrix6d::Zero();
	for (int i = 0; i < 6; i++) {
		Eigen::Matrix4d temp = X_inv * T_j_inv * diff[i] * T_i;
		output.block<6, 1>(0, i) = Extract6DVector(temp);
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

Eigen::VectorXd ComputeLineprocess(
	const PoseGraph &pose_graph, const Eigen::VectorXd &evec) 
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	int line_process_cnt = 0;
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		if (abs(t.target_node_id_ - t.source_node_id_) != 1) {
			Eigen::Vector6d e = evec.block<6, 1>(iter_edge * 6, 0);
			double residual_square = e.transpose() * t.information_ * e;
			double temp = MU / (MU + residual_square);
			double temp2 = temp * temp;
			if (temp2 < PRUNE) // prunning
				line_process(line_process_cnt++) = 0.0;
			else
				line_process(line_process_cnt++) = temp2;
		}
	}
	return std::move(line_process);
}

Eigen::VectorXd ComputeE(const PoseGraph &pose_graph) 
{
	int n_edges = (int)pose_graph.edges_.size();
	Eigen::VectorXd output(n_edges * 6);
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
		Eigen::Vector6d e = GetDiffVec(X_inv, Ts, Tt_inv);
		output.block<6, 1>(iter_edge * 6, 0) = e;
	}
	return std::move(output);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double> ComputeH(
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
		double residual = e.transpose() * t.information_ * e;

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
		b.block<6, 1>(id_i, 0).noalias() +=
			line_process_iter * eT_Info.transpose() * J_source;
		b.block<6, 1>(id_j, 0).noalias() +=
			line_process_iter * eT_Info.transpose() * J_target;
		total_residual += line_process_iter * residual;
	}
	return std::make_tuple(std::move(H), std::move(b), total_residual);
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d nodes and %d edges\n", 
			n_nodes, n_edges);

	InitAnalysticalJacobian();

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::VectorXd evec = ComputeE(*pose_graph_refined);

	for (int iter = 0; iter < MAX_ITER; iter++) {
		
		int line_process_cnt = 0;

		Eigen::MatrixXd H;
		Eigen::VectorXd b;
		double total_residual;
		std::tie(H, b, total_residual) = ComputeH(
				*pose_graph_refined, evec, line_process);
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual);

		// why determinant of H is inf?
		H += 10 * Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6); // simple LM
		Eigen::VectorXd delta = -H.ldlt().solve(b);

		// update pose of nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			pose_graph_refined->nodes_[iter_node].pose_ = 
					TransformVector6dToMatrix4d(delta_iter) * 
					pose_graph_refined->nodes_[iter_node].pose_;
		}
		evec = ComputeE(*pose_graph_refined);
		line_process = ComputeLineprocess(*pose_graph_refined, evec);

		// update line process only for loop edges
		
		if (stopping_criterion()) // todo: adding stopping criterion
			break;
	}

	return pose_graph_refined;
}

std::shared_ptr<PoseGraph> GlobalOptimizationLM(const PoseGraph &pose_graph)
{
	////////////////////////////////
	////// codes working for LM
	//Eigen::VectorXd H_diag = H.diagonal();
	//double tau = 1.0; // not sure about tau
	//double vu = 2.0;
	//double lambda = tau * H_diag.maxCoeff();
	//bool stop = false;
	//if (b.maxCoeff() < EPS_1) // b is near zero. Bad condition.
	//	stop = true;
	//double rho = 1.0;

	//for (int inner_iter = 0; inner_iter < MAX_ITER && !stop; inner_iter++) {
	//	do {
	//		Eigen::MatrixXd H_LM = H + lambda * H_I;
	//		Eigen::VectorXd delta = -H_LM.ldlt().solve(b);
	//		if (delta.norm() < EPS_2 * (x.norm() + EPS_2)) {
	//			stop = true;
	//		}
	//		else {
	//			for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
	//				Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
	//				node_matrix_array[iter_node] =
	//					TransformVector6dToMatrix4d(delta_iter) *
	//					node_matrix_array[iter_node];
	//				nodeinv_matrix_array[iter_node] =
	//					node_matrix_array[iter_node].inverse();
	//				x.block<6, 1>(iter_node * 6, 0) =
	//					TransformMatrix4dToVector6d(node_matrix_array[iter_node]);
	//			}
	//			Eigen::VectorXd e_new(n_edges * 6);
	//			for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
	//				const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
	//				Eigen::Vector6d e_iter = GetDiffVec(
	//					xinv_matrix_array[iter_edge],
	//					node_matrix_array[t.source_node_id_],
	//					nodeinv_matrix_array[t.target_node_id_]);
	//				e_new.block<6, 1>(iter_edge * 6, 0) = e_iter;
	//			}
	//			rho = (e.norm() - e_new.norm()) /
	//				(delta.transpose() * (lambda * delta + b));
	//			if (rho > 0) {
	//				if (e.norm() - e_new.norm() < EPS_4 * e.norm())
	//					stop = true;
	//				// todo: Update H, b, and e = e_new, 
	//				stop = stop || (b.maxCoeff() < EPS_1);
	//				lambda = lambda * fmax(1 / 3, 1 - pow(2 * rho - 1, 3.0));
	//				vu = 2;
	//			}
	//			else {
	//				lambda = lambda * vu;
	//				vu = 2 * vu;
	//			}
	//		}
	//	} while ((rho > 0) || stop);
	//	stop = e.norm() < EPS_3;
	//}	// end for
	//	//////////////////////////////

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	return pose_graph_refined;
}

}	// namespace three
