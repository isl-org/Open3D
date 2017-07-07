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

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d nodes and %d edges\n", 
			n_nodes, n_edges);

	InitAnalysticalJacobian();

	Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b(n_nodes * 6);
	Eigen::VectorXd x(n_nodes * 6);
	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	std::vector<Eigen::Matrix4d> node_matrix_array;
	std::vector<Eigen::Matrix4d> nodeinv_matrix_array;
	std::vector<Eigen::Matrix4d> xinv_matrix_array;
	node_matrix_array.resize(n_nodes);
	nodeinv_matrix_array.resize(n_nodes);
	xinv_matrix_array.resize(n_edges);	
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		const PoseGraphNode &t = pose_graph.nodes_[iter_node];
		node_matrix_array[iter_node] = t.pose_;
		nodeinv_matrix_array[iter_node] = t.pose_.inverse();
		x.block<6, 1>(iter_node * 6, 0) = TransformMatrix4dToVector6d(t.pose_);
	}
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		xinv_matrix_array[iter_edge] = t.transformation_.inverse();
	}

	for (int iter = 0; iter < MAX_ITER; iter++) {
		H.setZero();
		b.setZero();		
		double total_residual = 0.0;
		int line_process_cnt = 0;
		
		// build information matrix of the system
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
			Eigen::Vector6d e = GetDiffVec(
					xinv_matrix_array[iter_edge],
					node_matrix_array[t.source_node_id_],
					nodeinv_matrix_array[t.target_node_id_]);
			double residual = e.transpose() * t.information_ * e;
			
			//if (iter_edge == 100)
			//	std::cout << e.transpose() << std::endl;
			
			Eigen::Matrix6d J_source, J_target;
			//std::tie(J_source, J_target) = GetNumericalJacobian(
			//	xinv_matrix_array[iter_edge],
			//	node_matrix_array[t.source_node_id_],
			//	nodeinv_matrix_array[t.target_node_id_]);
			std::tie(J_source, J_target) = GetAnalysticalJacobian(
				xinv_matrix_array[iter_edge],
				node_matrix_array[t.source_node_id_],
				nodeinv_matrix_array[t.target_node_id_]);
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
			//if (iter_edge == 10) {
			//	std::cout << "Numeric" << std::endl;
			//	std::cout << J_source << std::endl;
			//	std::cout << J_target << std::endl;
			//}				
		}
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual);

		//std::ofstream file_H("H.txt");
		//file_H << H;
		//file_H.close();

		//std::ofstream file_b("b.txt");
		//file_b << b;
		//file_b.close();

		//std::cout << H.block<6, 6>(90, 90) << std::endl;
		//std::cout << H.block<6, 6>(96, 96) << std::endl;

		// why determinant of H is inf?
		H += 10 * Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6); // simple LM
		Eigen::VectorXd delta = -H.ldlt().solve(b);
		//std::cout << "delta.norm()" << delta.norm() << std::endl;
		//x += delta;

		// update pose of nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			node_matrix_array[iter_node] = 
					TransformVector6dToMatrix4d(delta_iter) * 
					node_matrix_array[iter_node];
			//Eigen::Vector6d x_iter = x.block<6, 1>(iter_node * 6, 0);
			//node_matrix_array[iter_node] = TransformVector6dToMatrix4d(x_iter);
			nodeinv_matrix_array[iter_node] = 
					node_matrix_array[iter_node].inverse();
		}
		
		// update line process only for loop edges
		line_process_cnt = 0;
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
			if (abs(t.target_node_id_ - t.source_node_id_) != 1) { 
				Eigen::Vector6d e = GetDiffVec(
						xinv_matrix_array[iter_edge],
						node_matrix_array[t.source_node_id_],
						nodeinv_matrix_array[t.target_node_id_]);
				double residual_square = 
						e.transpose() * t.information_ * e;
				double temp = MU / (MU + residual_square);
				double temp2 = temp * temp;
				if (temp2 < PRUNE) // prunning
					line_process(line_process_cnt++) = 0.0;
				else
					line_process(line_process_cnt++) = temp2;
			}
		}
		if (stopping_criterion()) // todo: adding stopping criterion
			break;
	}

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		PoseGraphNode new_node(node_matrix_array[iter_node]);
		pose_graph_refined->nodes_.push_back(new_node);
	}
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		Eigen::Vector6d diff_vec = GetDiffVec(
				xinv_matrix_array[iter_edge],
				node_matrix_array[t.source_node_id_],
				node_matrix_array[t.target_node_id_]);
		PoseGraphEdge new_edge(t.source_node_id_, t.target_node_id_, 
				TransformVector6dToMatrix4d(diff_vec),
				Eigen::Matrix6d::Identity(), false);
		pose_graph_refined->edges_.push_back(new_edge);
	}
	return pose_graph_refined;
}

}	// namespace three
