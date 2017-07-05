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
const double MU = 30000;
const double DELTA = 1e-9;

bool stopping_criterion(/* what could be an input for this function? */) {
	return false;
}

/// todo: we may also do batch inverse for T_j
inline Eigen::Vector6d GetDiffVec(const Eigen::Matrix4d &X_inv, 
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j)
{
	Eigen::Matrix4d temp;
	temp.noalias() = X_inv * T_j.inverse() * T_i;
	return TransformMatrix4dToVector6d(temp);
}

inline Eigen::Matrix4d LinearizedSmallTransform(Eigen::Vector6d delta) {
	Eigen::Matrix4d delta_mat;
	delta_mat << 1, -delta(2), delta(1), delta(3),
				delta(2), 1, -delta(0), delta(4),
				-delta(1), delta(0), 1, delta(5),
				0, 0, 0, 1;
	return delta_mat;
}

inline Eigen::Matrix4d GetIncrementalForJ(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv, 
		const Eigen::Vector6d delta)
{
	return X_inv * T_j_inv * LinearizedSmallTransform(delta).inverse() * T_i;
}

inline Eigen::Matrix4d GetIncrementalForI(const Eigen::Matrix4d &X_inv,
	const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv,
	const Eigen::Vector6d delta)
{
	return X_inv * T_j_inv * LinearizedSmallTransform(delta) * T_i;
}

inline Eigen::Matrix6d GetNumericalJacobian(const Eigen::Matrix4d &X_inv,
		const Eigen::Matrix4d &T_i, const Eigen::Matrix4d &T_j_inv, bool is_I)
{
	Eigen::Matrix6d output;
	Eigen::Vector6d delta;
	output.setZero();
	for (int i = 0; i < 6; i++) {
		delta.setZero();
		delta(i) = DELTA;
		if (is_I) {
			Eigen::Vector6d temp_p = TransformMatrix4dToVector6d(
				GetIncrementalForI(X_inv, T_i, T_j_inv, delta));
			Eigen::Vector6d temp_n = TransformMatrix4dToVector6d(
				GetIncrementalForI(X_inv, T_i, T_j_inv, -delta));
			output.block<6, 1>(0, i) = (temp_p - temp_n) / (2 * DELTA);
		} else {
			Eigen::Vector6d temp_p = TransformMatrix4dToVector6d(
				GetIncrementalForJ(X_inv, T_i, T_j_inv, delta));
			Eigen::Vector6d temp_n = TransformMatrix4dToVector6d(
				GetIncrementalForJ(X_inv, T_i, T_j_inv, -delta));
			output.block<6, 1>(0, i) = (temp_p - temp_n) / (2 * DELTA);
		}
	}
	return output;
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d edges and %d nodes\n", 
			n_nodes, n_edges);

	Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b(n_nodes * 6);
	std::vector<Eigen::Matrix4d> node_matrix_array;
	std::vector<Eigen::Matrix4d> xinv_matrix_array;
	node_matrix_array.resize(n_nodes);
	xinv_matrix_array.resize(n_edges);
	
	//x.setZero();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		const PoseGraphNode &t = pose_graph.nodes_[iter_node];
		node_matrix_array[iter_node] = t.pose_;
	}
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
		xinv_matrix_array[iter_edge] = t.transformation_.inverse();
	}

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	// main iteration
	for (int iter = 0; iter < MAX_ITER; iter++) {

		H.setZero();
		b.setZero();		
		double total_residual2 = 0.0;
	
		// depends on the definition of nodes maybe need to do (n_nodes + 1)?
		int line_process_cnt = 0;

		// building jacobian for loop edges
		// todo: this initialization should be done one time.
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
			Eigen::Vector6d trans_vec = GetDiffVec(
					xinv_matrix_array[iter_edge],
					node_matrix_array[t.source_node_id_],
					node_matrix_array[t.target_node_id_]);
			double residual = sqrt(trans_vec.transpose() * t.information_ * trans_vec);

			Eigen::Matrix6d J_i = 
					GetNumericalJacobian(xinv_matrix_array[iter_edge],
					node_matrix_array[t.source_node_id_],
					node_matrix_array[t.target_node_id_],
					true);
			Eigen::Matrix6d J_j =
					GetNumericalJacobian(xinv_matrix_array[iter_edge],
					node_matrix_array[t.source_node_id_],
					node_matrix_array[t.target_node_id_],
					false);
			//if (iter_edge == 0) {
			//	std::cout << J_vec_i.transpose() << std::endl;
			//	std::cout << J_vec_j.transpose() << std::endl;
			//}
			double line_process_sqrt = 1.0;
			if (abs(t.target_node_id_ - t.source_node_id_) != 1) // loop edge
				line_process_sqrt = sqrt(line_process(line_process_cnt++));
			//std::cout << iter_edge << " line_process_sqrt " << line_process_sqrt << std::endl;
			// this is what we are doing for GetDiffVec. Can be more efficient
			// can be more efficient by using block matrix multiplication
			int id_i = t.source_node_id_ * 6;
			int id_j = t.target_node_id_ * 6;
			H.block<6, 6>(id_i, id_i) += line_process_sqrt * J_i.transpose() * t.information_ * J_i;
			H.block<6, 6>(id_i, id_j) += line_process_sqrt * J_i.transpose() * t.information_ * J_j;
			H.block<6, 6>(id_j, id_i) += line_process_sqrt * J_j.transpose() * t.information_ * J_i;
			H.block<6, 6>(id_j, id_j) += line_process_sqrt * J_j.transpose() * t.information_ * J_j;
			// I am not sure about r.
			b.block<6, 1>(id_i, 0) += line_process_sqrt * trans_vec.transpose() * t.information_ * J_i;
			b.block<6, 1>(id_j, 0) += line_process_sqrt * trans_vec.transpose() * t.information_ * J_j;
			total_residual2 += line_process_sqrt * line_process_sqrt * residual * residual;
			//std::cout << trans_vec << std::endl;
			//std::cout << J_vec.transpose() << std::endl;
			//std::cout << J.block<1, 6>(row_id, t.source_node_id_ * 6) << std::endl;
			//std::cout << J.block<1, 6>(row_id, t.target_node_id_ * 6) << std::endl;
			//std::cout << r(row_id) << std::endl;
		}
		// solve equation
		//Eigen::MatrixXd JtJ = J.transpose() * J;
		//Eigen::MatrixXd Jtr = J.transpose() * r;
		
		//std::ofstream outfile;
		//outfile.open("JtJ.txt");
		//outfile << JtJ;
		//outfile.close();
		//outfile.open("Jtr.txt");
		//outfile << Jtr;
		//outfile.close();
		
		bool is_success;
		//Eigen::VectorXd delta;
		//std::tie(is_success, delta) = SolveLinearSystem(JtJ, Jtr); // determinant is always inf.
		Eigen::VectorXd delta = -H.ldlt().solve(b);

		//std::cout << delta << std::endl;
		
		// change 6d vector to matrix form
		// to save time for computing matrix form multiple times.
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			//std::cout << x_iter_node << std::endl;
			node_matrix_array[iter_node] =
				TransformVector6dToMatrix4d(delta_iter) *
				node_matrix_array[iter_node];
		}
		
		// sub-iteration #2
		// update line process
		// just right after sub-iteration #1?
		line_process_cnt = 0;
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
			if (abs(t.target_node_id_ - t.source_node_id_) != 1) { // only for loop edge
				Eigen::Vector6d diff_vec = GetDiffVec(
						xinv_matrix_array[iter_edge],
						node_matrix_array[t.source_node_id_],
						node_matrix_array[t.target_node_id_]);
				double residual_square = 
						diff_vec.transpose() * t.information_ * diff_vec;
				double temp = MU / (MU + residual_square);
				double temp2 = temp * temp;
				if (temp2 < 0.25)
					line_process(line_process_cnt++) = 0.0;
				else
					line_process(line_process_cnt++) = temp;
			}
		}
		// adding stopping criterion
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual2);

		//std::cout << line_process << std::endl;

		if (stopping_criterion())
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
