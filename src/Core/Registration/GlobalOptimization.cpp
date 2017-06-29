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

#include <json/json.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>

namespace three{

namespace {

const int MAX_ITER = 100;

// matrix solver and vector-to-4x4 matrix things
// they should be in core/eigen?

Eigen::Matrix4d From6DVectorTo4x4Matrix(const Eigen::Vector6d &vec)
{
	// need to be implemented.
	return Eigen::Matrix4d::Identity();
}

Eigen::Vector6d From4x4MatrixTo6DVector(const Eigen::Matrix4d &mat)
{
	// need to be implemented.
	return Eigen::Vector6d::Zero();
}

bool stopping_criterion(/* what could be input for this function? */) {
	return false;
}

}	// unnamed namespace

std::shared_ptr<PoseGraph> GlobalOptimization(const PoseGraph &pose_graph)
{
	int n_nodes = pose_graph.nodes_.size();
	int n_edges = pose_graph.edges_.size();

	Eigen::MatrixXd J(n_edges + n_nodes, n_nodes * 6);
	Eigen::VectorXd r(n_edges + n_nodes);
	Eigen::VectorXd x(n_nodes * 6);

	// main iteration
	for (int iter = 0; iter < MAX_ITER; iter++) {
		
		// depends on the definition of nodes maybe need to do (n_nodes + 1)?
		J.setZero();
		r.setZero();
		x.setZero();
		Eigen::VectorXd line_process(n_edges);
		line_process.setOnes();

		// sub-iteration #1
		// building jacobian for pose graph nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Matrix4d pose = pose_graph.nodes_[iter_node].pose_;
			Eigen::Vector6d pose_vec = From4x4MatrixTo6DVector(pose);
			/* = why the node does not have information? */
			Eigen::Matrix6d information = Eigen::Matrix6d::Identity(); 
			//b.block<6, 1>(iter_node * 6, 0) = pose_vec;
			double residual = pose_vec.transpose() * information * pose_vec;
			Eigen::Vector6d J_vec = 
					(pose_vec.transpose() * (information + information.transpose())) /
					(2 * sqrt(residual));
			int source_node_id = iter_node;
			int target_node_id = iter_node + 1;
			int row_id = iter_node;			
			J.block<6, 1>(row_id, source_node_id * 6) = J_vec;
			J.block<6, 1>(row_id, target_node_id * 6) = -J_vec;
			r(row_id) = residual;
		}
		// building jacobian for pose graph edges
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			Eigen::Matrix4d trans = pose_graph.edges_[iter_edge].transformation_;
			Eigen::Vector6d trans_vec = From4x4MatrixTo6DVector(trans);
			Eigen::Matrix6d information = pose_graph.edges_[iter_edge].information_;
			//b.block<6, 1>(iter_edge * 6, 0) = trans_vec;
			double residual = trans_vec.transpose() * information * trans_vec;
			Eigen::Vector6d J_vec =
				(trans_vec.transpose() * (information + information.transpose())) /
				(2 * sqrt(trans_vec.transpose() * information * trans_vec));
			int source_node_id = pose_graph.edges_[iter_edge].source_node_id_;
			int target_node_id = pose_graph.edges_[iter_edge].target_node_id_;
			double line_process_sqrt = sqrt(line_process(iter_edge));
			int row_id = n_nodes + iter_edge;
			J.block<6, 1>(row_id, source_node_id * 6) = line_process_sqrt * J_vec;
			J.block<6, 1>(row_id, target_node_id * 6) = line_process_sqrt * -J_vec;
			r(row_id) = residual;
		}
		// solve equation
		Eigen::MatrixXd JtJ = J.transpose() * J;
		Eigen::MatrixXd Jtr = J.transpose() * r;
		//x += -1 * JtJ.llt() * Jtr; /* not sure. Use solver in Eigen */

		// sub-iteration #2
		// update line process
		// just right after sub-iteration #1?
		for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
			int source_node_id = pose_graph.edges_[iter_edge].source_node_id_;
			int target_node_id = pose_graph.edges_[iter_edge].target_node_id_;
			Eigen::Vector6d source_vec = x.block<6, 1>(source_node_id * 6, 1);
			Eigen::Vector6d target_vec = x.block<6, 1>(target_node_id * 6, 1);
			Eigen::Vector6d diff_vec = source_vec - target_vec;
			Eigen::Matrix6d information = pose_graph.edges_[iter_edge].information_;
			double residual = diff_vec.transpose() * information * diff_vec;
			double temp = 1.0 / (1.0 + residual);
			line_process(iter_edge) = temp * temp;
		}
		// adding stopping criterion

		if (stopping_criterion())
			break;
	}

	std::shared_ptr<PoseGraph> output_pose_graph = std::make_shared<PoseGraph>();
	for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
		Eigen::Vector6d pose_vec = x.block<6, 1>(iter_node * 6, 1);
		PoseGraphNode new_node(From6DVectorTo4x4Matrix(pose_vec));
		output_pose_graph->nodes_.push_back(new_node);
	}
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		int source_node_id = pose_graph.edges_[iter_edge].source_node_id_;
		int target_node_id = pose_graph.edges_[iter_edge].target_node_id_;
		Eigen::Vector6d source_vec = x.block<6, 1>(source_node_id * 6, 1);
		Eigen::Vector6d target_vec = x.block<6, 1>(target_node_id * 6, 1);
		Eigen::Vector6d diff_vec = source_vec - target_vec;
		PoseGraphEdge new_edge(source_node_id, target_node_id, 
				From6DVectorTo4x4Matrix(diff_vec),
				Eigen::Matrix6d::Identity(), false);
		output_pose_graph->edges_.push_back(new_edge);
	}

	return output_pose_graph;
}

}	// namespace three
