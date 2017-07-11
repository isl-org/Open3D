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
const double EPS_1 = 1e-6;
const double EPS_2 = 1e-6;
const double EPS_3 = 1e-6;
const double EPS_4 = 1e-6;
const double EPS_5 = 1e-6;
const double _goodStepUpperScale = 2./3.;
const double _goodStepLowerScale = 1./3.;

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

std::tuple<Eigen::VectorXd, double> ComputeE(const PoseGraph &pose_graph) 
{
	int n_edges = (int)pose_graph.edges_.size();
	double residual;
	Eigen::VectorXd output(n_edges * 6);
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X_inv, Ts, Tt_inv;
		std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
		Eigen::Vector6d e = GetDiffVec(X_inv, Ts, Tt_inv);
		output.block<6, 1>(iter_edge * 6, 0) = e;
		const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		residual += e.transpose() * te.information_ * e;
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

		//if (iter_edge == 100) {
		//	std::cout << J_source << std::endl;
		//	std::cout << J_target << std::endl;
		//}			

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

	Eigen::VectorXd evec;
	double total_residual;
	std::tie(evec, total_residual) = ComputeE(*pose_graph_refined);

	for (int iter = 0; iter < MAX_ITER; iter++) {
		
		int line_process_cnt = 0;

		Eigen::MatrixXd H;
		Eigen::VectorXd b;
		std::tie(H, b) = ComputeH(
				*pose_graph_refined, evec, line_process);
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual);

		// why determinant of H is inf?
		//H += 10 * Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6); // simple LM
		Eigen::VectorXd delta = H.colPivHouseholderQr().solve(b);
		Eigen::VectorXd err = H*(delta)-b;
		std::cout << "err norm : " << err.norm() << std::endl;
		
		//if (iter == 0) {
		//	//std::cout << "Saving matrix" << std::endl;
		//	//std::ofstream file("H.txt");
		//	//file << H;
		//	//file.close();
		//	//std::ofstream file2("b.txt");
		//	//file2 << b;
		//	//file2.close();
		//	std::cout << "Loading matrix" << std::endl;
		//	std::ifstream file3("x.txt");
		//	for (int iter_node = 0; iter_node < n_nodes * 6; iter_node++) {
		//		file3 >> delta(iter_node);
		//	}
		//	file3.close();
		//}

		// update pose of nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			pose_graph_refined->nodes_[iter_node].pose_ = 
					TransformVector6dToMatrix4d(delta_iter) * 
					pose_graph_refined->nodes_[iter_node].pose_;

			//if (iter_node == 100)
			//	std::cout << delta_iter << std::endl;
		}
		std::tie(evec, total_residual) = ComputeE(*pose_graph_refined);
		line_process = ComputeLineprocess(*pose_graph_refined, evec);

		// update line process only for loop edges
		
		if (stopping_criterion()) // todo: adding stopping criterion
			break;
	}

	return pose_graph_refined;
}

std::shared_ptr<PoseGraph> GlobalOptimizationLM(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("Optimizing PoseGraph having %d nodes and %d edges\n",
		n_nodes, n_edges);

	InitAnalysticalJacobian();

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd evec;
	double current_residual, new_residual;
	std::tie(evec, new_residual) = ComputeE(*pose_graph_refined);
	current_residual = new_residual;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
	Eigen::MatrixXd H;
	Eigen::VectorXd b;
	//Eigen::VectorXd x(n_nodes * 6);	
	std::tie(H, b) = ComputeH(
		*pose_graph_refined, evec, line_process);

	//////////////////////////////
	//// codes working for LM
	Eigen::VectorXd H_diag = H.diagonal();
	double tau = 1e-5;
	double currentLambda = tau * H_diag.maxCoeff();
	double ni = 2.0;
	double rho = 0.0;

	bool stop = false;
	if (b.maxCoeff() < EPS_1) // b is near zero. Bad condition.
		stop = true;	

	PrintDebug("[Initial     ] residual : %e, lambda : %e\n",
		current_residual, currentLambda);

	for (int iter = 0; iter < MAX_ITER && !stop; iter++) {
		int lm_count = 0;
		do {
			Eigen::MatrixXd H_LM = H + currentLambda * H_I;
			Eigen::VectorXd delta = H_LM.colPivHouseholderQr().solve(b);
			
			double solver_error = (H_LM*(delta)-b).norm();
			if (solver_error > EPS_5) {
				PrintWarning("[Job finished] error norm %e is higher than %e\n",
					solver_error, EPS_5);
				stop = true;
			}

			//if (delta.norm() < EPS_2 * (x.norm() + EPS_2)) {
			//	stop = true;
			//	std::cout << "delta.norm() < EPS_2 * (x.norm() + EPS_2)" 
			//		<< std::endl;
			//}
			//else {
				// update pose of nodes
				std::shared_ptr<PoseGraph> pose_graph_refined_new = 
						std::make_shared<PoseGraph>();
				*pose_graph_refined_new = *pose_graph_refined;
				for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
					Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
					pose_graph_refined_new->nodes_[iter_node].pose_ =
						TransformVector6dToMatrix4d(delta_iter) *
						pose_graph_refined_new->nodes_[iter_node].pose_;
				}
				// todo: update x as well
				Eigen::VectorXd evec_new;
				std::tie(evec_new, new_residual) = ComputeE(*pose_graph_refined_new);
				rho = (current_residual - new_residual) / (delta.dot(currentLambda * delta + b) + 1e-3);
				//std::cout << "rho : " << rho << std::endl;
				if (rho > 0) {
					if (current_residual - new_residual < EPS_4 * evec.norm()) {
						stop = true;
						PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n", EPS_4);
					}
					double alpha = 1. - pow((2 * rho - 1), 3);
					// crop lambda between minimum and maximum factors
					alpha = (std::min)(alpha, _goodStepUpperScale);
					double scaleFactor = (std::max)(_goodStepLowerScale, alpha);
					currentLambda *= scaleFactor;
					ni = 2;
					current_residual = new_residual;
					//_optimizer->discardTop();

					evec = evec_new;
					*pose_graph_refined = *pose_graph_refined_new;
					std::tie(H, b) = ComputeH(
						*pose_graph_refined, evec, line_process);
					
					if (b.maxCoeff() < EPS_1) {
						stop = true;
						PrintDebug("[Job finished] b.maxCoeff() < %e\n", EPS_1);
					}					
				}
				else {
					currentLambda *= ni;
					ni *= 2;
				}
			lm_count++;
			//}
			//PrintDebug("[LM Loop %02d] residual : %e, lambda : %e\n", 
			//	lm_count++, total_residual, lambda);
		} while (!((rho > 0) || stop));
		if (!stop) {
			PrintDebug("[Iteration %02d] residual : %e, lambda : %e, LM iteration : %d\n",
				iter, current_residual, currentLambda, lm_count);
		}
		if (current_residual < EPS_3)
			stop = true;
	}	// end for
	return pose_graph_refined;
}

//////////////////////////
///// from g2o
///// g2o testing function
Eigen::Quaterniond& normalize(Eigen::Quaterniond& q) {
	q.normalize();
	if (q.w()<0) {
		q.coeffs() *= -1;
	}
	return q;
}

inline std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, Eigen::Matrix4d>
GetRelativePosesG2O(const PoseGraph &pose_graph, int edge_id)
{
	const PoseGraphEdge &te = pose_graph.edges_[edge_id];
	const PoseGraphNode &ts = pose_graph.nodes_[te.source_node_id_];
	const PoseGraphNode &tt = pose_graph.nodes_[te.target_node_id_];
	Eigen::Matrix4d X = te.transformation_;
	Eigen::Matrix4d Ts = ts.pose_;
	Eigen::Matrix4d Tt = tt.pose_;
	return std::make_tuple(std::move(X), std::move(Ts), std::move(Tt));
}

Eigen::Vector3d toCompactQuaternion(const Eigen::Matrix3d& R) {
	Eigen::Quaterniond q(R);
	normalize(q);
	// return (x,y,z) of the quaternion
	return q.coeffs().head<3>();
}

Eigen::Matrix3d fromCompactQuaternion(const Eigen::Vector3d& v) {
	double w = 1 - v.squaredNorm();
	if (w<0)
		return Eigen::Matrix3d::Identity();
	else
		w = sqrt(w);
	return Eigen::Quaterniond(w, v[0], v[1], v[2]).toRotationMatrix();
}

inline Eigen::Isometry3d::ConstLinearPart extractRotation(const Eigen::Isometry3d &A)
{
	return A.matrix().topLeftCorner<3, 3>();
}

Eigen::Vector6d toVectorMQT(const Eigen::Isometry3d& t) {
	Eigen::Vector6d v;
	v.block<3, 1>(3, 0) = toCompactQuaternion(extractRotation(t));
	v.block<3, 1>(0, 0) = t.translation();
	return v;
}

Eigen::Isometry3d fromVectorMQT(const Eigen::Vector6d& v) {
	Eigen::Isometry3d t;
	t = fromCompactQuaternion(v.block<3, 1>(3, 0));
	t.translation() = v.block<3, 1>(0, 0);
	return t;
}

template <typename Derived, typename DerivedOther>
void skew(Eigen::MatrixBase<Derived>& Sx,
	Eigen::MatrixBase<Derived>& Sy,
	Eigen::MatrixBase<Derived>& Sz,
	const Eigen::MatrixBase<DerivedOther>& R) {
	const double
		r11 = 2 * R(0, 0), r12 = 2 * R(0, 1), r13 = 2 * R(0, 2),
		r21 = 2 * R(1, 0), r22 = 2 * R(1, 1), r23 = 2 * R(1, 2),
		r31 = 2 * R(2, 0), r32 = 2 * R(2, 1), r33 = 2 * R(2, 2);
	Sx << 0, 0, 0, -r31, -r32, -r33, r21, r22, r23;
	Sy << r31, r32, r33, 0, 0, 0, -r11, -r12, -r13;
	Sz << -r21, -r22, -r23, r11, r12, r13, 0, 0, 0;
}

template <typename Derived, typename DerivedOther>
inline void skewT(Eigen::MatrixBase<Derived>& Sx,
	Eigen::MatrixBase<Derived>& Sy,
	Eigen::MatrixBase<Derived>& Sz,
	const Eigen::MatrixBase<DerivedOther>& R) {
	const double
		r11 = 2 * R(0, 0), r12 = 2 * R(0, 1), r13 = 2 * R(0, 2),
		r21 = 2 * R(1, 0), r22 = 2 * R(1, 1), r23 = 2 * R(1, 2),
		r31 = 2 * R(2, 0), r32 = 2 * R(2, 1), r33 = 2 * R(2, 2);
	Sx << 0, 0, 0, r31, r32, r33, -r21, -r22, -r23;
	Sy << -r31, -r32, -r33, 0, 0, 0, r11, r12, r13;
	Sz << r21, r22, r23, -r11, -r12, -r13, 0, 0, 0;
}

template <typename Derived, typename DerivedOther>
inline void skewT(Eigen::MatrixBase<Derived>& s, const Eigen::MatrixBase<DerivedOther>& v) {
	const double x = 2 * v(0);
	const double y = 2 * v(1);
	const double z = 2 * v(2);
	s << 0., -z, y, z, 0, -x, -y, x, 0;
}

int _q2m(double& S, double& qw, const double&  r00, const double&  r10, const double&  r20, const double&  r01, const double&  r11, const double&  r21, const double&  r02, const double&  r12, const double&  r22) {
	double tr = r00 + r11 + r22;
	if (tr > 0) {
		S = sqrt(tr + 1.0) * 2; // S=4*qw 
		qw = 0.25 * S;
		// qx = (r21 - r12) / S;
		// qy = (r02 - r20) / S; 
		// qz = (r10 - r01) / S; 
		return 0;
	}
	else if ((r00 > r11)&(r00 > r22)) {
		S = sqrt(1.0 + r00 - r11 - r22) * 2; // S=4*qx 
		qw = (r21 - r12) / S;
		// qx = 0.25 * S;
		// qy = (r01 + r10) / S; 
		// qz = (r02 + r20) / S; 
		return 1;
	}
	else if (r11 > r22) {
		S = sqrt(1.0 + r11 - r00 - r22) * 2; // S=4*qy
		qw = (r02 - r20) / S;
		// qx = (r01 + r10) / S; 
		// qy = 0.25 * S;
		// qz = (r12 + r21) / S; 
		return 2;
	}
	else {
		S = sqrt(1.0 + r22 - r00 - r11) * 2; // S=4*qz
		qw = (r10 - r01) / S;
		// qx = (r02 + r20) / S;
		// qy = (r12 + r21) / S;
		// qz = 0.25 * S;
		return 3;
	}
}

void  compute_dq_dR_w(Eigen::Matrix<double, 3, 9 >&  dq_dR_w, const double&  qw, const double&  r00, const double&  r10, const double&  r20, const double&  r01, const double&  r11, const double&  r21, const double&  r02, const double&  r12, const double&  r22) {
	(void)r00;
	(void)r11;
	(void)r22;
	double  _aux1 = 1 / pow(qw, 3);
	double  _aux2 = -0.03125*(r21 - r12)*_aux1;
	double  _aux3 = 1 / qw;
	double  _aux4 = 0.25*_aux3;
	double  _aux5 = -0.25*_aux3;
	double  _aux6 = 0.03125*(r20 - r02)*_aux1;
	double  _aux7 = -0.03125*(r10 - r01)*_aux1;
	dq_dR_w(0, 0) = _aux2;
	dq_dR_w(0, 1) = 0;
	dq_dR_w(0, 2) = 0;
	dq_dR_w(0, 3) = 0;
	dq_dR_w(0, 4) = _aux2;
	dq_dR_w(0, 5) = _aux4;
	dq_dR_w(0, 6) = 0;
	dq_dR_w(0, 7) = _aux5;
	dq_dR_w(0, 8) = _aux2;
	dq_dR_w(1, 0) = _aux6;
	dq_dR_w(1, 1) = 0;
	dq_dR_w(1, 2) = _aux5;
	dq_dR_w(1, 3) = 0;
	dq_dR_w(1, 4) = _aux6;
	dq_dR_w(1, 5) = 0;
	dq_dR_w(1, 6) = _aux4;
	dq_dR_w(1, 7) = 0;
	dq_dR_w(1, 8) = _aux6;
	dq_dR_w(2, 0) = _aux7;
	dq_dR_w(2, 1) = _aux4;
	dq_dR_w(2, 2) = 0;
	dq_dR_w(2, 3) = _aux5;
	dq_dR_w(2, 4) = _aux7;
	dq_dR_w(2, 5) = 0;
	dq_dR_w(2, 6) = 0;
	dq_dR_w(2, 7) = 0;
	dq_dR_w(2, 8) = _aux7;
}
void  compute_dq_dR_x(Eigen::Matrix<double, 3, 9 >&  dq_dR_x, const double&  qx, const double&  r00, const double&  r10, const double&  r20, const double&  r01, const double&  r11, const double&  r21, const double&  r02, const double&  r12, const double&  r22) {
	(void)r00;
	(void)r11;
	(void)r21;
	(void)r12;
	(void)r22;
	double  _aux1 = 1 / qx;
	double  _aux2 = -0.125*_aux1;
	double  _aux3 = 1 / pow(qx, 3);
	double  _aux4 = r10 + r01;
	double  _aux5 = 0.25*_aux1;
	double  _aux6 = 0.03125*_aux3*_aux4;
	double  _aux7 = r20 + r02;
	double  _aux8 = 0.03125*_aux3*_aux7;
	dq_dR_x(0, 0) = 0.125*_aux1;
	dq_dR_x(0, 1) = 0;
	dq_dR_x(0, 2) = 0;
	dq_dR_x(0, 3) = 0;
	dq_dR_x(0, 4) = _aux2;
	dq_dR_x(0, 5) = 0;
	dq_dR_x(0, 6) = 0;
	dq_dR_x(0, 7) = 0;
	dq_dR_x(0, 8) = _aux2;
	dq_dR_x(1, 0) = -0.03125*_aux3*_aux4;
	dq_dR_x(1, 1) = _aux5;
	dq_dR_x(1, 2) = 0;
	dq_dR_x(1, 3) = _aux5;
	dq_dR_x(1, 4) = _aux6;
	dq_dR_x(1, 5) = 0;
	dq_dR_x(1, 6) = 0;
	dq_dR_x(1, 7) = 0;
	dq_dR_x(1, 8) = _aux6;
	dq_dR_x(2, 0) = -0.03125*_aux3*_aux7;
	dq_dR_x(2, 1) = 0;
	dq_dR_x(2, 2) = _aux5;
	dq_dR_x(2, 3) = 0;
	dq_dR_x(2, 4) = _aux8;
	dq_dR_x(2, 5) = 0;
	dq_dR_x(2, 6) = _aux5;
	dq_dR_x(2, 7) = 0;
	dq_dR_x(2, 8) = _aux8;
}
void  compute_dq_dR_y(Eigen::Matrix<double, 3, 9 >&  dq_dR_y, const double&  qy, const double&  r00, const double&  r10, const double&  r20, const double&  r01, const double&  r11, const double&  r21, const double&  r02, const double&  r12, const double&  r22) {
	(void)r00;
	(void)r20;
	(void)r11;
	(void)r02;
	(void)r22;
	double  _aux1 = 1 / pow(qy, 3);
	double  _aux2 = r10 + r01;
	double  _aux3 = 0.03125*_aux1*_aux2;
	double  _aux4 = 1 / qy;
	double  _aux5 = 0.25*_aux4;
	double  _aux6 = -0.125*_aux4;
	double  _aux7 = r21 + r12;
	double  _aux8 = 0.03125*_aux1*_aux7;
	dq_dR_y(0, 0) = _aux3;
	dq_dR_y(0, 1) = _aux5;
	dq_dR_y(0, 2) = 0;
	dq_dR_y(0, 3) = _aux5;
	dq_dR_y(0, 4) = -0.03125*_aux1*_aux2;
	dq_dR_y(0, 5) = 0;
	dq_dR_y(0, 6) = 0;
	dq_dR_y(0, 7) = 0;
	dq_dR_y(0, 8) = _aux3;
	dq_dR_y(1, 0) = _aux6;
	dq_dR_y(1, 1) = 0;
	dq_dR_y(1, 2) = 0;
	dq_dR_y(1, 3) = 0;
	dq_dR_y(1, 4) = 0.125*_aux4;
	dq_dR_y(1, 5) = 0;
	dq_dR_y(1, 6) = 0;
	dq_dR_y(1, 7) = 0;
	dq_dR_y(1, 8) = _aux6;
	dq_dR_y(2, 0) = _aux8;
	dq_dR_y(2, 1) = 0;
	dq_dR_y(2, 2) = 0;
	dq_dR_y(2, 3) = 0;
	dq_dR_y(2, 4) = -0.03125*_aux1*_aux7;
	dq_dR_y(2, 5) = _aux5;
	dq_dR_y(2, 6) = 0;
	dq_dR_y(2, 7) = _aux5;
	dq_dR_y(2, 8) = _aux8;
}
void  compute_dq_dR_z(Eigen::Matrix<double, 3, 9 >&  dq_dR_z, const double&  qz, const double&  r00, const double&  r10, const double&  r20, const double&  r01, const double&  r11, const double&  r21, const double&  r02, const double&  r12, const double&  r22) {
	(void)r00;
	(void)r10;
	(void)r01;
	(void)r11;
	(void)r22;
	double  _aux1 = 1 / pow(qz, 3);
	double  _aux2 = r20 + r02;
	double  _aux3 = 0.03125*_aux1*_aux2;
	double  _aux4 = 1 / qz;
	double  _aux5 = 0.25*_aux4;
	double  _aux6 = r21 + r12;
	double  _aux7 = 0.03125*_aux1*_aux6;
	double  _aux8 = -0.125*_aux4;
	dq_dR_z(0, 0) = _aux3;
	dq_dR_z(0, 1) = 0;
	dq_dR_z(0, 2) = _aux5;
	dq_dR_z(0, 3) = 0;
	dq_dR_z(0, 4) = _aux3;
	dq_dR_z(0, 5) = 0;
	dq_dR_z(0, 6) = _aux5;
	dq_dR_z(0, 7) = 0;
	dq_dR_z(0, 8) = -0.03125*_aux1*_aux2;
	dq_dR_z(1, 0) = _aux7;
	dq_dR_z(1, 1) = 0;
	dq_dR_z(1, 2) = 0;
	dq_dR_z(1, 3) = 0;
	dq_dR_z(1, 4) = _aux7;
	dq_dR_z(1, 5) = _aux5;
	dq_dR_z(1, 6) = 0;
	dq_dR_z(1, 7) = _aux5;
	dq_dR_z(1, 8) = -0.03125*_aux1*_aux6;
	dq_dR_z(2, 0) = _aux8;
	dq_dR_z(2, 1) = 0;
	dq_dR_z(2, 2) = 0;
	dq_dR_z(2, 3) = 0;
	dq_dR_z(2, 4) = _aux8;
	dq_dR_z(2, 5) = 0;
	dq_dR_z(2, 6) = 0;
	dq_dR_z(2, 7) = 0;
	dq_dR_z(2, 8) = 0.125*_aux4;
}
void  compute_dR_dq(Eigen::Matrix<double, 9, 3 >&  dR_dq, const double&  qx, const double&  qy, const double&  qz, const double&  qw) {
	double  _aux1 = -4 * qy;
	double  _aux2 = -4 * qz;
	double  _aux3 = 1 / qw;
	double  _aux4 = 2 * qx*qz;
	double  _aux5 = -_aux3*(_aux4 - 2 * qw*qy);
	double  _aux6 = 2 * qy*qz;
	double  _aux7 = -_aux3*(_aux6 - 2 * qw*qx);
	double  _aux8 = -2 * pow(qw, 2);
	double  _aux9 = _aux8 + 2 * pow(qz, 2);
	double  _aux10 = 2 * qw*qz;
	double  _aux11 = (_aux10 + 2 * qx*qy)*_aux3;
	double  _aux12 = _aux8 + 2 * pow(qy, 2);
	double  _aux13 = _aux3*(_aux6 + 2 * qw*qx);
	double  _aux14 = _aux3*(_aux4 + 2 * qw*qy);
	double  _aux15 = -4 * qx;
	double  _aux16 = _aux8 + 2 * pow(qx, 2);
	double  _aux17 = (_aux10 - 2 * qx*qy)*_aux3;
	dR_dq(0, 0) = 0;
	dR_dq(0, 1) = _aux1;
	dR_dq(0, 2) = _aux2;
	dR_dq(1, 0) = _aux5;
	dR_dq(1, 1) = _aux7;
	dR_dq(1, 2) = -_aux3*_aux9;
	dR_dq(2, 0) = _aux11;
	dR_dq(2, 1) = _aux12*_aux3;
	dR_dq(2, 2) = _aux13;
	dR_dq(3, 0) = _aux14;
	dR_dq(3, 1) = _aux13;
	dR_dq(3, 2) = _aux3*_aux9;
	dR_dq(4, 0) = _aux15;
	dR_dq(4, 1) = 0;
	dR_dq(4, 2) = _aux2;
	dR_dq(5, 0) = -_aux16*_aux3;
	dR_dq(5, 1) = _aux17;
	dR_dq(5, 2) = _aux5;
	dR_dq(6, 0) = _aux17;
	dR_dq(6, 1) = -_aux12*_aux3;
	dR_dq(6, 2) = _aux7;
	dR_dq(7, 0) = _aux16*_aux3;
	dR_dq(7, 1) = _aux11;
	dR_dq(7, 2) = _aux14;
	dR_dq(8, 0) = _aux15;
	dR_dq(8, 1) = _aux1;
	dR_dq(8, 2) = 0;
}


void  compute_dq_dR(Eigen::Matrix<double, 3, 9, Eigen::ColMajor>&  dq_dR, const double&  r11, const double&  r21, const double&  r31, const double&  r12, const double&  r22, const double&  r32, const double&  r13, const double&  r23, const double&  r33) {
	double qw;
	double S;
	int whichCase = _q2m(S, qw, r11, r21, r31, r12, r22, r32, r13, r23, r33);
	S *= .25;
	switch (whichCase) {
	case 0: compute_dq_dR_w(dq_dR, S, r11, r21, r31, r12, r22, r32, r13, r23, r33);
		break;
	case 1: compute_dq_dR_x(dq_dR, S, r11, r21, r31, r12, r22, r32, r13, r23, r33);
		break;
	case 2: compute_dq_dR_y(dq_dR, S, r11, r21, r31, r12, r22, r32, r13, r23, r33);
		break;
	case 3: compute_dq_dR_z(dq_dR, S, r11, r21, r31, r12, r22, r32, r13, r23, r33);
		break;
	}
	if (qw <= 0)
		dq_dR *= -1;
}

void computeEdgeSE3Gradient(Eigen::Isometry3d& E,
	Eigen::Matrix6d &Ji,
	Eigen::Matrix6d &Jj,
	const Eigen::Isometry3d& Z,
	const Eigen::Isometry3d& Xi,
	const Eigen::Isometry3d& Xj)
{
	// compute the error at the linearization point
	const Eigen::Isometry3d A = Z.inverse();
	const Eigen::Isometry3d B = Xi.inverse()*Xj;

	E = A*B;

	Eigen::Isometry3d::ConstLinearPart Re = extractRotation(E);
	Eigen::Isometry3d::ConstLinearPart Ra = extractRotation(A);
	Eigen::Isometry3d::ConstLinearPart Rb = extractRotation(B);
	Eigen::Isometry3d::ConstTranslationPart tb = B.translation();

	Eigen::Matrix<double, 3, 9, Eigen::ColMajor>  dq_dR;
	compute_dq_dR(dq_dR,
		Re(0, 0), Re(1, 0), Re(2, 0),
		Re(0, 1), Re(1, 1), Re(2, 1),
		Re(0, 2), Re(1, 2), Re(2, 2));

	Ji.setZero();
	Jj.setZero();

	// dte/dti
	Ji.block<3, 3>(0, 0) = -Ra;

	// dte/dtj
	Jj.block<3, 3>(0, 0) = Re;

	// dte/dqi
	{
		Eigen::Matrix3d S;
		skewT(S, tb);
		Ji.block<3, 3>(0, 3) = Ra*S;
	}

	// dte/dqj: this is zero

	double buf[27];
	Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::ColMajor> > M(buf);
	Eigen::Matrix3d Sxt, Syt, Szt;
	// dre/dqi
	{
		skewT(Sxt, Syt, Szt, Rb);
		Eigen::Map<Eigen::Matrix3d> Mx(buf);    Mx.noalias() = Ra*Sxt;
		Eigen::Map<Eigen::Matrix3d> My(buf + 9);  My.noalias() = Ra*Syt;
		Eigen::Map<Eigen::Matrix3d> Mz(buf + 18); Mz.noalias() = Ra*Szt;
		Ji.block<3, 3>(3, 3) = dq_dR * M;
	}

	// dre/dqj
	{
		Eigen::Matrix3d& Sx = Sxt;
		Eigen::Matrix3d& Sy = Syt;
		Eigen::Matrix3d& Sz = Szt;
		skew(Sx, Sy, Sz, Eigen::Matrix3d::Identity());
		Eigen::Map<Eigen::Matrix3d> Mx(buf);    Mx.noalias() = Re*Sx;
		Eigen::Map<Eigen::Matrix3d> My(buf + 9);  My.noalias() = Re*Sy;
		Eigen::Map<Eigen::Matrix3d> Mz(buf + 18); Mz.noalias() = Re*Sz;
		Jj.block<3, 3>(3, 3) = dq_dR * M;
	}
}


std::tuple<Eigen::Matrix6d, Eigen::Matrix6d> linearizeOplus(
		Eigen::Isometry3d Xi, Eigen::Isometry3d Xj, Eigen::Isometry3d Z) {
	// BaseBinaryEdge<6, Isometry3D, VertexSE3, VertexSE3>::linearizeOplus();
	// return;
	Eigen::Isometry3d E;
	Eigen::Matrix6d _jacobianOplusXi;
	Eigen::Matrix6d _jacobianOplusXj;
	computeEdgeSE3Gradient(E, _jacobianOplusXi, _jacobianOplusXj, Z, Xi, Xj);
	return std::make_tuple(std::move(_jacobianOplusXi), std::move(_jacobianOplusXj));
}

std::tuple<Eigen::VectorXd, double> ComputeEG20(const PoseGraph &pose_graph)
{
	int n_edges = (int)pose_graph.edges_.size();
	double residual = 0.0;
	Eigen::VectorXd output(n_edges * 6);
	for (int iter_edge = 0; iter_edge < n_edges; iter_edge++) {
		Eigen::Matrix4d X, Ts, Tt;
		std::tie(X, Ts, Tt) = GetRelativePosesG2O(pose_graph, iter_edge);
		// VertexSE3 *from = static_cast<VertexSE3*>(_vertices[0]);
		// VertexSE3 *to = static_cast<VertexSE3*>(_vertices[1]);
		// Isometry3D delta=_inverseMeasurement * from->estimate().inverse() * to->estimate();
		Eigen::Isometry3d delta = Eigen::Isometry3d(X.inverse() * Ts.inverse() * Tt); 
		Eigen::Vector6d e = toVectorMQT(delta);
		output.block<6, 1>(iter_edge * 6, 0) = e;

		const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		residual += e.transpose() * te.information_ * e;
		
		//const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
		//if (te.source_node_id_ == 195 && te.target_node_id_ == 209) {
		//	std::cout << "[ComputeEG20]" << 
		//		te.source_node_id_ << "-"  << te.target_node_id_ << std::endl;
		//	std::cout << "X.inverse().matrix()" << std::endl;
		//	std::cout << X.inverse().matrix() << std::endl;
		//	std::cout << "Ts.inverse().matrix()" << std::endl;
		//	std::cout << Ts.inverse().matrix() << std::endl;
		//	std::cout << "Tt.matrix()" << std::endl;
		//	std::cout << Tt.matrix() << std::endl;
		//	std::cout << "delta.matrix()" << std::endl;
		//	std::cout << delta.matrix() << std::endl;
		//	std::cout << "e" << std::endl;
		//	std::cout << e << std::endl;
		//}	
		
	}
	return std::make_tuple(std::move(output), residual);
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeHG2O(
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

		Eigen::Matrix4d X, Ts, Tt;
		std::tie(X, Ts, Tt) = GetRelativePosesG2O(pose_graph, iter_edge);

		Eigen::Isometry3d E;
		Eigen::Matrix6d J_source, J_target;
		std::tie(J_source, J_target) = linearizeOplus(
			Eigen::Isometry3d(Ts), Eigen::Isometry3d(Tt), Eigen::Isometry3d(X));

		//if (iter_edge == 2000) {
		//	std::cout << "[ComputeHG2O]" << std::endl;
		//	std::cout << "J_source" << std::endl;
		//	std::cout << J_source << std::endl;
		//	std::cout << "J_target" << std::endl;
		//	std::cout << J_target << std::endl;
		//}
		
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

void approximateNearestOrthogonalMatrix(const Eigen::Matrix3d& R)
{
	Eigen::Matrix3d E = R.transpose() * R;
	E.diagonal().array() -= 1;
	const_cast<Eigen::Matrix3d&>(R) -= 0.5 * R * E;
}

/**
* update the position of this vertex. The update is in the form
* (x,y,z,qx,qy,qz) whereas (x,y,z) represents the translational update
* and (qx,qy,qz) corresponds to the respective elements. The missing
* element qw of the quaternion is recovred by
* || (qw,qx,qy,qz) || == 1 => qw = sqrt(1 - || (qx,qy,qz) ||
*/
/// this function looks okay
void oplusImpl(const Eigen::Vector6d &update, Eigen::Isometry3d &_estimate)
{
	Eigen::Isometry3d increment = fromVectorMQT(update);

	//std::cout << "[oplusImpl]" << std::endl;
	//std::cout << "update" << std::endl;
	//std::cout << update << std::endl;
	//std::cout << "_estimate.matrix()" << std::endl;
	//std::cout << _estimate.matrix() << std::endl;
	//std::cout << "increment.matrix()" << std::endl;
	//std::cout << increment.matrix() << std::endl;

	_estimate = _estimate * increment;
	
	//orthogonalizeAfter = 1000
	//if (++_numOplusCalls > orthogonalizeAfter) {
	//	_numOplusCalls = 0;
	//	internal::approximateNearestOrthogonalMatrix(_estimate.matrix().topLeftCorner<3, 3>());
	//}
	//approximateNearestOrthogonalMatrix(_estimate.matrix().topLeftCorner<3, 3>());
}

/// inconsistency: the loop iteration
/// how to use x?
std::shared_ptr<PoseGraph> GlobalOptimizationG2O(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("[GlobalOptimizationG2O] PoseGraph having %d nodes and %d edges\n",
		n_nodes, n_edges);

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();

	//Eigen::VectorXd x(n_nodes * 6);
	//for (int node_iter = 0; node_iter < n_nodes; node_iter++) {
	//	Eigen::Matrix4d eigen_mat = pose_graph.nodes_[node_iter].pose_;
	//	x.block<6,1>(node_iter*6,0) = toVectorMQT(Eigen::Isometry3d(eigen_mat));
	//}

	Eigen::VectorXd evec;
	double total_residual;
	std::tie(evec, total_residual) = ComputeEG20(*pose_graph_refined);

	//double MAX_ITER_DEBUG = 2;
	for (int iter = 0; iter < MAX_ITER; iter++) {

		int line_process_cnt = 0;

		Eigen::MatrixXd H;
		Eigen::VectorXd b;
		std::tie(H, b) = ComputeHG2O(
			*pose_graph_refined, evec, line_process);
		PrintDebug("Iter : %d, residual : %e\n", iter, total_residual);

		// why determinant of H is inf?
		//H += 10.0 * Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6); // simple LM
		Eigen::VectorXd delta = H.colPivHouseholderQr().solve(b);
		Eigen::VectorXd err = H*(delta) - b;
		std::cout << "err norm : " << err.norm() << std::endl;

		//if (iter == 0) {
		//	//std::cout << "Saving matrix" << std::endl;
		//	//std::ofstream file("H.txt");
		//	//file << H;
		//	//file.close();
		//	//std::ofstream file2("b.txt");
		//	//file2 << b;
		//	//file2.close();
		//	//std::cout << "Loading matrix" << std::endl;
		//	//std::ifstream file3("x.txt");
		//	//for (int iter_node = 0; iter_node < n_nodes * 6; iter_node++) {
		//	//	file3 >> delta(iter_node);
		//	//}
		//	//file3.close();
		//}

		// update pose of nodes
		for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
			Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
			Eigen::Isometry3d temp = Eigen::Isometry3d(pose_graph_refined->nodes_[iter_node].pose_);
			oplusImpl(delta_iter, temp);
			pose_graph_refined->nodes_[iter_node].pose_ = temp.matrix();
		}
		std::tie(evec, total_residual) = ComputeEG20(*pose_graph_refined);
		//line_process = ComputeLineprocess(*pose_graph_refined, evec);

		// update line process only for loop edges

		if (stopping_criterion()) // todo: adding stopping criterion
			break;
	}

	return pose_graph_refined;
}

/// inconsistency: the loop iteration
/// how to use x?
std::shared_ptr<PoseGraph> GlobalOptimizationG2OLM(const PoseGraph &pose_graph)
{
	int n_nodes = (int)pose_graph.nodes_.size();
	int n_edges = (int)pose_graph.edges_.size();

	PrintDebug("[GlobalOptimizationG2OLM] PoseGraph having %d nodes and %d edges\n",
		n_nodes, n_edges);

	std::shared_ptr<PoseGraph> pose_graph_refined = std::make_shared<PoseGraph>();
	*pose_graph_refined = pose_graph;

	Eigen::VectorXd line_process(n_edges - (n_nodes - 1));
	line_process.setOnes();
	
	Eigen::VectorXd evec;
	double current_residual;	
	std::tie(evec, current_residual) = ComputeEG20(*pose_graph_refined);
	double new_residual = current_residual;

	Eigen::MatrixXd H;
	Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
	Eigen::VectorXd b;
	std::tie(H, b) = ComputeHG2O(
		*pose_graph_refined, evec, line_process);

	Eigen::VectorXd H_diag = H.diagonal();
	double tau = 1e-5; 
	double currentLambda = tau * H_diag.maxCoeff();	
	double ni = 2.0;
	double rho = 0;

	bool stop = false;
	if (b.maxCoeff() < EPS_1) // b is near zero. Bad condition.
		stop = true;

	PrintDebug("[Initial     ] residual : %e, lambda : %e\n",
		current_residual, currentLambda);

	//double MAX_ITER_DEBUG = 2;
	for (int iter = 0; iter < MAX_ITER && !stop; iter++) {
		int lm_count = 0;
		do {
			// why determinant of H is inf?
			Eigen::MatrixXd H_LM = H + currentLambda * H_I;
			Eigen::VectorXd delta = H_LM.colPivHouseholderQr().solve(b);
			
			double solver_error = (H_LM*(delta) - b).norm();
			if (solver_error > EPS_5) {
				PrintWarning("[Job finished] error norm %e is higher than %e\n", 
						solver_error, EPS_5);
				stop = true;
			}

			// update pose of nodes
			std::shared_ptr<PoseGraph> pose_graph_refined_new =
				std::make_shared<PoseGraph>();
			*pose_graph_refined_new = *pose_graph_refined;
			for (int iter_node = 0; iter_node < n_nodes; iter_node++) {
				Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
				Eigen::Isometry3d temp = Eigen::Isometry3d(pose_graph_refined_new->nodes_[iter_node].pose_);
				oplusImpl(delta_iter, temp);
				pose_graph_refined_new->nodes_[iter_node].pose_ = temp.matrix();
			}
			Eigen::VectorXd evec_new;
			std::tie(evec_new, new_residual) = ComputeEG20(*pose_graph_refined_new);
			//PrintDebug("Chi2 error %e -> %e\n", current_residual, new_residual);
			//line_process = ComputeLineprocess(*pose_graph_refined, evec);

			// update line process only for loop edges

			//double temp_rho = (delta.dot(currentLambda * delta + b) + 1e-3);
			//PrintDebug("temp rho : %e\n", temp_rho);

			rho = (current_residual - new_residual) / (delta.dot(currentLambda * delta + b) + 1e-3);
			//PrintDebug("rho : %e\n", rho);
			if (rho > 0) { // last step was good
				if (current_residual - new_residual < EPS_4 * current_residual) {
					stop = true;
					PrintDebug("[Job finished] current_residual - new_residual < %e * current_residual\n", EPS_4);
				}
				double alpha = 1. - pow((2 * rho - 1), 3);
				// crop lambda between minimum and maximum factors
				alpha = (std::min)(alpha, _goodStepUpperScale);
				double scaleFactor = (std::max)(_goodStepLowerScale, alpha);
				currentLambda *= scaleFactor;
				ni = 2;
				current_residual = new_residual;
				//_optimizer->discardTop();

				evec = evec_new;
				*pose_graph_refined = *pose_graph_refined_new;
				std::tie(H, b) = ComputeHG2O(
					*pose_graph_refined, evec, line_process);

				if (b.maxCoeff() < EPS_1) {
					stop = true;
					PrintDebug("[Job finished] b.maxCoeff() < %e\n", EPS_1);
				}
			}
			else {
				currentLambda *= ni;
				ni *= 2;
			}

			//if (iter == 0) {
			//	//std::cout << "Saving matrix" << std::endl;
			//	//std::ofstream file("H.txt");
			//	//file << H;
			//	//file.close();
			//	//std::ofstream file2("b.txt");
			//	//file2 << b;
			//	//file2.close();
			//	std::cout << "Loading matrix" << std::endl;
			//	std::ifstream file("x.txt");
			//	for (int iter_node = 0; iter_node < n_nodes * 6; iter_node++) {
			//		file >> delta(iter_node);
			//	}
			//	file.close();
			//}

			//PrintDebug("[LM Loop %02d] changing lambda : %e\n",
			//	lm_count, currentLambda);
			lm_count++;		

		} while (!(rho > 0 || stop));
		if (!stop) {
			PrintDebug("[Iteration %02d] residual : %e, lambda : %e, LM iteration : %d\n",
					iter, current_residual, currentLambda, lm_count);
		}			
		if (current_residual < EPS_3)
			stop = true;
	}

	return pose_graph_refined;
}

}	// namespace three
