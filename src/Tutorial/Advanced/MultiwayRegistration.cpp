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

#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Core/Utility/Timer.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

using namespace three;

int main(int argc, char *argv[])
{
	SetVerbosityLevel(VerbosityLevel::VerboseAlways);

	bool visualization = true;

#ifdef _OPENMP
	PrintDebug("OpenMP is supported. Using %d threads.", omp_get_num_threads());
#endif

	std::vector<std::shared_ptr<PointCloud> > pcds;
	for(int i=0; i<3; i++){
		std::string filename = std::string("../../../lib/TestData/ICP/cloud_bin_")
				+ std::to_string(i) + std::string(".pcd");
		std::shared_ptr<PointCloud> pcd = CreatePointCloudFromFile(filename);
		double voxel_size = 0.02;
		std::shared_ptr<PointCloud> downpcd = VoxelDownSample(*pcd, voxel_size);
		pcds.push_back(downpcd);
	}

	if(visualization){
	        std::vector<std::shared_ptr<const Geometry> > geoms; // cast?
		geoms.assign(pcds.begin(), pcds.end());
		DrawGeometries({geoms});
	}

	PoseGraph pose_graph;
	Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
	pose_graph.nodes_.push_back(PoseGraphNode(odometry));

	int n_pcds = pcds.size();
	Eigen::Matrix4d transformation_icp = Eigen::Matrix4d::Identity();
	Eigen::Matrix6d information_icp = Eigen::Matrix6d::Identity();
	for(int source_id = 0; source_id < n_pcds; source_id++){
		for(int target_id = source_id + 1; target_id < n_pcds; target_id++){
			std::shared_ptr<PointCloud> source = pcds[source_id];
			std::shared_ptr<PointCloud> target = pcds[target_id];

			std::cout << "Apply point-to-plane ICP" << std::endl;
                        // RegistrationResult
			auto icp_coarse = RegistrationICP(*source, *target, 0.3,
					Eigen::Matrix4d::Identity(),
					TransformationEstimationPointToPlane());
                        // RegistrationResult
			auto icp_fine = RegistrationICP(*source, *target, 0.03,
					icp_coarse.transformation_,
					TransformationEstimationPointToPlane());
			transformation_icp = icp_fine.transformation_;
			information_icp = GetInformationMatrixFromPointClouds(
					*source, *target, 0.03f, icp_fine.transformation_);
			std::cout << information_icp << std::endl;

			// VisualizeRegistration(*source, *target, Eigen::Matrix4d::Identity());
			std::cout << "Build PoseGraph" << std::endl;
                        bool uncertain;
			if(target_id == source_id + 1){ // odometry case
				odometry *= transformation_icp;
				pose_graph.nodes_.push_back(
						PoseGraphNode(odometry.inverse()));
				uncertain = false;
				pose_graph.edges_.push_back(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, uncertain));
			}
			else { // loop closure case
				uncertain = true;
				pose_graph.edges_.push_back(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, uncertain));
			}
		}
	}

	std::cout << "Optimizing PoseGraph ..." << std::endl;
	GlobalOptimizationConvergenceCriteria criteria;
	double max_correspondence_distance = 0.03;
	double edge_prune_threshold = 0.25;
	int reference_node = 0;
	GlobalOptimizationOption option(max_correspondence_distance,
			edge_prune_threshold,
			reference_node);
	GlobalOptimizationLevenbergMarquardt optimization_method;
	GlobalOptimization(pose_graph, optimization_method,
			criteria, option); 

	std::cout << "Transform points and display" << std::endl;
	for(int point_id=0; point_id<n_pcds; point_id++){
		std::cout << pose_graph.nodes_[point_id].pose_ << std::endl;
                pcds[point_id]->Transform(pose_graph.nodes_[point_id].pose_);
	}

	if(visualization){
                std::vector<std::shared_ptr<const Geometry> > geoms; // cast?
	        geoms.assign(pcds.begin(), pcds.end());
                DrawGeometries({geoms});
	}

	return 0;
}
