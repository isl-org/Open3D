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

#include <Core/Registration/ColoredICP.h>

using namespace three;

void VisualizeRegistration(const three::PointCloud &source,
		const three::PointCloud &target, const Eigen::Matrix4d &Transformation)
{
	std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
	std::shared_ptr<PointCloud> target_ptr(new PointCloud);
	*source_transformed_ptr = source;
	*target_ptr = target;
	source_transformed_ptr->Transform(Transformation);
	DrawGeometries({ source_transformed_ptr, target_ptr }, "Registration result");
}

int main(int argc, char *argv[])
{
	SetVerbosityLevel(VerbosityLevel::VerboseAlways);

	if (argc != 3) {
		PrintDebug("Usage : %s [path_to_first_point_cloud] [path_to_second_point_cloud]\n",
				argv[0]);
		return 1;
	}

	bool visualization = true;

#ifdef _OPENMP
	PrintDebug("OpenMP is supported. Using %d threads.", omp_get_num_threads());
#endif

	std::cout << "1. Load two point clouds and show initial pose" << std::endl;
        std::shared_ptr<PointCloud> source, target;
	source = CreatePointCloudFromFile(argv[1]);
        target = CreatePointCloudFromFile(argv[2]);

	// draw initial alignment
	Eigen::Matrix4d current_transformation = Eigen::Matrix4d::Identity();
        if(visualization){
		VisualizeRegistration(*source, *target,
	                               current_transformation);
        }

	// point to plane ICP
	current_transformation = Eigen::Matrix4d::Identity();
	std::cout << "2. point-to-plane ICP registration is applied on original point" << std::endl;
	std::cout << "   clouds to refine the alignment. Distance threshold 0.02." << std::endl;
        RegistrationResult result_icp;
	result_icp = RegistrationICP(*source, *target, 0.02,
			current_transformation, TransformationEstimationPointToPlane());
	std::cout << result_icp.transformation_ << std::endl;
	if(visualization){
                VisualizeRegistration(*source, *target,
                                result_icp.transformation_);
        }

	// Colored pointcloud registration
	// This is implementation of following paper
	// J. Park, Q.-Y. Zhou, V. Koltun,
	// Colored Point Cloud Registration Revisited, ICCV 2017
	double voxel_radius[] = {0.04, 0.02, 0.01};
	int max_iter[] = {50, 30, 14};
	current_transformation = Eigen::Matrix4d::Identity();
	std::cout << "3. Colored point cloud registration" << std::endl;
	for(int scale=0; scale<3; scale++){
		ScopeTimer t("one iteration");

		std::shared_ptr<PointCloud> source_down, target_down;

		int iter = max_iter[scale];
		double radius = voxel_radius[scale];
		std::cout << iter << " " << radius << " " << scale << std::endl;

		std::cout << "3-1. Downsample with a voxel size " << radius << std::endl;
		source_down = VoxelDownSample(*source, radius);
		target_down = VoxelDownSample(*target, radius);

		std::cout << "3-2. Estimate normal." << std::endl;
		EstimateNormals(*source_down, KDTreeSearchParamHybrid(radius * 2.0, 30));
		EstimateNormals(*target_down, KDTreeSearchParamHybrid(radius * 2.0, 30));

		std::cout << "3-3. Applying colored point cloud registration" << std::endl;
		result_icp = RegistrationColoredICP(*source_down, *target_down, radius, current_transformation,
                         ICPConvergenceCriteria(1e-16, 1e-6, iter));
		current_transformation = result_icp.transformation_;
		std::cout << result_icp.transformation_ << std::endl;

		if(visualization){
			VisualizeRegistration(*source, *target,
				result_icp.transformation_);
		}
	}

	return 0;
}
