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

using namespace three;

void VisualizeRegistration(const three::PointCloud &source,
		const three::PointCloud &target, const Eigen::Matrix4d &Transformation)
{
	std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
	std::shared_ptr<PointCloud> target_ptr(new PointCloud);
	*source_transformed_ptr = source;
	*target_ptr = target;
	source_transformed_ptr->PaintUniformColor(Eigen::Vector3d(1.0, 0.706, 0.0));
	target_ptr->PaintUniformColor(Eigen::Vector3d(0.0, 0.651, 0.929));
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

	bool visualization = true; // false;

#ifdef _OPENMP
	PrintDebug("OpenMP is supported. Using %d threads.", omp_get_num_threads());
#endif

	std::cout << "1. Load two point clouds and disturb initial pose." << std::endl;
        std::shared_ptr<PointCloud> source, target;
	source = three::CreatePointCloudFromFile(argv[1]);
        target = three::CreatePointCloudFromFile(argv[2]);
	Eigen::Matrix4d trans_init;
        trans_init << 0.0, 0.0, 1.0, 0.0,
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0; 
	source->Transform(trans_init);
        if(visualization){
                VisualizeRegistration(*source, *target,
				Eigen::Matrix4d::Identity());
        }

	std::cout << "2. Downsample with a voxel size 0.05." << std::endl;
	std::shared_ptr<PointCloud> source_down, target_down;
        double voxel_size = 0.05;
	source_down = VoxelDownSample(*source, voxel_size);
        target_down = VoxelDownSample(*target, voxel_size);

	std::cout << "3. Estimate normal with search radius 0.1." << std::endl;
	double kdt_radius = 0.1;
	int kdt_max_nn = 30;
	EstimateNormals(*source_down, KDTreeSearchParamHybrid(kdt_radius, kdt_max_nn));
        EstimateNormals(*target_down, KDTreeSearchParamHybrid(kdt_radius, kdt_max_nn));

	std::cout << "4. Compute FPFH feature with search radius 0.25" << std::endl;
	std::shared_ptr<Feature> source_fpfh, target_fpfh;
	double fpfh_radius = 0.25;
	int fpfh_max_nn = 100;
	source_fpfh = ComputeFPFHFeature(
                        *source_down, three::KDTreeSearchParamHybrid(fpfh_radius, fpfh_max_nn));
        target_fpfh = ComputeFPFHFeature(
                        *target_down, three::KDTreeSearchParamHybrid(fpfh_radius, fpfh_max_nn));

	std::cout << "5. RANSAC registration on downsampled point clouds." << std::endl;
	std::cout << "   Since the downsampling voxel size is 0.05, we use a liberal" << std::endl;
	std::cout << "   distance threshold 0.075." << std::endl;

        std::vector<std::reference_wrapper<const CorrespondenceChecker>>
			correspondence_checker;
        // std::reference_wrapper<const CorrespondenceChecker>
        auto correspondence_checker_edge_length =
			CorrespondenceCheckerBasedOnEdgeLength(0.9);
        // std::reference_wrapper<const CorrespondenceChecker>
        auto correspondence_checker_distance =
			CorrespondenceCheckerBasedOnDistance(0.075);

        correspondence_checker.push_back(correspondence_checker_edge_length);
	correspondence_checker.push_back(correspondence_checker_distance);

        // RegistrationResult
	auto result_ransac = RegistrationRANSACBasedOnFeatureMatching(
			*source_down, *target_down, *source_fpfh, *target_fpfh, 0.075,
			TransformationEstimationPointToPoint(false), 4,
			correspondence_checker, RANSACConvergenceCriteria(4000000, 500));
	std::cout << result_ransac.transformation_ << std::endl;
        
	if(visualization){
		VisualizeRegistration(*source, *target,
				result_ransac.transformation_);
        }

	std::cout << "6. Point-to-plane ICP registration is applied on original point" << std::endl;
	std::cout << "   clouds to refine the alignment. This time we use a strict" << std::endl;
	std::cout << "   distance threshold 0.02." << std::endl;

        // RegistrationResult
	auto result_icp = RegistrationICP(*source, *target, 0.02,
			result_ransac.transformation_,
			TransformationEstimationPointToPlane());
	std::cout << result_icp.transformation_ << std::endl;

	if(visualization){
		VisualizeRegistration(*source, *target,
				result_icp.transformation_);
        }

	return 0;
}
