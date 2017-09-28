// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

using namespace three;

std::tuple<std::shared_ptr<PointCloud>, std::shared_ptr<Feature>>
		PreprocessPointCloud(const char* file_name)
{
	auto pcd = three::CreatePointCloudFromFile(file_name);
	auto pcd_down = VoxelDownSample(*pcd, 0.04);
	EstimateNormals(*pcd_down, three::KDTreeSearchParamHybrid(0.01, 30));
	auto pcd_fpfh = ComputeFPFHFeature(
			*pcd_down, three::KDTreeSearchParamHybrid(0.25, 100));
	return std::make_tuple(pcd_down, pcd_fpfh);
}

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
	using namespace three;

	SetVerbosityLevel(VERBOSE_ALWAYS);

	for (int i = 0; i < 3; i++) {
		std::string source_path("D:/Dropbox/Intel/fragment/011/fragments/fragment_001.ply");
		std::string target_path("D:/Dropbox/Intel/fragment/011/fragments/fragment_001.ply");

		std::shared_ptr<PointCloud> source, target;
		std::shared_ptr<Feature> source_fpfh, target_fpfh;
		std::tie(source, source_fpfh) = 
				PreprocessPointCloud(source_path.c_str());
		std::tie(target, target_fpfh) = 
				PreprocessPointCloud(target_path.c_str());

		std::vector<std::reference_wrapper<const CorrespondenceChecker>>
			correspondence_checker;
		auto correspondence_checker_edge_length =
			CorrespondenceCheckerBasedOnEdgeLength(0.9);
		auto correspondence_checker_distance =
			CorrespondenceCheckerBasedOnDistance(0.075);

		correspondence_checker.push_back(correspondence_checker_edge_length);
		correspondence_checker.push_back(correspondence_checker_distance);
		auto registration_result = RegistrationRANSACBasedOnFeatureMatching(
			*source, *target, *source_fpfh, *target_fpfh, 0.075,
			TransformationEstimationPointToPoint(false), 4,
			correspondence_checker, RANSACConvergenceCriteria(400000, 500));
	}
	
	//VisualizeRegistration(*source, *target, registration_result.transformation_);
}
