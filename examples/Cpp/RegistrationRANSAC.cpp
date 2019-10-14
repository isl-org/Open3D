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

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "Open3D/Open3D.h"

using namespace open3d;

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<registration::Feature>>
PreprocessPointCloud(const char *file_name) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(0.05);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(0.25, 100));
    return std::make_tuple(pcd_down, pcd_fpfh);
}

void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3 && argc != 4) {
        utility::LogInfo(
                "Usage : RegistrationRANSAC [path_to_first_point_cloud] "
                "[path_to_second_point_cloud] --visualize");
        return 1;
    }

    bool visualize = false;
    if (utility::ProgramOptionExists(argc, argv, "--visualize"))
        visualize = true;

    std::shared_ptr<geometry::PointCloud> source, target;
    std::shared_ptr<registration::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_fpfh) = PreprocessPointCloud(argv[1]);
    std::tie(target, target_fpfh) = PreprocessPointCloud(argv[2]);

    std::vector<
            std::reference_wrapper<const registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
    auto correspondence_checker_distance =
            registration::CorrespondenceCheckerBasedOnDistance(0.075);
    auto correspondence_checker_normal =
            registration::CorrespondenceCheckerBasedOnNormal(0.52359878);

    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    correspondence_checker.push_back(correspondence_checker_normal);
    auto registration_result =
            registration::RegistrationRANSACBasedOnFeatureMatching(
                    *source, *target, *source_fpfh, *target_fpfh, 0.075,
                    registration::TransformationEstimationPointToPoint(false),
                    4, correspondence_checker,
                    registration::RANSACConvergenceCriteria(4000000, 1000));

    if (visualize)
        VisualizeRegistration(*source, *target,
                              registration_result.transformation_);

    return 0;
}
