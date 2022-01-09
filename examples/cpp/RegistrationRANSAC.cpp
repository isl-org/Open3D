// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/Open3D.h"

using namespace open3d;

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
PreprocessPointCloud(const char *file_name, const float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down,
            open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd, pcd_down, pcd_fpfh);
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

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RegistrationRANSAC source_pcd target_pcd"
                     "[--method=feature_matching] "
                     "[--voxel_size=0.05] [--distance_multiplier=1.5]"
                     "[--max_iterations 1000000] [--confidence 0.999]"
                     "[--mutual_filter]");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string method = "";
    const std::string kMethodFeature = "feature_matching";
    const std::string kMethodCorres = "correspondence";
    if (utility::ProgramOptionExists(argc, argv, "--method")) {
        method = utility::GetProgramOptionAsString(argc, argv, "--method");
    } else {
        method = "feature_matching";
    }
    if (method != kMethodFeature && method != kMethodCorres) {
        utility::LogInfo(
                "--method must be \'feature_matching\' or "
                "\'correspondence\'.");
        return 1;
    }

    bool mutual_filter = false;
    if (utility::ProgramOptionExists(argc, argv, "--mutual_filter")) {
        mutual_filter = true;
    }
    float voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    float distance_multiplier = utility::GetProgramOptionAsDouble(
            argc, argv, "--distance_multiplier", 1.5);
    float distance_threshold = voxel_size * distance_multiplier;
    int max_iterations = utility::GetProgramOptionAsInt(
            argc, argv, "--max_iterations", 1000000);
    float confidence = utility::GetProgramOptionAsDouble(argc, argv,
                                                         "--confidence", 0.999);

    // Prepare input
    std::shared_ptr<geometry::PointCloud> source, source_down, target,
            target_down;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_down, source_fpfh) =
            PreprocessPointCloud(argv[1], voxel_size);
    std::tie(target, target_down, target_fpfh) =
            PreprocessPointCloud(argv[2], voxel_size);

    pipelines::registration::RegistrationResult registration_result;

    // Prepare checkers
    std::vector<std::reference_wrapper<
            const pipelines::registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
                    0.9);
    auto correspondence_checker_distance =
            pipelines::registration::CorrespondenceCheckerBasedOnDistance(
                    distance_threshold);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);

    if (method == kMethodFeature) {
        registration_result = pipelines::registration::
                RegistrationRANSACBasedOnFeatureMatching(
                        *source_down, *target_down, *source_fpfh, *target_fpfh,
                        mutual_filter, distance_threshold,
                        pipelines::registration::
                                TransformationEstimationPointToPoint(false),
                        3, correspondence_checker,
                        pipelines::registration::RANSACConvergenceCriteria(
                                max_iterations, confidence));
    } else if (method == kMethodCorres) {
        // Manually search correspondences
        int nPti = int(source_down->points_.size());
        int nPtj = int(target_down->points_.size());

        geometry::KDTreeFlann feature_tree_i(*source_fpfh);
        geometry::KDTreeFlann feature_tree_j(*target_fpfh);

        pipelines::registration::CorrespondenceSet corres_ji;
        std::vector<int> i_to_j(nPti, -1);

        // Buffer all correspondences
        for (int j = 0; j < nPtj; j++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)),
                                     1, corres_tmp, dist_tmp);
            int i = corres_tmp[0];
            corres_ji.push_back(Eigen::Vector2i(i, j));
        }

        if (mutual_filter) {
            pipelines::registration::CorrespondenceSet mutual_corres;
            for (auto &corres : corres_ji) {
                int j = corres(1);
                int j2i = corres(0);

                std::vector<int> corres_tmp(1);
                std::vector<double> dist_tmp(1);
                feature_tree_j.SearchKNN(
                        Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1,
                        corres_tmp, dist_tmp);
                int i2j = corres_tmp[0];
                if (i2j == j) {
                    mutual_corres.push_back(corres);
                }
            }

            utility::LogDebug("{:d} points remain after mutual filter",
                              mutual_corres.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source_down, *target_down, mutual_corres,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    max_iterations, confidence));
        } else {
            utility::LogDebug("{:d} points remain", corres_ji.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source_down, *target_down, corres_ji,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    max_iterations, confidence));
        }
    }

    VisualizeRegistration(*source, *target,
                          registration_result.transformation_);

    return 0;
}
