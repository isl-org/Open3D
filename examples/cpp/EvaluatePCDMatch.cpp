// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > EvaluatePCDMatch [options]");
    utility::LogInfo("      View pairwise matching result of point clouds.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --log file                : A log file of the pairwise matching results. Must have.");
    utility::LogInfo("    --gt file                 : A log file of the ground truth pairwise matching results. Must have.");
    utility::LogInfo("    --threshold t             : Distance threshold. Must have.");
    utility::LogInfo("    --threshold_rmse t        : Distance threshold to decide if a match is good or not. Default: 2t.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    // clang-format on
}

bool ReadLogFile(const std::string &filename,
                 std::vector<std::pair<int, int>> &pair_ids,
                 std::vector<Eigen::Matrix4d> &transformations) {
    using namespace open3d;
    pair_ids.clear();
    transformations.clear();
    FILE *f = utility::filesystem::FOpen(filename, "r");
    if (f == NULL) {
        utility::LogWarning("Read LOG failed: unable to open file.");
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    int i, j, k;
    Eigen::Matrix4d trans;
    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
        if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
            if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                return false;
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0, 0),
                       &trans(0, 1), &trans(0, 2), &trans(0, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1, 0),
                       &trans(1, 1), &trans(1, 2), &trans(1, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2, 0),
                       &trans(2, 1), &trans(2, 2), &trans(2, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3, 0),
                       &trans(3, 1), &trans(3, 2), &trans(3, 3));
            }
            pair_ids.push_back(std::make_pair(i, j));
            transformations.push_back(trans);
        }
    }
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    std::string log_filename =
            utility::GetProgramOptionAsString(argc, argv, "--log");
    std::string gt_filename =
            utility::GetProgramOptionAsString(argc, argv, "--gt");

    double threshold =
            utility::GetProgramOptionAsDouble(argc, argv, "--threshold");
    double threshold_rmse = utility::GetProgramOptionAsDouble(
            argc, argv, "--threshold_rmse", threshold * 2.0);

    double threshold2 = threshold * threshold;

    data::DemoICPPointClouds sample_data;
    std::vector<geometry::PointCloud> pcds(sample_data.GetPaths().size());
    std::vector<geometry::KDTreeFlann> kdtrees(sample_data.GetPaths().size());
    for (size_t i = 0; i < sample_data.GetPaths().size(); i++) {
        io::ReadPointCloud(sample_data.GetPaths()[i], pcds[i]);
        kdtrees[i].SetGeometry(pcds[i]);
    }

    std::vector<std::pair<int, int>> pair_ids;
    std::vector<Eigen::Matrix4d> transformations;
    ReadLogFile(log_filename, pair_ids, transformations);
    std::vector<Eigen::Matrix4d> gt_trans;
    ReadLogFile(gt_filename, pair_ids, gt_trans);

    double total_rmse = 0.0;
    int positive = 0;
    double positive_rmse = 0;
    for (size_t k = 0; k < pair_ids.size(); k++) {
        geometry::PointCloud source = pcds[pair_ids[k].second];
        source.Transform(transformations[k]);
        geometry::PointCloud gtsource = pcds[pair_ids[k].second];
        gtsource.Transform(gt_trans[k]);
        std::vector<int> indices(1);
        std::vector<double> distance2(1);
        int correspondence_num = 0;
        double rmse = 0.0;
        for (size_t i = 0; i < source.points_.size(); i++) {
            if (kdtrees[pair_ids[k].first].SearchKNN(gtsource.points_[i], 1,
                                                     indices, distance2) > 0) {
                if (distance2[0] < threshold2) {
                    correspondence_num++;
                    double new_dis =
                            (source.points_[i] -
                             pcds[pair_ids[k].first].points_[indices[0]])
                                    .norm();
                    rmse += new_dis * new_dis;
                }
            }
        }
        rmse = std::sqrt(rmse / (double)correspondence_num);
        utility::LogInfo("#{:d} < -- #{:d} : rmse {:.4f}", pair_ids[k].first,
                         pair_ids[k].second, rmse);
        total_rmse += rmse;
        if (rmse < threshold_rmse) {
            positive++;
            positive_rmse += rmse;
        }
    }
    utility::LogInfo("Average rmse {:.8f} ({:.8f} / {:d})",
                     total_rmse / (double)pair_ids.size(), total_rmse,
                     (int)pair_ids.size());
    utility::LogInfo("Average rmse of positives {:.8f} ({:.8f} / {:d})",
                     positive_rmse / (double)positive, positive_rmse, positive);
    utility::LogInfo("Accuracy {:.2f}% ({:d} / {:d})",
                     (double)positive * 100.0 / (double)pair_ids.size(),
                     positive, (int)pair_ids.size());
    return 0;
}
