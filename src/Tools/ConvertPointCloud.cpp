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

#include <limits>

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:\n");
    utility::LogInfo("    > ConvertPointCloud source_file target_file [options]\n");
    utility::LogInfo("    > ConvertPointCloud source_directory target_directory [options]\n");
    utility::LogInfo("      Read point cloud from source file and convert it to target file.\n");
    utility::LogInfo("\n");
    utility::LogInfo("Options (listed in the order of execution priority):\n");
    utility::LogInfo("    --help, -h                : Print help information.\n");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).\n");
    utility::LogInfo("    --clip_x_min x0           : Clip points with x coordinate < x0.\n");
    utility::LogInfo("    --clip_x_max x1           : Clip points with x coordinate > x1.\n");
    utility::LogInfo("    --clip_y_min y0           : Clip points with y coordinate < y0.\n");
    utility::LogInfo("    --clip_y_max y1           : Clip points with y coordinate > y1.\n");
    utility::LogInfo("    --clip_z_min z0           : Clip points with z coordinate < z0.\n");
    utility::LogInfo("    --clip_z_max z1           : Clip points with z coordinate > z1.\n");
    utility::LogInfo("    --filter_mahalanobis d    : Filter out points with Mahalanobis distance > d.\n");
    utility::LogInfo("    --uniform_sample_every K  : Downsample the point cloud uniformly. Keep only\n");
    utility::LogInfo("                              : one point for every K points.\n");
    utility::LogInfo("    --voxel_sample voxel_size : Downsample the point cloud with a voxel.\n");
    utility::LogInfo("    --estimate_normals radius : Estimate normals using a search neighborhood of\n");
    utility::LogInfo("                                radius. The normals are oriented w.r.t. the\n");
    utility::LogInfo("                                original normals of the pointcloud if they\n");
    utility::LogInfo("                                exist. Otherwise, they are oriented towards -Z\n");
    utility::LogInfo("                                direction.\n");
    utility::LogInfo("    --estimate_normals_knn k  : Estimate normals using a search with k nearest\n");
    utility::LogInfo("                                neighbors. The normals are oriented w.r.t. the\n");
    utility::LogInfo("                                original normals of the pointcloud if they\n");
    utility::LogInfo("                                exist. Otherwise, they are oriented towards -Z\n");
    utility::LogInfo("                                direction.\n");
    utility::LogInfo("    --orient_normals [x,y,z]  : Orient the normals w.r.t the direction [x,y,z].\n");
    utility::LogInfo("    --camera_location [x,y,z] : Orient the normals w.r.t camera location [x,y,z].\n");
    // clang-format on
}

void convert(int argc,
             char **argv,
             const std::string &file_in,
             const std::string &file_out) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;
    auto pointcloud_ptr = io::CreatePointCloudFromFile(file_in.c_str());
    size_t point_num_in = pointcloud_ptr->points_.size();
    bool processed = false;

    // clip
    if (utility::ProgramOptionExistsAny(
                argc, argv,
                {"--clip_x_min", "--clip_x_max", "--clip_y_min", "--clip_y_max",
                 "--clip_z_min", "--clip_z_max"})) {
        Eigen::Vector3d min_bound, max_bound;
        min_bound(0) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_x_min",
                std::numeric_limits<double>::lowest());
        min_bound(1) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_y_min",
                std::numeric_limits<double>::lowest());
        min_bound(2) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_z_min",
                std::numeric_limits<double>::lowest());
        max_bound(0) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_x_max", std::numeric_limits<double>::max());
        max_bound(1) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_y_max", std::numeric_limits<double>::max());
        max_bound(2) = utility::GetProgramOptionAsDouble(
                argc, argv, "--clip_z_max", std::numeric_limits<double>::max());
        pointcloud_ptr = pointcloud_ptr->Crop(min_bound, max_bound);
        processed = true;
    }

    // filter_mahalanobis
    double mahalanobis_threshold = utility::GetProgramOptionAsDouble(
            argc, argv, "--filter_mahalanobis", 0.0);
    if (mahalanobis_threshold > 0.0) {
        auto mahalanobis = pointcloud_ptr->ComputeMahalanobisDistance();
        std::vector<size_t> indices;
        for (size_t i = 0; i < pointcloud_ptr->points_.size(); i++) {
            if (mahalanobis[i] < mahalanobis_threshold) {
                indices.push_back(i);
            }
        }
        auto pcd = pointcloud_ptr->SelectDownSample(indices);
        utility::LogDebug(
                "Based on Mahalanobis distance, {:d} points were filtered.\n",
                (int)(pointcloud_ptr->points_.size() - pcd->points_.size()));
        pointcloud_ptr = pcd;
    }

    // uniform_downsample
    int every_k = utility::GetProgramOptionAsInt(argc, argv,
                                                 "--uniform_sample_every", 0);
    if (every_k > 1) {
        utility::LogDebug(
                "Downsample point cloud uniformly every {:d} points.\n",
                every_k);
        pointcloud_ptr = pointcloud_ptr->UniformDownSample(every_k);
        processed = true;
    }

    // voxel_downsample
    double voxel_size = utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_sample", 0.0);
    if (voxel_size > 0.0) {
        utility::LogDebug("Downsample point cloud with voxel size {:.4f}.\n",
                          voxel_size);
        pointcloud_ptr = pointcloud_ptr->VoxelDownSample(voxel_size);
        processed = true;
    }

    // estimate_normals
    double radius = utility::GetProgramOptionAsDouble(
            argc, argv, "--estimate_normals", 0.0);
    if (radius > 0.0) {
        utility::LogDebug("Estimate normals with search radius {:.4f}.\n",
                          radius);
        pointcloud_ptr->EstimateNormals(
                geometry::KDTreeSearchParamRadius(radius));
        processed = true;
    }

    int k = utility::GetProgramOptionAsInt(argc, argv, "--estimate_normals_knn",
                                           0);
    if (k > 0) {
        utility::LogDebug("Estimate normals with search knn {:d}.\n", k);
        pointcloud_ptr->EstimateNormals(geometry::KDTreeSearchParamKNN(k));
        processed = true;
    }

    // orient_normals
    Eigen::VectorXd direction = utility::GetProgramOptionAsEigenVectorXd(
            argc, argv, "--orient_normals");
    if (direction.size() == 3 && pointcloud_ptr->HasNormals()) {
        utility::LogDebug("Orient normals to [%.2f, %.2f, %.2f].\n",
                          direction(0), direction(1), direction(2));
        Eigen::Vector3d dir(direction);
        pointcloud_ptr->OrientNormalsToAlignWithDirection(dir);
        processed = true;
    }
    Eigen::VectorXd camera_loc = utility::GetProgramOptionAsEigenVectorXd(
            argc, argv, "--camera_location");
    if (camera_loc.size() == 3 && pointcloud_ptr->HasNormals()) {
        utility::LogDebug("Orient normals towards [%.2f, %.2f, %.2f].\n",
                          camera_loc(0), camera_loc(1), camera_loc(2));
        Eigen::Vector3d loc(camera_loc);
        pointcloud_ptr->OrientNormalsTowardsCameraLocation(loc);
        processed = true;
    }

    size_t point_num_out = pointcloud_ptr->points_.size();
    if (processed) {
        utility::LogInfo(
                "Processed point cloud from {:d} points to {:d} points.\n",
                (int)point_num_in, (int)point_num_out);
    }
    io::WritePointCloud(file_out.c_str(), *pointcloud_ptr, false, true);
}

int main(int argc, char **argv) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;

    if (argc < 3 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);

    if (FileExists(argv[1])) {
        convert(argc, argv, argv[1], argv[2]);
    } else if (DirectoryExists(argv[1])) {
        MakeDirectoryHierarchy(argv[2]);
        std::vector<std::string> filenames;
        ListFilesInDirectory(argv[1], filenames);
        for (const auto &fn : filenames) {
            convert(argc, argv, fn,
                    GetRegularizedDirectoryName(argv[2]) +
                            GetFileNameWithoutDirectory(fn));
        }
    } else {
        utility::LogWarning("File or directory does not exist.\n");
    }

    return 1;
}
