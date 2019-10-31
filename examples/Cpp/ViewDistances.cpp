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

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ViewDistances source_file [options]");
    utility::LogInfo("      View color coded distances of a point cloud.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    utility::LogInfo("    --max_distance d          : Set max distance. Must be positive.");
    utility::LogInfo("    --mahalanobis_distance    : Compute the Mahalanobis distance.");
    utility::LogInfo("    --nn_distance             : Compute the NN distance.");
    utility::LogInfo("    --write_color_back        : Write color back to source_file.");
    utility::LogInfo("    --without_gui             : Without GUI.");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    double max_distance = utility::GetProgramOptionAsDouble(
            argc, argv, "--max_distance", 0.0);
    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    if (pcd->IsEmpty()) {
        utility::LogWarning("Empty point cloud.");
        return 1;
    }
    std::string binname =
            utility::filesystem::GetFileNameWithoutExtension(argv[1]) + ".bin";
    std::vector<double> distances(pcd->points_.size());
    if (utility::ProgramOptionExists(argc, argv, "--mahalanobis_distance")) {
        distances = pcd->ComputeMahalanobisDistance();
        FILE *f = utility::filesystem::FOpen(binname, "wb");
        fwrite(distances.data(), sizeof(double), distances.size(), f);
        fclose(f);
    } else if (utility::ProgramOptionExists(argc, argv, "--nn_distance")) {
        distances = pcd->ComputeNearestNeighborDistance();
        FILE *f = utility::filesystem::FOpen(binname, "wb");
        fwrite(distances.data(), sizeof(double), distances.size(), f);
        fclose(f);
    } else {
        FILE *f = utility::filesystem::FOpen(binname, "rb");
        if (f == NULL) {
            utility::LogWarning("Cannot open bin file.");
            return 1;
        }
        if (fread(distances.data(), sizeof(double), pcd->points_.size(), f) !=
            pcd->points_.size()) {
            utility::LogWarning("Cannot open bin file.");
            return 1;
        }
    }
    if (max_distance <= 0.0) {
        utility::LogWarning("Max distance must be a positive value.");
        return 1;
    }
    pcd->colors_.resize(pcd->points_.size());
    visualization::ColorMapHot colormap;
    for (size_t i = 0; i < pcd->points_.size(); i++) {
        pcd->colors_[i] = colormap.GetColor(distances[i] / max_distance);
    }
    if (utility::ProgramOptionExists(argc, argv, "--write_color_back")) {
        io::WritePointCloud(argv[1], *pcd);
    }
    if (!utility::ProgramOptionExists(argc, argv, "--without_gui")) {
        visualization::DrawGeometries({pcd}, "Point Cloud", 1920, 1080);
    }
    return 0;
}
