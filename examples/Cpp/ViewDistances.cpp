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
    utility::PrintInfo("Usage:\n");
    utility::PrintInfo("    > ViewDistances source_file [options]\n");
    utility::PrintInfo("      View color coded distances of a point cloud.\n");
    utility::PrintInfo("\n");
    utility::PrintInfo("Basic options:\n");
    utility::PrintInfo("    --help, -h                : Print help information.\n");
    utility::PrintInfo("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
    utility::PrintInfo("    --max_distance d          : Set max distance. Must be positive.\n");
    utility::PrintInfo("    --mahalanobis_distance    : Compute the Mahalanobis distance.\n");
    utility::PrintInfo("    --nn_distance             : Compute the NN distance.\n");
    utility::PrintInfo("    --write_color_back        : Write color back to source_file.\n");
    utility::PrintInfo("    --without_gui             : Without GUI.\n");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    double max_distance = utility::GetProgramOptionAsDouble(
            argc, argv, "--max_distance", 0.0);
    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    if (pcd->IsEmpty()) {
        utility::PrintInfo("Empty point cloud.\n");
        return 1;
    }
    std::string binname =
            utility::filesystem::GetFileNameWithoutExtension(argv[1]) + ".bin";
    std::vector<double> distances(pcd->points_.size());
    if (utility::ProgramOptionExists(argc, argv, "--mahalanobis_distance")) {
        distances = pcd->ComputeMahalanobisDistance();
        FILE *f = fopen(binname.c_str(), "wb");
        fwrite(distances.data(), sizeof(double), distances.size(), f);
        fclose(f);
    } else if (utility::ProgramOptionExists(argc, argv, "--nn_distance")) {
        distances = pcd->ComputeNearestNeighborDistance();
        FILE *f = fopen(binname.c_str(), "wb");
        fwrite(distances.data(), sizeof(double), distances.size(), f);
        fclose(f);
    } else {
        FILE *f = fopen(binname.c_str(), "rb");
        if (f == NULL) {
            utility::PrintInfo("Cannot open bin file.\n");
            return 1;
        }
        if (fread(distances.data(), sizeof(double), pcd->points_.size(), f) !=
            pcd->points_.size()) {
            utility::PrintInfo("Cannot open bin file.\n");
            return 1;
        }
    }
    if (max_distance <= 0.0) {
        utility::PrintInfo("Max distance must be a positive value.\n");
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
