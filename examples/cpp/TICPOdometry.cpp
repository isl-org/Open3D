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

// This is an example of Multi-Scale ICP Registration.
// This takes a config.txt file, an example of which is provided in
// Open3D/examples/test_data/ICP/TMultiScaleICPRegConfig.txt.
//
// To run this from build directory use the following command:
// build$ ./bin/examples/TICPRegistration [Device] [Path to Config]
// [Device]: CPU:0 / CUDA:0 ...
// [Sample Config Path]: ../examples/test_data/ICP/TMultiScaleICPRegConfig.txt

#include <fstream>
#include <sstream>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::t::pipelines::registration;

int end_range = 100;
bool visualize_output = false;

// ICP ConvergenceCriteria.
double relative_fitness = 1e-6;
double relative_rmse = 1e-6;
// This is overriden by the scale-wise iteration set in config file.
int max_iterations = 30;

// For each frame registration using MultiScaleICP.
std::vector<int> iterations;
std::vector<double> voxel_sizes;
std::vector<double> search_radius;

std::string path_config_file;
std::string path_dataset;
std::string registration_method;
std::string verbosity;

void PrintHelp() {
    PrintOpen3DVersion();
    utility::LogInfo("Usage :");
    utility::LogInfo("    > TMultiScaleICP [device] [path to config.txt file]");
}

void ReadConfigFile() {
    std::ifstream cFile(path_config_file);
    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                       line.end());
            if (line[0] == '#' || line.empty()) continue;

            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);

            if (name == "dataset_path") {
                path_dataset = value;
            } else if (name == "end_range") {
                std::istringstream is(value);
                end_range = std::stoi(value);
            } else if (name == "registration_method") {
                registration_method = value;
            } else if (name == "iteration") {
                std::istringstream is(value);
                iterations.push_back(std::stoi(value));
            } else if (name == "voxel_size") {
                std::istringstream is(value);
                voxel_sizes.push_back(std::stod(value));
            } else if (name == "search_radii") {
                std::istringstream is(value);
                search_radius.push_back(std::stod(value));
            } else if (name == "verbosity") {
                std::istringstream is(value);
                verbosity = value;
            }
        }
    } else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

    utility::LogInfo(" Dataset path: {}", path_dataset);
    if (end_range > 500) {
        utility::LogWarning(" Too large range. Memory might exceed.");
    }
    utility::LogInfo(" Range: 0 to {} pointcloud files in sequence.",
                     end_range);
    utility::LogInfo(" Registrtion method: {}", registration_method);
    std::cout << std::endl;

    std::cout << " Per frame registration: " << std::endl;

    std::cout << " Iterations: ";
    for (auto iteration : iterations) std::cout << iteration << " ";
    std::cout << std::endl;

    std::cout << " Voxel Sizes: ";
    for (auto voxel_size : voxel_sizes) std::cout << voxel_size << " ";
    std::cout << std::endl;

    std::cout << " Search Radius Sizes: ";
    for (auto search_radii : search_radius) std::cout << search_radii << " ";
    std::cout << std::endl;

    std::cout << " Press Enter To Continue... " << std::endl;
    std::getchar();
}

// Visualize transformed source and target tensor pointcloud.
void VisualizeRegistration(const open3d::t::geometry::PointCloud &source,
                           const open3d::t::geometry::PointCloud &target,
                           const core::Tensor &transformation,
                           const std::string &window_name) {
    auto source_transformed = source;
    source_transformed = source_transformed.Transform(transformation);
    auto source_transformed_legacy = source_transformed.ToLegacyPointCloud();
    auto target_legacy = target.ToLegacyPointCloud();

    std::shared_ptr<geometry::PointCloud> source_transformed_ptr =
            std::make_shared<geometry::PointCloud>(source_transformed_legacy);
    std::shared_ptr<geometry::PointCloud> target_ptr =
            std::make_shared<geometry::PointCloud>(target_legacy);

    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  window_name);
}

void LoadPointCloudsCPU(
        std::vector<open3d::t::geometry::PointCloud> &pointclouds_host) {
    std::vector<std::string> filename;

    for (int i = 0; i < end_range; i++) {
        filename.push_back(path_dataset + std::to_string(i) +
                           std::string(".pcd"));
    }

    // Saving the vector of pointclouds on CPU RAM, as GPU RAM might go out of
    // memory.
    std::vector<t::geometry::PointCloud> pointclouds(filename.size());

    try {
        int i = 0;
        for (auto &path : filename)
            t::io::ReadPointCloud(path, pointclouds[i++],
                                  {"auto", false, false, true});
    } catch (...) {
        utility::LogError(
                " Failed to read pointcloud in sequence. Ensure pointcloud "
                "files are present in the given dataset path in continuous "
                "sequence from 0 to {}. Also, in case of large range, the "
                "system might be going out-of-memory. ",
                end_range);
    }
    pointclouds_host = pointclouds;
}

int main(int argc, char *argv[]) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc != 3) {
        PrintHelp();
        return 1;
    }

    auto device = core::Device(argv[1]);
    path_config_file = std::string(argv[2]);
    ReadConfigFile();
    std::vector<open3d::t::geometry::PointCloud> pointclouds_host;
    LoadPointCloudsCPU(pointclouds_host);

    utility::VerbosityLevel verb;
    if (verbosity == "Debug") {
        verb = utility::VerbosityLevel::Debug;
    } else if (verbosity == "Info") {
        verb = utility::VerbosityLevel::Info;
    } else if (verbosity == "Warning") {
        verb = utility::VerbosityLevel::Warning;
    } else {
        verb = utility::VerbosityLevel::Error;
    }
    utility::SetVerbosityLevel(verb);

    std::shared_ptr<TransformationEstimation> estimation;
    if (registration_method == "PointToPoint") {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
    } else if (registration_method == "PointToPlane") {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
    } else {
        utility::LogError(" Registration method {}, not implemented.",
                          registration_method);
    }

    // Warm up.
    auto pointcloud_warmup = pointclouds_host[0].To(device, true);
    // Getting dtype.
    auto dtype = pointcloud_warmup.GetPoints().GetDtype();

    auto warm_up_result = RegistrationICPMultiScale(
            pointcloud_warmup, pointcloud_warmup, {5}, {0.05}, {0.1},
            core::Tensor::Eye(4, dtype, device), *estimation,
            ICPConvergenceCriteria(relative_fitness, relative_rmse, 1));

    core::Tensor initial_transform = core::Tensor::Eye(4, dtype, device);
    core::Tensor cumulative_transform = initial_transform.Clone();

    double total_processing_time = 0;
    for (int i = 0; i < end_range - 1; i++) {
        utility::Timer time_icp_odom_loop;
        time_icp_odom_loop.Start();
        auto source = pointclouds_host[i].To(device);
        auto target = pointclouds_host[i + 1].To(device);

        auto result = RegistrationICPMultiScale(
                source, target, iterations, voxel_sizes, search_radius,
                initial_transform, *estimation,
                ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                       max_iterations));

        cumulative_transform =
                cumulative_transform.Matmul(result.transformation_.Inverse());

        time_icp_odom_loop.Stop();
        total_processing_time += time_icp_odom_loop.GetDuration();
        if (visualize_output) {
            VisualizeRegistration(source, target, result.transformation_,
                                  " Registration of " + std::to_string(i) +
                                          " and " + std::to_string(i + 1) +
                                          " frame.");
        }
        utility::LogDebug(" Registraion took: {}",
                          time_icp_odom_loop.GetDuration());
        utility::LogDebug(" Cumulative Transformation: \n{}\n",
                          cumulative_transform.ToString());
    }

    utility::LogInfo(" Total Time: {}, \n Transformation: \n{}\n",
                     total_processing_time, cumulative_transform.ToString());

    return 0;
}
