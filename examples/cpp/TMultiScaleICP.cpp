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

// For each frame registration using MultiScaleICP.
std::vector<double> voxel_sizes;
std::vector<double> search_radius;
std::vector<ICPConvergenceCriteria> criterias;

std::string path_config_file;
std::string path_source;
std::string path_target;
std::string registration_method;
std::string verbosity;

// Initial transformation guess for registation.
std::vector<double> initial_transform_flat = {
        0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
        0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

void PrintHelp() {
    PrintOpen3DVersion();
    utility::LogInfo("Usage :");
    utility::LogInfo("    > TMultiScaleICP [device] [path to config.txt file]");
}

void ReadConfigFile() {
    std::ifstream cFile(path_config_file);
    std::vector<double> relative_fitness;
    std::vector<double> relative_rmse;
    std::vector<int> max_iterations;

    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                       line.end());
            if (line[0] == '#' || line.empty()) continue;

            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);

            if (name == "source_path") {
                path_source = value;
            } else if (name == "target_path") {
                path_target = value;
            } else if (name == "registration_method") {
                registration_method = value;
            } else if (name == "criteria.relative_fitness") {
                std::istringstream is(value);
                relative_fitness.push_back(std::stod(value));
            } else if (name == "criteria.relative_rmse") {
                std::istringstream is(value);
                relative_rmse.push_back(std::stod(value));
            } else if (name == "criteria.max_iterations") {
                std::istringstream is(value);
                max_iterations.push_back(std::stoi(value));
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

    utility::LogInfo(" Source path: {}", path_source);
    utility::LogInfo(" Target path: {}", path_target);
    utility::LogInfo(" Registrtion method: {}", registration_method);
    std::cout << std::endl;

    std::cout << " Initial Transformation Guess: " << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << " " << initial_transform_flat[i * 4 + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << " Voxel Sizes: ";
    for (auto voxel_size : voxel_sizes) std::cout << voxel_size << " ";
    std::cout << std::endl;

    std::cout << " Search Radius Sizes: ";
    for (auto search_radii : search_radius) std::cout << search_radii << " ";
    std::cout << std::endl;

    std::cout << " ICPCriteria: " << std::endl;
    std::cout << "   Max Iterations: ";
    for (auto iteration : max_iterations) std::cout << iteration << " ";
    std::cout << std::endl;
    std::cout << "   Relative Fitness: ";
    for (auto fitness : relative_fitness) std::cout << fitness << " ";
    std::cout << std::endl;
    std::cout << "   Relative RMSE: ";
    for (auto rmse : relative_rmse) std::cout << rmse << " ";
    std::cout << std::endl;

    size_t length = voxel_sizes.size();
    if (search_radius.size() != length || max_iterations.size() != length ||
        relative_fitness.size() != length || relative_rmse.size() != length) {
        utility::LogError(
                " Length of vector: voxel_sizes, search_sizes, max_iterations, "
                "relative_fitness, relative_rmse must be same.");
    }

    for (int i = 0; i < (int)length; i++) {
        auto criteria = ICPConvergenceCriteria(
                relative_fitness[i], relative_rmse[i], max_iterations[i]);
        criterias.push_back(criteria);
    }

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

int main(int argc, char *argv[]) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc != 3) {
        PrintHelp();
        return 1;
    }

    auto device = core::Device(argv[1]);
    path_config_file = std::string(argv[2]);
    ReadConfigFile();

    t::geometry::PointCloud source(device), target(device);

    t::io::ReadPointCloud(path_source, source, {"auto", false, false, true});
    t::io::ReadPointCloud(path_target, target, {"auto", false, false, true});

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

    source = source.To(device);
    target = target.To(device);

    auto dtype = source.GetPoints().GetDtype();

    std::shared_ptr<TransformationEstimation> estimation;
    if (registration_method == "PointToPoint") {
        estimation = std::make_shared<TransformationEstimationPointToPoint>();
    } else if (registration_method == "PointToPlane") {
        estimation = std::make_shared<TransformationEstimationPointToPlane>();
    } else {
        utility::LogError(" Registration method {}, not implemented.",
                          registration_method);
    }

    core::Tensor initial_transformation =
            core::Tensor(initial_transform_flat, {4, 4}, dtype, device);
    utility::Timer time_multiscaleICP;

    // Warm Up.
    std::vector<ICPConvergenceCriteria> warm_up_criteria = {
            ICPConvergenceCriteria(0.01, 0.01, 1)};
    auto warm_up_result = RegistrationICPMultiScale(
            source, target, {0.05}, warm_up_criteria, {0.1},
            core::Tensor::Eye(4, dtype, device), *estimation);

    VisualizeRegistration(source, target, initial_transformation,
                          " Before Registration ");

    time_multiscaleICP.Start();
    auto result = RegistrationICPMultiScale(
            source, target, voxel_sizes, criterias, search_radius,
            initial_transformation, *estimation);
    time_multiscaleICP.Stop();

    utility::LogInfo(
            " Total Time: {}, Fitness: {}, RMSE: {}, \n Transformation: \n{}\n",
            time_multiscaleICP.GetDuration(), result.fitness_,
            result.inlier_rmse_, result.transformation_.ToString());

    VisualizeRegistration(source, target, result.transformation_,
                          " After Registration ");

    return 0;
}
