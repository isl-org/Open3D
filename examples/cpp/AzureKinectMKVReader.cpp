// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <json/json.h>

#include <chrono>
#include <fstream>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;

void WriteJsonToFile(const std::string &filename, const Json::Value &value) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        utility::LogError("Cannot write to {}", filename);
    }

    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "\t";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(value, &out);
}

Json::Value GenerateDatasetConfig(const std::string &output_path) {
    Json::Value value;

    utility::LogInfo("Writing to config.json");
    utility::LogInfo(
            "Please change path_dataset and path_intrinsic when you move the "
            "dataset.");

    if (output_path[0] == '/') {  // global dir
        value["path_dataset"] = output_path;
        value["path_intrinsic"] = output_path + "/intrinsic.json";
    } else {  // relative dir
        auto pwd = utility::filesystem::GetWorkingDirectory();
        value["path_dataset"] = pwd + "/" + output_path;
        value["path_intrinsic"] = pwd + "/" + output_path + "/intrinsic.json";
    }

    value["name"] = "Azure Kinect Record";
    value["depth_max"] = 3.0;
    value["voxel_size"] = 0.05;
    value["depth_diff_max"] = 0.07;
    value["preference_loop_closure_odometry"] = 0.1;
    value["preference_loop_closure_registration"] = 5.0;
    value["tsdf_cubic_size"] = 3.0;
    value["icp_method"] = "color";
    value["global_registration"] = "ransac";
    value["python_multi_threading"] = true;

    return value;
}

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > AzureKinectMKVReader --input input.mkv [--output] [path]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char **argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc == 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"}) ||
        !utility::ProgramOptionExists(argc, argv, "--input")) {
        PrintHelp();
        return 1;
    }

    std::string mkv_filename =
            utility::GetProgramOptionAsString(argc, argv, "--input");

    bool write_image = false;
    std::string output_path;
    if (!utility::ProgramOptionExists(argc, argv, "--output")) {
        utility::LogInfo("No output image path, only play mkv.");
    } else {
        output_path = utility::GetProgramOptionAsString(argc, argv, "--output");
        if (output_path.empty()) {
            utility::LogError("Output path {} is empty, only play mkv.",
                              output_path);
            return 1;
        }
        if (utility::filesystem::DirectoryExists(output_path)) {
            utility::LogError("Output path {} already existing, only play mkv.",
                              output_path);
            return 1;
        } else if (!utility::filesystem::MakeDirectory(output_path)) {
            utility::LogError("Unable to create path {}, only play mkv.",
                              output_path);
            return 1;
        } else {
            utility::LogInfo("Decompress images to {}", output_path);
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/color");
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/depth");
            write_image = true;
        }
    }

    io::MKVReader mkv_reader;
    mkv_reader.Open(mkv_filename);
    if (!mkv_reader.IsOpened()) {
        utility::LogError("Unable to open {}", mkv_filename);
        return 1;
    }

    bool flag_exit = false;
    bool flag_play = true;
    visualization::VisualizerWithKeyCallback vis;
    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer *vis) {
                                flag_exit = true;
                                return true;
                            });
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
                if (flag_play) {
                    utility::LogInfo(
                            "Playback paused, press [SPACE] to continue");
                } else {
                    utility::LogInfo(
                            "Playback resumed, press [SPACE] to pause");
                }
                flag_play = !flag_play;
                return true;
            });

    vis.CreateVisualizerWindow("Open3D Azure Kinect MKV player", 1920, 540);
    utility::LogInfo(
            "Starting to play. Press [SPACE] to pause. Press [ESC] to "
            "exit.");

    bool is_geometry_added = false;
    int idx = 0;
    if (write_image) {
        io::WriteIJsonConvertibleToJSON(
                fmt::format("{}/intrinsic.json", output_path),
                mkv_reader.GetMetadata());
        WriteJsonToFile(fmt::format("{}/config.json", output_path),
                        GenerateDatasetConfig(output_path));
    }
    while (!mkv_reader.IsEOF() && !flag_exit) {
        if (flag_play) {
            auto im_rgbd = mkv_reader.NextFrame();
            if (im_rgbd == nullptr) continue;

            if (!is_geometry_added) {
                vis.AddGeometry(im_rgbd);
                is_geometry_added = true;
            }

            if (write_image) {
                auto color_file =
                        fmt::format("{0}/color/{1:05d}.jpg", output_path, idx);
                utility::LogInfo("Writing to {}", color_file);
                io::WriteImage(color_file, im_rgbd->color_);

                auto depth_file =
                        fmt::format("{0}/depth/{1:05d}.png", output_path, idx);
                utility::LogInfo("Writing to {}", depth_file);
                io::WriteImage(depth_file, im_rgbd->depth_);

                ++idx;
            }
        }

        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();
    }

    mkv_reader.Close();
}
