// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <json/json.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;
namespace sc = std::chrono;

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

Json::Value GenerateDatasetConfig(const std::string &output_path,
                                  const std::string &bagfile) {
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

    value["name"] = bagfile;
    value["max_depth"] = 3.0;
    value["voxel_size"] = 0.05;
    value["max_depth_diff"] = 0.07;
    value["preference_loop_closure_odometry"] = 0.1;
    value["preference_loop_closure_registration"] = 5.0;
    value["tsdf_cubic_size"] = 3.0;
    value["icp_method"] = "color";
    value["global_registration"] = "ransac";
    value["python_multi_threading"] = true;

    return value;
}

void PrintUsage() {
    PrintOpen3DVersion();
    utility::LogInfo("Usage:");
    // clang-format off
    utility::LogInfo("RealSenseBagReader [-V] --input input.bag [--output path]");
    // clang-format on
}

int main(int argc, char **argv) {
    if (!utility::ProgramOptionExists(argc, argv, "--input")) {
        PrintUsage();
        return 1;
    }
    if (utility::ProgramOptionExists(argc, argv, "-V")) {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    } else {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    }
    std::string bag_filename =
            utility::GetProgramOptionAsString(argc, argv, "--input");

    bool write_image = false;
    std::string output_path;
    if (!utility::ProgramOptionExists(argc, argv, "--output")) {
        utility::LogInfo("No output image path, only play bag.");
    } else {
        output_path = utility::GetProgramOptionAsString(argc, argv, "--output");
        if (output_path.empty()) {
            utility::LogWarning("Output path {} is empty, only play bag.",
                                output_path);
        }
        if (utility::filesystem::DirectoryExists(output_path)) {
            utility::LogWarning(
                    "Output path {} already existing, only play bag.",
                    output_path);
        } else if (!utility::filesystem::MakeDirectory(output_path)) {
            utility::LogWarning("Unable to create path {}, only play bag.",
                                output_path);
        } else {
            utility::LogInfo("Decompress images to {}", output_path);
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/color");
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/depth");
            write_image = true;
        }
    }

    t::io::RSBagReader bag_reader;
    bag_reader.Open(bag_filename);
    if (!bag_reader.IsOpened()) {
        utility::LogError("Unable to open {}", bag_filename);
        return 1;
    }

    bool flag_exit = false;
    bool flag_play = true;
    visualization::VisualizerWithKeyCallback vis;
    visualization::SetGlobalColorMap(
            visualization::ColorMap::ColorMapOption::Gray);
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
    vis.RegisterKeyCallback(GLFW_KEY_LEFT, [&](visualization::Visualizer *vis) {
        uint64_t now = bag_reader.GetTimestamp();
        if (bag_reader.SeekTimestamp(now < 1'000'000 ? 0 : now - 1'000'000))
            utility::LogInfo("Seek back 1s");
        else
            utility::LogWarning("Seek back 1s failed");
        return true;
    });
    vis.RegisterKeyCallback(
            GLFW_KEY_RIGHT, [&](visualization::Visualizer *vis) {
                uint64_t now = bag_reader.GetTimestamp();
                if (bag_reader.SeekTimestamp(now + 1'000'000))
                    utility::LogInfo("Seek forward 1s");
                else
                    utility::LogWarning("Seek forward 1s failed");
                return true;
            });

    vis.CreateVisualizerWindow("Open3D Intel RealSense bag player", 1920, 540);
    utility::LogInfo(
            "Starting to play. Press [SPACE] to pause. Press [ESC] to "
            "exit.");

    bool is_geometry_added = false;
    int idx = 0;
    const auto bag_metadata = bag_reader.GetMetadata();
    utility::LogInfo("Recorded with device {}", bag_metadata.device_name_);
    utility::LogInfo("    Serial number: {}", bag_metadata.serial_number_);
    utility::LogInfo("Video resolution: {}x{}", bag_metadata.width_,
                     bag_metadata.height_);
    utility::LogInfo("      frame rate: {}", bag_metadata.fps_);
    utility::LogInfo(
            "      duration: {:.6f}s",
            static_cast<double>(bag_metadata.stream_length_usec_) * 1e-6);
    utility::LogInfo("      color pixel format: {}",
                     bag_metadata.color_format_);
    utility::LogInfo("      depth pixel format: {}",
                     bag_metadata.depth_format_);

    if (write_image) {
        io::WriteIJsonConvertibleToJSON(
                fmt::format("{}/intrinsic.json", output_path), bag_metadata);
        WriteJsonToFile(fmt::format("{}/config.json", output_path),
                        GenerateDatasetConfig(output_path, bag_filename));
    }
    const auto frame_interval = sc::duration<double>(1. / bag_metadata.fps_);

    using legacyRGBDImage = open3d::geometry::RGBDImage;
    auto last_frame_time = std::chrono::steady_clock::now();
    legacyRGBDImage im_rgbd = bag_reader.NextFrame().ToLegacyRGBDImage();
    while (!bag_reader.IsEOF() && !flag_exit) {
        if (flag_play) {
            // create shared_ptr with no-op deleter for stack RGBDImage
            auto ptr_im_rgbd = std::shared_ptr<legacyRGBDImage>(
                    &im_rgbd, [](legacyRGBDImage *) {});
            // Improve depth visualization by scaling
            /* im_rgbd.depth_.LinearTransform(0.25); */
            if (ptr_im_rgbd->IsEmpty()) continue;

            if (!is_geometry_added) {
                vis.AddGeometry(ptr_im_rgbd);
                is_geometry_added = true;
            }

            ++idx;
            if (write_image)
#pragma omp parallel sections
            {
#pragma omp section
                {
                    auto color_file = fmt::format("{0}/color/{1:05d}.jpg",
                                                  output_path, idx);
                    utility::LogInfo("Writing to {}", color_file);
                    io::WriteImage(color_file, im_rgbd.color_);
                }
#pragma omp section
                {
                    auto depth_file = fmt::format("{0}/depth/{1:05d}.png",
                                                  output_path, idx);
                    utility::LogInfo("Writing to {}", depth_file);
                    io::WriteImage(depth_file, im_rgbd.depth_);
                }
            }
            vis.UpdateGeometry();
            vis.UpdateRender();

            std::this_thread::sleep_until(last_frame_time + frame_interval);
            last_frame_time = std::chrono::steady_clock::now();
            im_rgbd = bag_reader.NextFrame().ToLegacyRGBDImage();
        }
        vis.PollEvents();
    }
    bag_reader.Close();
}
