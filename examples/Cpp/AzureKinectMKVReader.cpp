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
#include <thread>

#include "Open3D/Open3D.h"

using namespace open3d;

void PrintUsage() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:\n");
    utility::LogInfo("AzureKinectMKVReader --input input.mkv [--output] [path]\n");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (!utility::ProgramOptionExists(argc, argv, "--input")) {
        PrintUsage();
        return 1;
    }
    std::string mkv_filename =
            utility::GetProgramOptionAsString(argc, argv, "--input");

    bool write_image = false;
    std::string output_path;
    if (!utility::ProgramOptionExists(argc, argv, "--output")) {
        utility::LogInfo("No output image path, only play mkv.\n");
    } else {
        output_path = utility::GetProgramOptionAsString(argc, argv, "--output");
        if (utility::filesystem::DirectoryExists(output_path)) {
            utility::LogError(
                    "Output path {} already existing, only play mkv.\n",
                    output_path);
        } else if (!utility::filesystem::MakeDirectory(output_path)) {
            utility::LogError("Unable to create path {}, only play mkv.\n",
                              output_path);
        } else {
            utility::LogInfo("Decompress images to {}\n", output_path);
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/color");
            utility::filesystem::MakeDirectoryHierarchy(output_path + "/depth");
            write_image = true;
        }
    }

    io::MKVReader mkv_reader;
    mkv_reader.Open(mkv_filename);
    if (!mkv_reader.IsOpened()) {
        utility::LogError("Unable to open {}\n", mkv_filename);
        return 1;
    }

    bool flag_stop = false;
    bool flag_play = true;
    visualization::VisualizerWithKeyCallback vis;
    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer *vis) {
                                flag_stop = true;
                                return true;
                            });
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
                if (flag_play) {
                    utility::LogInfo(
                            "Playback paused, press [SPACE] to continue\n");
                } else {
                    utility::LogInfo(
                            "Playback resumed, press [SPACE] to pause\n");
                }
                flag_play = !flag_play;
                return true;
            });

    vis.CreateVisualizerWindow("Open3D Azure Kinect MKV player", 1920, 540);
    utility::LogInfo(
            "Starting to play. Press [SPACE] to pause. Press [ESC] to exit.\n");
    bool is_geometry_added = false;
    int idx = 0;
    while (!mkv_reader.IsEOF() && !flag_stop) {
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
                utility::LogInfo("Writing to {}\n", color_file);
                io::WriteImage(color_file, im_rgbd->color_);

                auto depth_file =
                        fmt::format("{0}/depth/{1:05d}.png", output_path, idx);
                utility::LogInfo("Writing to {}\n", depth_file);
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
