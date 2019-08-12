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

int main(int argc, char **argv) {
    using namespace open3d;

    if (argc < 2) {
        utility::LogInfo("Please provide input .mkv file\n");
        return -1;
    }

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    io::MKVReader mkv_reader;
    mkv_reader.Open(argv[1]);
    if (!mkv_reader.IsOpened()) {
        return -1;
    }

    auto json = mkv_reader.GetMetaData();
    for (auto iter = json.begin(); iter != json.end(); ++iter) {
        utility::LogInfo("{}: {}\n", iter.key(), json[iter.name()]);
    }

    bool stop = false;
    bool toggle_pause = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.CreateVisualizerWindow("Open3D Azure Kinect MKV player", 1920, 540);
    vis.GetRenderOption().image_stretch_option_ =
            visualization::RenderOption::ImageStretchOption::StretchKeepRatio;

    vis.RegisterKeyCallback(GLFW_KEY_ESCAPE,
                            [&](visualization::Visualizer *vis) {
                                stop = true;
                                return true;
                            });
    vis.RegisterKeyCallback(GLFW_KEY_SPACE,
                            [&](visualization::Visualizer *vis) {
                                toggle_pause = !toggle_pause;
                                return true;
                            });

    std::shared_ptr<geometry::RGBDImage> rgbd_vis = nullptr;
    while (!mkv_reader.IsEOF()) {
        if (stop) break;

        if (!toggle_pause) {
            std::shared_ptr<geometry::RGBDImage> rgbd = mkv_reader.NextFrame();
            if (rgbd == nullptr) continue;

            if (rgbd_vis == nullptr) {
                rgbd_vis = std::make_shared<geometry::RGBDImage>();
                vis.AddGeometry(rgbd_vis);
            } else {
                *rgbd_vis = *rgbd;
            }
        }

        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();
    }

    mkv_reader.Close();
}
