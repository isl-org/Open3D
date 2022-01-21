// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/visualization/app/Viewer.h"

#include <iostream>
#include <string>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Native.h"
#include "open3d/visualization/visualizer/GuiVisualizer.h"

namespace open3d {
namespace visualization {
namespace app {

static const std::string usage = "usage: ./Open3D [meshfile|pointcloud]";
static const int width = 1280;
static const int height = 960;

void RunViewer(int argc, const char *argv[]) {
    std::function<void(const std::string &)> print_fcn =
            utility::Logger::GetInstance().GetPrintFunction();
    utility::Logger::GetInstance().ResetPrintFunction();

    const char *path = nullptr;
    if (argc == 2) {
        path = argv[1];
    } else if (argc > 2) {
        utility::LogWarning(usage.c_str());
    }

    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto vis = std::make_shared<open3d::visualization::GuiVisualizer>(
            "Open3D", width, height);
    if (path && path[0] != '\0') {
        vis->LoadGeometry(path);
    }
    gui::Application::GetInstance().AddWindow(vis);
    // When Run() ends, Filament will be stopped, so we can't be holding on
    // to any GUI objects.
    vis.reset();

    app.Run();

    utility::Logger::GetInstance().SetPrintFunction(print_fcn);
}

}  // namespace app
}  // namespace visualization
}  // namespace open3d
