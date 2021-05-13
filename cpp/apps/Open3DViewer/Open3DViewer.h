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

#include "open3d/visualization/visualizer/O3DVisualizer.h"

namespace open3d {
namespace visualization {
namespace gui {
class Menu;
}  // namespace gui
}  // namespace visualization
}  // namespace open3d

class Open3DViewer : public open3d::visualization::visualizer::O3DVisualizer {
    using Super = open3d::visualization::visualizer::O3DVisualizer;

public:
    Open3DViewer(const std::string& title, int width, int height);

    void LoadGeometry(const std::string& path);

    void OnFileOpen();
    void OnDragDropped(const char* path) override;

protected:
    void OnMenuItemSelected(
            open3d::visualization::gui::Menu::ItemId item_id) override;
};

#define WIDTH 1280
#define HEIGHT 960

int Run(int argc, const char* argv[]);
