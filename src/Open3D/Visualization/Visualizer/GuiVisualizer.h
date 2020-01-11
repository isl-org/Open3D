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

#include "Open3D/GUI/Window.h"

namespace open3d {

namespace geometry {
class Geometry;
}  // namespace geometry

namespace gui {
struct Theme;
}

namespace visualization {
class GuiVisualizer : public gui::Window {
    using Super = gui::Window;
public:
    GuiVisualizer(const std::vector<std::shared_ptr<const geometry::Geometry>>& geometries,
                  const std::string &title, int width, int height,
                  int left, int top);
    virtual ~GuiVisualizer();

    void LoadGeometry(const std::string& path);
    void ExportRGB(const std::string& path);
    void ExportDepth(const std::string& path);

    void Layout(const gui::Theme& theme) override;
    void OnMenuItemSelected(gui::Menu::ItemId itemId) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
