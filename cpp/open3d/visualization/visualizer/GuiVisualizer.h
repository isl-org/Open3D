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

#pragma once

#include <vector>

#include "open3d/visualization/gui/Window.h"

namespace open3d {

namespace geometry {
class AxisAlignedBoundingBox;
class Geometry;
}  // namespace geometry

namespace visualization {

namespace gui {
struct Theme;
}

class GuiVisualizer : public gui::Window {
    using Super = gui::Window;

public:
    GuiVisualizer(const std::string& title, int width, int height);
    GuiVisualizer(const std::vector<std::shared_ptr<const geometry::Geometry>>&
                          geometries,
                  const std::string& title,
                  int width,
                  int height,
                  int left,
                  int top);
    virtual ~GuiVisualizer();

    void SetTitle(const std::string& title);
    void SetGeometry(std::shared_ptr<const geometry::Geometry> geometry,
                     bool loaded_model);

    bool SetIBL(const char* path);

    /// Loads asynchronously, will return immediately.
    void LoadGeometry(const std::string& path);

    void ExportCurrentImage(const std::string& path);

    void Layout(const gui::Theme& theme) override;

protected:
    // Add custom items to the application menu (only relevant on macOS)
    void AddItemsToAppMenu(
            const std::vector<std::pair<std::string, gui::Menu::ItemId>>&
                    items);

    void OnMenuItemSelected(gui::Menu::ItemId item_id) override;
    void OnDragDropped(const char* path) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void Init();
};

}  // namespace visualization
}  // namespace open3d
