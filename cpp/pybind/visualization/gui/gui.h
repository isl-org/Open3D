// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace geometry {
class Image;
}

namespace visualization {
namespace rendering {
class Open3DScene;
}

namespace gui {

void InitializeForPython(std::string resource_path = "", bool headless = false);
std::shared_ptr<geometry::Image> RenderToImageWithoutWindow(
        rendering::Open3DScene *scene, int width, int height);
std::shared_ptr<geometry::Image> RenderToDepthImageWithoutWindow(
        rendering::Open3DScene *scene,
        int width,
        int height,
        bool z_in_view_space = false);

void pybind_gui(py::module &m);

void pybind_gui_events(py::module &m);
void pybind_gui_classes(py::module &m);

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
