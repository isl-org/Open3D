// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/visualization/visualization.h"

#include "pybind/visualization/app/viewer.h"
#include "pybind/visualization/gui/gui.h"
#include "pybind/visualization/rendering/material.h"
#include "pybind/visualization/rendering/rendering.h"

#ifdef BUILD_WEBRTC
#include "pybind/visualization/webrtc_server/webrtc_window_system.h"
#endif

namespace open3d {
namespace visualization {

void pybind_visualization(py::module &m) {
    py::module m_visualization = m.def_submodule("visualization");
    pybind_renderoption(m_visualization);
    pybind_viewcontrol(m_visualization);
    pybind_visualizer(m_visualization);
    pybind_visualization_utility(m_visualization);
    pybind_renderoption_method(m_visualization);
    pybind_viewcontrol_method(m_visualization);
    pybind_visualizer_method(m_visualization);
    pybind_visualization_utility_methods(m_visualization);
    rendering::pybind_material(m_visualization);  // For RPC serialization

#ifdef BUILD_GUI
    rendering::pybind_rendering(m_visualization);
    gui::pybind_gui(m_visualization);
    pybind_o3dvisualizer(m_visualization);
    app::pybind_app(m_visualization);
#endif

#ifdef BUILD_WEBRTC
    webrtc_server::pybind_webrtc_server(m_visualization);
#endif
}

}  // namespace visualization
}  // namespace open3d
