// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

void pybind_visualization_declarations(py::module &m) {
    py::module m_visualization = m.def_submodule("visualization");
    pybind_renderoption_declarations(m_visualization);
    pybind_viewcontrol_declarations(m_visualization);
    pybind_visualizer_declarations(m_visualization);
    pybind_visualization_utility_declarations(m_visualization);
    // For RPC serialization
    rendering::pybind_material_declarations(m_visualization);
#ifdef BUILD_GUI
    rendering::pybind_rendering_declarations(m_visualization);
    gui::pybind_gui_declarations(m_visualization);
    pybind_o3dvisualizer_declarations(m_visualization);
    app::pybind_app_declarations(m_visualization);
#endif
#ifdef BUILD_WEBRTC
    webrtc_server::pybind_webrtc_server_declarations(m_visualization);
#endif
}

void pybind_visualization_definitions(py::module &m) {
    auto m_visualization = static_cast<py::module>(m.attr("visualization"));
    pybind_renderoption_definitions(m_visualization);
    pybind_viewcontrol_definitions(m_visualization);
    pybind_visualizer_definitions(m_visualization);
    pybind_visualization_utility_definitions(m_visualization);
    // For RPC serialization
    rendering::pybind_material_definitions(m_visualization);
#ifdef BUILD_GUI
    rendering::pybind_rendering_definitions(m_visualization);
    gui::pybind_gui_definitions(m_visualization);
    pybind_o3dvisualizer_definitions(m_visualization);
    app::pybind_app_definitions(m_visualization);
#endif
#ifdef BUILD_WEBRTC
    webrtc_server::pybind_webrtc_server_definitions(m_visualization);
#endif
}

}  // namespace visualization
}  // namespace open3d
