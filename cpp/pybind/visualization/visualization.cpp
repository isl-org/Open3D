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
