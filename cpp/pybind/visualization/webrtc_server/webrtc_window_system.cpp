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

#include "pybind/visualization/webrtc_server/webrtc_window_system.h"

#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

static void pybind_webrtc_server_classes(py::module &m) {
    py::class_<WebRTCWindowSystem> webrtc_ws(
            m, "WebRTCWindowSystem", "Global WebRTCWindowSystem singleton.");

    webrtc_ws.def("__repr__", [](const WebRTCWindowSystem &ws) {
        return std::string("Global Open3D WebRTCWindowSystem instance.");
    });
    webrtc_ws.def_property_readonly_static(
            "instance",
            [](py::object) -> std::shared_ptr<WebRTCWindowSystem> {
                return WebRTCWindowSystem::GetInstance();
            },
            "Gets the WebRTCWindowSystem singleton.");
    webrtc_ws.def(
            "call_http_api", &WebRTCWindowSystem::CallHttpAPI, "entry_point"_a,
            "query_string"_a = "", "data"_a = "",
            "Emulates Open3D WebRTCWindowSystem's HTTP API calls. This is used "
            "when the HTTP handshake server is disabled (e.g. in Jupyter), and "
            "handshakes are done by this function.");
    webrtc_ws.def("enable_webrtc", &WebRTCWindowSystem::EnableWebRTC,
                  "Use WebRTC streams to display rendered gui window.");
    webrtc_ws.def("disable_http_handshake",
                  &WebRTCWindowSystem::DisableHttpHandshake,
                  "Disables the HTTP handshake server. In Jupyter environemnt, "
                  "WebRTC handshake is performed by call_http_api() with "
                  "Jupyter's own COMMS interface, thus the HTTP server shall "
                  "be turned off.");
}

void pybind_webrtc_server(py::module &m) {
    py::module m_submodule = m.def_submodule("webrtc_server");
    pybind_webrtc_server_classes(m_submodule);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
