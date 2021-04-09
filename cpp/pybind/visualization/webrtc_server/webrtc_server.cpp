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

#include "pybind/visualization/webrtc_server/webrtc_server.h"

#include "open3d/visualization/webrtc_server/WebRTCServer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

void pybind_webrtc_server(py::module &m) {
    py::class_<WebRTCServer> webrtc_server(m, "WebRTCServer",
                                           "Global WebRTCServer singleton.");

    webrtc_server.def("__repr__", [](const WebRTCServer &ws) {
        return std::string("Global WebRTCServer instance.");
    });
    webrtc_server.def_property_readonly_static(
            "instance",
            [](py::object) -> WebRTCServer & {
                return WebRTCServer::GetInstance();
            },
            py::return_value_policy::reference,
            "Gets the WebRTCServer singleton (read-only).");
    webrtc_server.def("call_web_request_api", &WebRTCServer::CallWebRequestAPI,
                      "entry_point"_a, "req_info_str"_a = "", "json_st"_a = "");
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
