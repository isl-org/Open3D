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

#include "pybind/visualization/webrtc_server/webrtc_window_system.h"

#include "open3d/utility/Logging.h"
#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"
#include "pybind/docstring.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

static void pybind_webrtc_server_functions(py::module &m) {
    m.def(
            "call_http_api",
            [](const std::string &entry_point, const std::string &query_string,
               const std::string &data) {
                return WebRTCWindowSystem::GetInstance()->CallHttpAPI(
                        entry_point, query_string, data);
            },
            "entry_point"_a, "query_string"_a = "", "data"_a = "",
            "Emulates Open3D WebRTCWindowSystem's HTTP API calls. This is used "
            "when the HTTP handshake server is disabled (e.g. in Jupyter), and "
            "handshakes are done by this function.");
    m.def(
            "enable_webrtc",
            []() { WebRTCWindowSystem::GetInstance()->EnableWebRTC(); },
            "Use WebRTC streams to display rendered gui window.");
    m.def(
            "disable_http_handshake",
            []() { WebRTCWindowSystem::GetInstance()->DisableHttpHandshake(); },
            "Disables the HTTP handshake server. In Jupyter environemnt, "
            "WebRTC handshake is performed by call_http_api() with "
            "Jupyter's own COMMS interface, thus the HTTP server shall "
            "be turned off.");
    m.def(
            "register_data_channel_message_callback",
            [](const std::string &class_name,
               std::function<std::string(const std::string &)> callback) {
                return WebRTCWindowSystem::GetInstance()
                        ->RegisterDataChannelMessageCallback(class_name,
                                                             callback);
            },
            "class_name"_a, "callback"_a,
            R"(
Register callback for a data channel message.

When the data channel receives a valid JSON string, the ``class_name`` property
of the JSON object will be examined and the corresponding callback function will
be called. The string return value of the callback will be sent back as a reply,
if it is not empty.

.. note:: Ordering between the message and the reply is not guaranteed, since
   some messages may take longer to process than others. If ordering is important,
   use a unique message id for every message and include it in the reply.

.. code:: python

    # Register callback in Python
    import open3d as o3d
    o3d.visualization.webrtc_server.enable_webrtc()
    def send_ack(data):
        print(data)
        return "Received WebRTC data channel message with data: " + data

    o3d.visualization.webrtc_server.register_data_channel_message_callback(
        "webapp/input", send_ack)

.. code:: js

    /* Send message in JavaScript to trigger callback. this is WebRTCStreamer object */
    this.dataChannel.send('{"class_name":"webapp/input", "data":"Test event"}')
            )");

    docstring::FunctionDocInject(
            m, "register_data_channel_message_callback",
            {{"class_name",
              "The value of of the ``class_name`` property of the JSON "
              "object."},
             {"callback",
              "The callback function that will be called when a JSON object "
              "with the matching ``class_name`` is received via the data "
              "channel. The function should accept a ``string`` argument "
              "(corresponding to the event data, such as form data or updated "
              "value of a slider) and return a ``string``."}});
}

void pybind_webrtc_server(py::module &m) {
    py::module m_submodule = m.def_submodule(
            "webrtc_server",
            "Functionality for remote visualization over WebRTC.");
    pybind_webrtc_server_functions(m_submodule);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
