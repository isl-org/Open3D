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
            "register_HTML_DOM_callback",
            [](const std::string &html_element_id, const std::string &event,
               std::function<void(const std::string &)> callback) {
                return WebRTCWindowSystem::GetInstance()
                        ->RegisterHTMLDOMCallback(html_element_id, event,
                                                  callback);
            },
            "html_element_id"_a, "event"_a, "callback"_a,
            R"(
Register callback for an HTML DOM event.

The callback will be executed when the corresponding event is received
on the WebRTC data channel as a message. This can be sent through
JavaScript from the client (browser) with a ``class_name`` of
"HTMLDOMEvent". Any arbitrary (``html_element_id``, ``event``) may be used
to register a callback, but see
`<https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement>`__ for
standard HTML elements and events.  A second callback for the same \p
(``html_element_id``, ``event``) combination will overwrite the previously
registered callback.

.. code:: python

    # Register callback in Python
    import open3d as o3d
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.webrtc_server.register_HTML_DOM_callback(
        "open3d-dashboard", "input",
        lambda data: print(f"Received HTML DOM message with data: {data}"))

.. code:: js

    /* Send message in JavaScript to trigger callback. this is WebRTCStreamer object */
    this.dataChannel.send('{"class_name":"HTMLDOMEvent",
        "element_id":"open3d-dashboard",
        "event":"input",
        "data":"Test event"}')

.. warning:: The event data passed to the callback must be validated inside
the callback before use.
            )");

    docstring::FunctionDocInject(
            m, "register_HTML_DOM_callback",
            {{"html_element_id",
              "Id of html element that triggered the ``event``."},
             {"event", "Name of event."},
             {"callback",
              "Function to call when this ``event`` occurs. The function "
              "should accept a string argument (corresponding to the event "
              "data, such as form data or updated value of a slider) and not "
              "return anything."}});

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
}

void pybind_webrtc_server(py::module &m) {
    py::module m_submodule = m.def_submodule("webrtc_server");
    pybind_webrtc_server_functions(m_submodule);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
