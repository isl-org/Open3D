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
/// \file WebRTCWindowSystem.h
///
/// The main header file for WebRTC visualizer.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open3d/visualization/gui/BitmapWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

/// \brief WebRTCWindowSystem is a BitmapWindowSystem with a WebRTC server that
/// sends video frames to remote clients for visualization.
///
/// WebRTCWindowSystem owns a PeerConnectionManager, which manages all things
/// related to the WebRTC connections, e.g. get media lists, get and add ICE
/// candidates, connect to a media and hangup.
///
/// When the client visit a Open3D visualizer's website for visualization
/// (a.k.a. standalone mode), an HTTP handshake server will be used to serve the
/// website and perform handshake to establish the WebRTC connection. In Jupyter
/// mode, the HTTP handshake server is disabled and the handshake is done via
/// Jupyter's JavaScript<->Python communication channel.
///
/// WebRTCWindowSystem shall be used as a global singleton. Both the
/// PeerConnectionManager and the HTTP handshake server runs on different
/// threads.
class WebRTCWindowSystem : public gui::BitmapWindowSystem {
public:
    static std::shared_ptr<WebRTCWindowSystem> GetInstance();
    virtual ~WebRTCWindowSystem();
    OSWindow CreateOSWindow(gui::Window* o3d_window,
                            int width,
                            int height,
                            const char* title,
                            int flags) override;
    void DestroyWindow(OSWindow w) override;

    /// List available windows.
    std::vector<std::string> GetWindowUIDs() const;
    std::string GetWindowUID(OSWindow w) const;
    OSWindow GetOSWindowByUID(const std::string& uid) const;

    /// Start WebRTC server in a background thread.
    void StartWebRTCServer();

    /// Client -> server message.
    void OnDataChannelMessage(const std::string& message);

    /// Server -> client frame.
    void OnFrame(const std::string& window_uid,
                 const std::shared_ptr<core::Tensor>& im);

    /// Send initial frames. This flushes the WebRTC video stream. After the
    /// initial frames, new frames will only be sent at triggered events.
    void SendInitFrames(const std::string& window_uid);

    /// \brief Call PeerConnectionManager's web request API.
    ///
    /// This function is called in JavaScript via Python binding to mimic the
    /// behavior of sending HTTP request via fetch() in JavaScript.
    ///
    /// With fetch:
    /// data = {method: "POST", body: JSON.stringify(candidate)};
    /// fetch("/api/addIceCandidate?peerid=" + peerid, data);
    ///
    /// Now with CallHttpAPI:
    /// open3d.visualization.webrtc_server("/api/addIceCandidate",
    ///                                    "?peerid=" + peerid,
    ///                                    data["body"]);
    ///
    /// \param entry_point URL part before '?'.
    /// \param query_string URL part after '?', including '?'. If '?' is not the
    /// first character or if the stirng is empty, the query_string is ignored.
    /// \param data JSON-encoded string.
    std::string CallHttpAPI(const std::string& entry_point,
                            const std::string& query_string = "",
                            const std::string& data = "") const;

    /// Sets WebRTCWindowSystem as the default window system in Application.
    /// This enables a global WebRTC server and each gui::Window will be
    /// rendered to a WebRTC video stream.
    void EnableWebRTC();

    /// HTTP handshake server is enabled by default. In Jupyter environment,
    /// call DisableHttpHandshake() before StartWebRTCServer().
    void DisableHttpHandshake();

    /// Close all WebRTC connections that correspond to a Window.
    void CloseWindowConnections(const std::string& window_uid);

    /// \brief Register callback for an HTML DOM event.
    ///
    /// The callback will be executed when the corresponding event is received
    /// on the WebRTC data channel as a message. This can be sent through
    /// JavaScript from the client (browser) with a \verb class_name of
    /// "HTMLDOMEvent". Any arbitrary (\p html_element_id, \p event) may be used
    /// to register a callback, but see
    /// https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement for
    /// standard HTML elements and events.  A second callback for the same \p
    /// html_element_id \p event combination will overwrite the previously
    /// registered callback.
    ///
    /// \code{.cpp}
    /// // Register callback in C++
    /// open3d::visualization::webrtc_server::enable_webrtc()
    /// open3d::visualization::webrtc_server::RegisterHTMLDOMCallback(
    ///     "open3d-dashboard", "input",
    ///     lambda data: print(f"Received HTML DOM message with data: {data}"))
    /// \endcode
    ///
    /// \code{.js}
    /// /* Send message in JavaScript to trigger callback. this is
    /// WebRTCStreamer object */
    /// this.dataChannel.send('{"class_name":"HTMLDOMEvent",
    ///     "element_id":"open3d-dashboard",
    ///     "event":"input",
    ///     "data":"Test event"}')
    /// \endcode
    ///
    /// \warning The event data passed to the callback must be validated inside
    /// the callback before use.
    ///
    /// \param html_element_id Id of html element that triggered the \p event.
    /// \param event Name of event.
    /// \param callback Function to call when this \p event occurs. The function
    /// should accept a std::string argument (corresponding to the event data,
    /// such as form data or updated value of a slider) and return void.
    void RegisterHTMLDOMCallback(
            const std::string& html_element_id,
            const std::string& event,
            std::function<void(const std::string&)> callback);

private:
    WebRTCWindowSystem();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
