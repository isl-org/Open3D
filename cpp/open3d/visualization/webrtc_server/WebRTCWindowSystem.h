// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <memory>
#include <string>
#include <vector>

#include "open3d/visualization/gui/BitmapWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

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

    /// Call PeerConnectionManager's web request API.
    /// This function is called in JavaScript via Python binding to mimic the
    /// behavior of sending HTTP request via fetch() in JavaScript.
    ///
    /// With fetch:
    /// data = {method: "POST", body: JSON.stringify(candidate)};
    /// fetch(this.srvurl + "/api/addIceCandidate?peerid=" + peerid, data);
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

private:
    WebRTCWindowSystem();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
