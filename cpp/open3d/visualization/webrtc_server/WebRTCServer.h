// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open3d/utility/FileSystem.h"

namespace open3d {

namespace geometry {
class Image;
}

namespace core {
class Tensor;
}

namespace visualization {
namespace gui {
struct MouseEvent;
class Window;
}  // namespace gui

namespace webrtc_server {

inline std::string GetEnvIP() {
    if (const char* env_p = std::getenv("WEBRTC_IP")) {
        return std::string(env_p);
    } else {
        return "localhost";
    }
}

inline std::string GetEnvPort() {
    if (const char* env_p = std::getenv("WEBRTC_PORT")) {
        return std::string(env_p);
    } else {
        return "8888";
    }
}

class WebRTCServer {
public:
    WebRTCServer(const std::string& http_address = GetEnvIP() + ":" +
                                                   GetEnvPort(),
                 const std::string& web_root =
                         utility::filesystem::GetUnixHome() +
                         "/repo/Open3D/cpp/open3d/visualization/"
                         "webrtc_server/html");
    void Run();

    // Client -> server message.
    void OnDataChannelMessage(const std::string& message);

    // Set MouseEvent callback function. If a client -> server message is of
    // MouseEvent type, the callback funciton will be triggered. The client
    // message shall also contain the corresponding window_uid.
    void SetMouseEventCallback(
            std::function<void(const std::string&, const gui::MouseEvent&)> f);

    // Set redraw callback function. Server can force a redraw. Then redraw then
    // triggers OnFrame(), where a server -> client frame will be sent.
    void SetRedrawCallback(std::function<void(const std::string&)> f);

    // Server -> client frame.
    void OnFrame(const std::string& window_uid,
                 const std::shared_ptr<core::Tensor>& im);

    // Send initial frames. This flushs the WebRTC vidoe stream. After the
    // initial frames, new frames will only be sent at redraw events.
    void SendInitFrames(const std::string& window_uid);

    // List available windows.
    std::vector<std::string> GetWindowUIDs() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
