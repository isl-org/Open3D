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

#include "open3d/utility/FileSystem.h"

namespace open3d {

namespace geometry {
class Image;
}

namespace visualization {
namespace webrtc_server {

class WebRTCServer {
public:
    WebRTCServer(const std::string& http_address = "localhost:8888",
                 const std::string& web_root =
                         utility::filesystem::GetUnixHome() +
                         "/repo/Open3D/cpp/open3d/visualization/"
                         "webrtc_server/html");
    void Run();

    // Client -> server message.
    void OnDataChannelMessage(const std::string& message);
    // Server -> client frame.
    void OnFrame(const geometry::Image& im);

    void SetMouseButtonCallback(
            std::function<void(int, double, double, int)> f);
    void SetMouseMoveCallback(std::function<void(int, double, double, int)> f);
    void SetMouseWheelCallback(
            std::function<void(double, double, int, double, double)> f);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
