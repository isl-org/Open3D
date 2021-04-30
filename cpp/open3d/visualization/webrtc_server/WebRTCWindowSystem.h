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

#include "open3d/visualization/gui/BitmapWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class WebRTCServer;

// TODO (Yixing): Merged this class with WebRTCServer.
class WebRTCWindowSystem : public gui::BitmapWindowSystem {
public:
    static std::shared_ptr<WebRTCWindowSystem> GetInstance();
    virtual ~WebRTCWindowSystem();
    void SetMouseEventCallback(
            std::function<void(const std::string&, const gui::MouseEvent&)> f);
    void SetRedrawCallback(std::function<void(const std::string&)> f);
    void StartWebRTCServer();
    void CloseWindowConnections(const std::string& window_uid);

private:
    WebRTCWindowSystem();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
