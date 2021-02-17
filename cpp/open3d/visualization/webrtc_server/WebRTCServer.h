#pragma once

#include <functional>
#include <memory>
#include <string>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class WebRTCServer {
public:
    WebRTCServer(const std::string& http_address = "localhost:8888",
                 const std::string& web_root =
                         "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                         "webrtc_server/html");
    void Run();

    void OnDataChannelMessage(const std::string& message);

    void SetMouseButtonCallback(std::function<void(int, double, double)> f);
    void SetMouseMoveCallback(std::function<void(int, double, double)> f);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
