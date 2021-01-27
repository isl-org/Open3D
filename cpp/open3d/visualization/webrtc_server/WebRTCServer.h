#pragma once

#include <functional>
#include <string>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class WebRTCServer {
public:
    WebRTCServer(const std::string& http_address = "localhost:8888",
                 const std::string& web_root =
                         "/home/yixing/repo/Open3D/cpp/open3d/visualization/"
                         "webrtc_server/html")
        : http_address_(http_address), web_root_(web_root) {}

    void Run();

    void SetMouseButtonCallback(std::function<void(int, double, double)> f) {
        mouse_button_callback_ = f;
    }

private:
    std::string http_address_;
    std::string web_root_;
    std::function<void(int, double, double)> mouse_button_callback_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
