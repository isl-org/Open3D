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
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------

#include "open3d/visualization/webrtc_server/WebRTCServer.h"

#include <p2p/base/basic_packet_socket_factory.h>
#include <p2p/base/stun_server.h>
#include <p2p/base/turn_server.h>
#include <rtc_base/ssl_adapter.h>
#include <rtc_base/thread.h>
#include <signal.h>

#include <fstream>
#include <iostream>

#include "open3d/geometry/Image.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"
#include "open3d/visualization/webrtc_server/GlobalBuffer.h"
#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"
#include "open3d/visualization/webrtc_server/ImageCapturer.h"
#include "open3d/visualization/webrtc_server/PeerConnectionManager.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

struct WebRTCServer::Impl {
    WebRTCServer* webrtc_server_;  // Parent.
    std::string http_address_;
    std::string web_root_;
    std::function<void(int, double, double, int)> mouse_button_callback_ =
            nullptr;
    std::function<void(int, double, double, int)> mouse_move_callback_ =
            nullptr;
    std::function<void(double, double, int, double, double)>
            mouse_wheel_callback_ = nullptr;
    // TODO: make this and Impl unique_ptr?
    std::shared_ptr<PeerConnectionManager> peer_connection_manager_ = nullptr;
    void OnDataChannelMessage(const std::string& message);
    void OnFrame(const geometry::Image& im);
    void Run();
    int mouse_button_status_ = 0;
};

void WebRTCServer::Impl::OnFrame(const geometry::Image& im) {
    // TODO: name this differently and handle multiple instances.
    // dynamic_cast is better but "-fno-rtti" is required for WebRTC.

    // video_track_source is nullptr if the server is running but no client is
    // connected.
    rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> video_track_source =
            peer_connection_manager_->GetVideoTrackSource("imageOpen3D");
    utility::LogInfo("WebRTCServer::Impl::OnFrame");
    if (video_track_source != nullptr) {
        utility::LogInfo(
                "WebRTCServer::Impl::OnFrame, video_track_source != nullptr");

        // auto image_capturer =
        //         reinterpret_cast<ImageCapturer*>(video_track_source.get());

        t::geometry::Image t_im = t::geometry::Image::FromLegacyImage(im);
        core::Tensor rgb_frame = t_im.AsTensor();
        GlobalBuffer::GetInstance().Write(rgb_frame);

        // utility::LogInfo("To call
        // image_capturer->OnCaptureResult(im_frame);");
        // image_capturer->OnFrame(im_frame);
        // utility::LogInfo("Called
        // image_capturer->OnCaptureResult(im_frame);");
    }
}

void WebRTCServer::Impl::OnDataChannelMessage(const std::string& message) {
    // TODO: use Json message.
    utility::LogInfo("WebRTCServer::Impl::OnDataChannelMessage: {}", message);
    std::vector<std::string> tokens;
    utility::SplitString(tokens, message);
    if (tokens.size() > 0) {
        std::string type = tokens[0];
        if (type == "mousemove") {
            double x = static_cast<double>(std::stoi(tokens[1]));
            double y = static_cast<double>(std::stoi(tokens[2]));
            int mods = std::stoi(tokens[3]);
            if (mouse_move_callback_) {
                mouse_move_callback_(mouse_button_status_, x, y, mods);
            }
        } else if (type == "mousedown") {
            mouse_button_status_ = 1;
            int action = 1;
            double x = static_cast<double>(std::stoi(tokens[1]));
            double y = static_cast<double>(std::stoi(tokens[2]));
            int mods = std::stoi(tokens[3]);
            if (mouse_button_callback_) {
                mouse_button_callback_(action, x, y, mods);
            }
        } else if (type == "mouseup") {
            mouse_button_status_ = 0;
            int action = 0;
            double x = static_cast<double>(std::stoi(tokens[1]));
            double y = static_cast<double>(std::stoi(tokens[2]));
            int mods = std::stoi(tokens[3]);
            if (mouse_button_callback_) {
                mouse_button_callback_(action, x, y, mods);
            }
        } else if (type == "wheel") {
            double x = static_cast<double>(std::stoi(tokens[1]));
            double y = static_cast<double>(std::stoi(tokens[2]));
            int mods = std::stoi(tokens[3]);
            double dx = static_cast<double>(std::stoi(tokens[4]));
            double dy = static_cast<double>(std::stoi(tokens[5]));
            // Flip the sign and set abaolute value to 5.
            // TODO: better scaling.
            dx = dx == 0 ? dx : -dx / std::abs(dx) * 5;
            dy = dy == 0 ? dy : -dy / std::abs(dy) * 5;
            if (mouse_wheel_callback_) {
                mouse_wheel_callback_(x, y, mods, dx, dy);
            }
        }
    }
}

void WebRTCServer::SetMouseButtonCallback(
        std::function<void(int, double, double, int)> f) {
    impl_->mouse_button_callback_ = f;
}

void WebRTCServer::SetMouseMoveCallback(
        std::function<void(int, double, double, int)> f) {
    impl_->mouse_move_callback_ = f;
}

void WebRTCServer::SetMouseWheelCallback(
        std::function<void(double, double, int, double, double)> f) {
    impl_->mouse_wheel_callback_ = f;
}

void WebRTCServer::OnDataChannelMessage(const std::string& message) {
    impl_->OnDataChannelMessage(message);
}

void WebRTCServer::OnFrame(const geometry::Image& im) { impl_->OnFrame(im); }

WebRTCServer::WebRTCServer(const std::string& http_address,
                           const std::string& web_root)
    : impl_(new WebRTCServer::Impl()) {
    impl_->webrtc_server_ = this;
    impl_->http_address_ = http_address;
    impl_->web_root_ = web_root;
}

void WebRTCServer::Impl::Run() {
    std::cout << "WebRTCServer::Run()" << std::endl;

    const std::string web_root = web_root_;
    const std::string http_address = http_address_;
    const std::vector<std::string> stun_urls{"stun:stun.l.google.com:19302"};

    // Logging settings.
    // src/rtc_base/logging.h: LS_VERBOSE, LS_ERROR
    rtc::LogMessage::LogToDebug((rtc::LoggingSeverity)rtc::LS_ERROR);
    // rtc::LogMessage::LogToDebug((rtc::LoggingSeverity)rtc::LS_VERBOSE);
    std::cout << "Logger level:" << rtc::LogMessage::GetLogToDebug()
              << std::endl;
    rtc::LogMessage::LogTimestamps();
    rtc::LogMessage::LogThreads();

    // WebRTC server (a PeerConnectionManager).
    rtc::Thread* thread = rtc::Thread::Current();
    rtc::InitializeSSL();
    std::list<std::string> ice_servers(stun_urls.begin(), stun_urls.end());
    Json::Value config;

    peer_connection_manager_ = std::make_shared<PeerConnectionManager>(
            this->webrtc_server_, ice_servers, config["urls"], ".*", "");
    if (peer_connection_manager_->InitializePeerConnection()) {
        std::cout << "InitializePeerConnection() succeeded." << std::endl;
    } else {
        throw std::runtime_error("InitializePeerConnection() failed.");
    }

    // TODO: fix me.
    // https://stackoverflow.com/a/20291676/1255535.
    // https://stackoverflow.com/q/7852101/1255535.
    // auto signal_handler = [this](int n) {
    //     printf("SIGINT\n");
    //     // delete need thread still running
    //     peer_connection_manager_ = nullptr;
    //     rtc::Thread::Current()->Quit();
    // };

    // CivetWeb http server.
    std::vector<std::string> options;
    options.push_back("document_root");
    options.push_back(web_root);
    options.push_back("enable_directory_listing");
    options.push_back("no");
    options.push_back("additional_header");
    options.push_back("X-Frame-Options: SAMEORIGIN");
    options.push_back("access_control_allow_origin");
    options.push_back("*");
    options.push_back("listening_ports");
    options.push_back(http_address);
    options.push_back("enable_keep_alive");
    options.push_back("yes");
    options.push_back("keep_alive_timeout_ms");
    options.push_back("1000");
    options.push_back("decode_url");
    options.push_back("no");
    try {
        // PeerConnectionManager provides a set of callback functions for
        // HttpServerRequestHandler.
        std::map<std::string, HttpServerRequestHandler::HttpFunction> func =
                peer_connection_manager_->GetHttpApi();

        // Main loop.
        std::cout << "HTTP Listen at " << http_address << std::endl;
        HttpServerRequestHandler civet_server(func, options);
        // signal(SIGINT, &signal_handler);  // TODO: fix me
        thread->Run();
    } catch (const CivetException& ex) {
        std::cout << "Cannot Initialize start HTTP server exception:"
                  << ex.what() << std::endl;
    }

    rtc::CleanupSSL();
    std::cout << "Exit" << std::endl;
}

void WebRTCServer::Run() { impl_->Run(); }

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
