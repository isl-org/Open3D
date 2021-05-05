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
#include "open3d/utility/IJsonConvertible.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"
#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"
#include "open3d/visualization/webrtc_server/ImageCapturer.h"
#include "open3d/visualization/webrtc_server/PeerConnectionManager.h"
#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

struct WebRTCServer::Impl {
    // HTTP handshake server settings.
    bool http_handshake_enabled_ = true;
    std::string http_address_;  // Used when http_handshake_enabled_ == true.
    std::string web_root_;      // Used when http_handshake_enabled_ == true.

    // Callback functions.
    std::function<void(int, double, double, int)> mouse_button_callback_ =
            nullptr;
    std::function<void(int, double, double, int)> mouse_move_callback_ =
            nullptr;
    std::function<void(double, double, int, double, double)>
            mouse_wheel_callback_ = nullptr;
    std::function<void(const std::string&, const gui::MouseEvent&)>
            mouse_event_callback_ = nullptr;
    std::function<void(const std::string&)> redraw_callback_ = nullptr;

    // PeerConnectionManager is used for setting up connections and managing API
    // call entry points.
    std::unique_ptr<PeerConnectionManager> peer_connection_manager_ = nullptr;

    // Utilities.
    static std::string GetEnvWebRTCIP() {
        if (const char* env_p = std::getenv("WEBRTC_IP")) {
            return std::string(env_p);
        } else {
            return "localhost";
        }
    }
    static std::string GetEnvWebRTCPort() {
        if (const char* env_p = std::getenv("WEBRTC_PORT")) {
            return std::string(env_p);
        } else {
            return "8888";
        }
    }
    static std::string GetEnvWebRTCWebRoot() {
        // TODO: package WEBRTC_WEB_ROOT with GUI resource files
        if (const char* env_p = std::getenv("WEBRTC_WEB_ROOT")) {
            return std::string(env_p);
        } else {
            std::string resource_path(
                    gui::Application::GetInstance().GetResourcePath());
            return resource_path + "/html";
        }
    }
};

void WebRTCServer::SetMouseEventCallback(
        std::function<void(const std::string&, const gui::MouseEvent&)> f) {
    impl_->mouse_event_callback_ = f;
}

void WebRTCServer::SetRedrawCallback(
        std::function<void(const std::string&)> f) {
    impl_->redraw_callback_ = f;
}

void WebRTCServer::OnDataChannelMessage(const std::string& message) {
    try {
        Json::Value value = utility::StringToJson(message);
        gui::MouseEvent me;
        if (value.get("class_name", "").asString() == "MouseEvent" &&
            value.get("window_uid", "").asString() != "" &&
            me.FromJson(value)) {
            const std::string window_uid =
                    value.get("window_uid", "").asString();
            utility::LogInfoConsole(
                    "WebRTCServer::Impl::OnDataChannelMessage: window_uid: {}, "
                    "MouseEvent: {}",
                    window_uid, me.ToString());
            if (impl_->mouse_event_callback_) {
                impl_->mouse_event_callback_(window_uid, me);
            }
        } else if (value.get("class_name", "").asString() == "ResizeEvent" &&
                   value.get("window_uid", "").asString() != "") {
            const std::string window_uid =
                    value.get("window_uid", "").asString();
            const int height = value.get("height", 0).asInt();
            const int width = value.get("width", 0).asInt();
            if (height <= 0 || width <= 0) {
                utility::LogInfoConsole(
                        "Invalid heigh {} or width {}, ResizeEvent ignored.",
                        height, width);
            }
            utility::LogInfoConsole("ResizeEvent {}: ({}, {})", window_uid,
                                    height, width);
            webrtc_server::WebRTCWindowSystem::GetInstance()->SetWindowSize(
                    gui::Application::GetInstance()
                            .GetWindowByUID(window_uid)
                            ->GetOSWindow(),
                    width, height);
        }
    } catch (...) {
        utility::LogInfoConsole(
                "WebRTCServer::Impl::OnDataChannelMessage: cannot parse {}.",
                message);
    }
}

void WebRTCServer::OnFrame(const std::string& window_uid,
                           const std::shared_ptr<core::Tensor>& im) {
    // Get the WebRTC stream that corresponds to the window_uid.
    rtc::scoped_refptr<BitmapTrackSourceInterface> video_track_source =
            impl_->peer_connection_manager_->GetVideoTrackSource(window_uid);

    // video_track_source is nullptr if the server is running but no client is
    // connected.
    if (video_track_source) {
        // TODO: this OnFrame(im); is a blocking call. Do we need to handle
        // OnFrame in a separate thread? e.g. attach to a queue of frames, even
        // if the queue size is just 1.
        video_track_source->OnFrame(im);
    }
}

void WebRTCServer::SendInitFrames(const std::string& window_uid) {
    auto sender = [this, &window_uid]() {
        static const int s_max_initial_frames = 5;
        static const int s_sleep_between_frames_ms = 100;
        for (int i = 0; i < s_max_initial_frames; ++i) {
            this->impl_->redraw_callback_(window_uid);
            std::this_thread::sleep_for(
                    std::chrono::milliseconds(s_sleep_between_frames_ms));
            utility::LogInfoConsole("Sent init frames {}", i);
        }
    };
    std::thread thread(sender);
    thread.join();
}

std::vector<std::string> WebRTCServer::GetWindowUIDs() const {
    return gui::Application::GetInstance().GetWindowUIDs();
}

WebRTCServer::WebRTCServer() : impl_(new WebRTCServer::Impl()) {
    // impl_->web_root_ is filled at WebRTCServer::Run(), since it relies on
    // the GetResourcePath(), which happens after Application::Initialize().
    impl_->http_handshake_enabled_ = true;
    impl_->http_address_ =
            Impl::GetEnvWebRTCIP() + ":" + Impl::GetEnvWebRTCPort();
}

WebRTCServer& WebRTCServer::GetInstance() {
    static WebRTCServer webrtc_server;
    return webrtc_server;
}

void WebRTCServer::Run() {
    std::cout << "WebRTCServer::Run()" << std::endl;

    impl_->web_root_ = Impl::GetEnvWebRTCWebRoot();
    utility::LogInfo("impl_->web_root_: {}", impl_->web_root_);

    const std::string web_root = impl_->web_root_;
    const std::string http_address = impl_->http_address_;
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

    impl_->peer_connection_manager_ = std::make_unique<PeerConnectionManager>(
            this, ice_servers, config["urls"], ".*", "");
    if (impl_->peer_connection_manager_->InitializePeerConnection()) {
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
    if (impl_->http_handshake_enabled_) {
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
                    impl_->peer_connection_manager_->GetHttpApi();

            // Main loop.
            std::cout << "HTTP Listen at " << http_address << std::endl;
            HttpServerRequestHandler civet_server(func, options);
            // signal(SIGINT, &signal_handler);  // TODO: fix me
            thread->Run();
        } catch (const CivetException& ex) {
            std::cout << "Cannot Initialize start HTTP server exception:"
                      << ex.what() << std::endl;
        }
    } else {
        thread->Run();
    }
    rtc::CleanupSSL();
    std::cout << "Exit" << std::endl;
}

std::string WebRTCServer::CallHttpRequest(const std::string& entry_point,
                                          const std::string& query_string,
                                          const std::string& data) const {
    utility::LogInfoConsole(
            "WebRTCServer::CallHttpRequest /////////////////////");

    std::string query_string_trimmed = "";
    if (!query_string.empty() && query_string[0] == '?') {
        query_string_trimmed =
                query_string.substr(1, query_string.length() - 1);
    }

    utility::LogInfoConsole("entry_point: {}", entry_point);
    utility::LogInfoConsole("query_string_trimmed: {}", query_string_trimmed);
    utility::LogInfoConsole("data: {}", data);

    std::string result = "";
    if (entry_point == "/api/getMediaList") {
        result = utility::JsonToString(
                impl_->peer_connection_manager_->GetMediaList());
    } else if (entry_point == "/api/getIceServers") {
        result = utility::JsonToString(
                impl_->peer_connection_manager_->GetIceServers());
    } else if (entry_point == "/api/getIceCandidate") {
        std::string peerid;
        if (!query_string_trimmed.empty()) {
            CivetServer::getParam(query_string_trimmed.c_str(), "peerid",
                                  peerid);
        }
        result = utility::JsonToString(
                impl_->peer_connection_manager_->GetIceCandidateList(peerid));
    } else if (entry_point == "/api/hangup") {
        std::string peerid;
        if (!query_string_trimmed.empty()) {
            CivetServer::getParam(query_string_trimmed.c_str(), "peerid",
                                  peerid);
        }
        result = utility::JsonToString(
                impl_->peer_connection_manager_->HangUp(peerid));
    } else if (entry_point == "/api/call") {
        std::string peerid;
        std::string url;
        std::string options;
        if (!query_string_trimmed.empty()) {
            CivetServer::getParam(query_string_trimmed.c_str(), "peerid",
                                  peerid);
            CivetServer::getParam(query_string_trimmed.c_str(), "url", url);
            CivetServer::getParam(query_string_trimmed.c_str(), "options",
                                  options);
        }
        result = utility::JsonToString(impl_->peer_connection_manager_->Call(
                peerid, url, options, utility::StringToJson(data)));
    } else if (entry_point == "/api/addIceCandidate") {
        std::string peerid;
        if (!query_string_trimmed.empty()) {
            CivetServer::getParam(query_string_trimmed.c_str(), "peerid",
                                  peerid);
        }
        result = utility::JsonToString(
                impl_->peer_connection_manager_->AddIceCandidate(
                        peerid, utility::StringToJson(data)));
    }

    utility::LogInfoConsole("result: {}", result);
    utility::LogInfoConsole(
            "///////////////////////////////////////////////////");

    return result;
}

void WebRTCServer::EnableWebRTC() {
    utility::LogInfo("WebRTC GUI backend enabled.");
    gui::Application::GetInstance().SetWindowSystem(
            webrtc_server::WebRTCWindowSystem::GetInstance());
}

void WebRTCServer::DisableHttpHandshake() {
    utility::LogInfo("WebRTCServer: HTTP handshake server disabled.");
    impl_->http_handshake_enabled_ = false;
}

void WebRTCServer::CloseWindowConnections(const std::string& window_uid) {
    utility::LogInfo("Calling WebRTCServer::CloseWindowConnections: {}",
                     window_uid);
    impl_->peer_connection_manager_->CloseWindowConnections(window_uid);
    utility::LogInfo("Done WebRTCServer::CloseWindowConnections: {}",
                     window_uid);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
