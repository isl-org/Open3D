/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** main.cpp
**
** -------------------------------------------------------------------------*/

#include "open3d/visualization/webrtc_server/WebRTCStreamer.h"

#include <p2p/base/basic_packet_socket_factory.h>
#include <p2p/base/stun_server.h>
#include <p2p/base/turn_server.h>
#include <rtc_base/ssl_adapter.h>
#include <rtc_base/thread.h>
#include <signal.h>

#include <fstream>
#include <iostream>

#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"
#include "open3d/visualization/webrtc_server/PeerConnectionManager.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

// TODO: move this into the class.
static PeerConnectionManager* peer_connection_manager = nullptr;

void SignalHandler(int n) {
    printf("SIGINT\n");
    // delete need thread still running
    delete peer_connection_manager;
    peer_connection_manager = nullptr;
    rtc::Thread::Current()->Quit();
}

void WebRTCStreamer::Run() {
    const std::string web_root = web_root_;
    const std::string http_address = http_address_;
    const std::vector<std::string> stun_urls{"stun:stun.l.google.com:19302"};

    // Logging settings.
    // src/rtc_base/logging.h: LS_VERBOSE, LS_ERROR
    rtc::LogMessage::LogToDebug((rtc::LoggingSeverity)rtc::LS_ERROR);
    std::cout << "Logger level:" << rtc::LogMessage::GetLogToDebug()
              << std::endl;
    rtc::LogMessage::LogTimestamps();
    rtc::LogMessage::LogThreads();

    // WebRTC server (a PeerConnectionManager).
    rtc::Thread* thread = rtc::Thread::Current();
    rtc::InitializeSSL();
    std::list<std::string> ice_servers(stun_urls.begin(), stun_urls.end());
    Json::Value config;

    config["urls"]["Bunny"]["video"] =
            "file:///home/yixing/repo/webrtc-streamer/html/"
            "Big_Buck_Bunny_360_10s_1MB.webm";
    peer_connection_manager =
            new PeerConnectionManager(ice_servers, config["urls"], ".*", "");
    if (peer_connection_manager->InitializePeerConnection()) {
        std::cout << "InitializePeerConnection() succeeded." << std::endl;
    } else {
        throw std::runtime_error("InitializePeerConnection() failed.");
    }

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
    try {
        // PeerConnectionManager provides a set of callback functions for
        // HttpServerRequestHandler.
        std::map<std::string, HttpServerRequestHandler::httpFunction> func =
                peer_connection_manager->getHttpApi();

        // Main loop.
        std::cout << "HTTP Listen at " << http_address << std::endl;
        HttpServerRequestHandler civet_server(func, options);
        signal(SIGINT, SignalHandler);
        thread->Run();
    } catch (const CivetException& ex) {
        std::cout << "Cannot Initialize start HTTP server exception:"
                  << ex.what() << std::endl;
    }

    rtc::CleanupSSL();
    std::cout << "Exit" << std::endl;
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
