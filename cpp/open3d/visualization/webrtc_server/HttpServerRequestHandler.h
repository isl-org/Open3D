// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from Open3D's public API. Do not
// put this in Open3D.h.in.

#pragma once

#include <CivetServer.h>
#include <json/json.h>

#include <functional>
#include <map>

namespace open3d {
namespace visualization {
namespace webrtc_server {

class HttpServerRequestHandler : public CivetServer {
public:
    typedef std::function<Json::Value(const struct mg_request_info* req_info,
                                      const Json::Value&)>
            HttpFunction;

    HttpServerRequestHandler(std::map<std::string, HttpFunction>& func,
                             const std::vector<std::string>& options);
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
