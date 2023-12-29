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

#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"

#include <functional>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

static int LogMessage(const struct mg_connection *conn, const char *message) {
    utility::LogInfo("CivetServer: {}", message);
    return 0;
}

static struct CivetCallbacks _callbacks;
static const struct CivetCallbacks *getCivetCallbacks() {
    memset(static_cast<void *>(&_callbacks), 0, sizeof(_callbacks));
    _callbacks.log_message = &LogMessage;
    return &_callbacks;
}

class RequestHandler : public CivetHandler {
public:
    RequestHandler(HttpServerRequestHandler::HttpFunction &func)
        : func_(func) {}

    bool handle(CivetServer *server, struct mg_connection *conn) {
        bool ret = false;
        const struct mg_request_info *req_info = mg_get_request_info(conn);

        // Read input.
        Json::Value in = this->getInputMessage(req_info, conn);

        // Invoke API implementation.
        Json::Value out(func_(req_info, in));

        // Fill out.
        std::string answer = "";
        if (out.isNull() == false) {
            answer = Json::writeString(writer_builder_, out);
            mg_printf(conn, "HTTP/1.1 200 OK\r\n");
            mg_printf(conn, "Access-Control-Allow-Origin: *\r\n");
            mg_printf(conn, "Content-Type: text/plain\r\n");
            mg_printf(conn, "Content-Length: %zd\r\n", answer.size());
            mg_printf(conn, "\r\n");
            mg_write(conn, answer.c_str(), answer.size());
            ret = true;
        }

        utility::LogDebug(
                "request_uri: {}, local_uri: {}, request_method: {}, "
                "query_string: {}, content_length: {}, answer: {}.",
                req_info->request_uri, req_info->local_uri,
                req_info->request_method,
                req_info->query_string ? req_info->query_string : "nullptr",
                req_info->content_length, answer);
        return ret;
    }
    bool handleGet(CivetServer *server, struct mg_connection *conn) override {
        return handle(server, conn);
    }
    bool handlePost(CivetServer *server, struct mg_connection *conn) override {
        return handle(server, conn);
    }

private:
    HttpServerRequestHandler::HttpFunction func_;
    Json::StreamWriterBuilder writer_builder_;
    Json::CharReaderBuilder reader_builder_;

    Json::Value getInputMessage(const struct mg_request_info *req_info,
                                struct mg_connection *conn) {
        Json::Value json_message;

        // Read input.
        long long tlen = req_info->content_length;
        if (tlen > 0) {
            std::string body;
            long long nlen = 0;
            const long long bufSize = 1024;
            char buf[bufSize];
            while (nlen < tlen) {
                long long rlen = tlen - nlen;
                if (rlen > bufSize) {
                    rlen = bufSize;
                }
                rlen = mg_read(conn, buf, (size_t)rlen);
                if (rlen <= 0) {
                    break;
                }
                body.append(buf, rlen);

                nlen += rlen;
            }

            // Parse in.
            std::unique_ptr<Json::CharReader> reader(
                    reader_builder_.newCharReader());
            std::string errors;
            if (!reader->parse(body.c_str(), body.c_str() + body.size(),
                               &json_message, &errors)) {
                utility::LogWarning("Received unknown message: {}, errors: {}.",
                                    body, errors);
            }
        }
        return json_message;
    }
};

HttpServerRequestHandler::HttpServerRequestHandler(
        std::map<std::string, HttpFunction> &func,
        const std::vector<std::string> &options)
    : CivetServer(options, getCivetCallbacks()) {
    // Register handlers.
    for (auto it : func) {
        this->addHandler(it.first, new RequestHandler(it.second));
    }
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
