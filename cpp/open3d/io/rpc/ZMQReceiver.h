// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include "open3d/utility/Logging.h"

namespace zmq {
class message_t;
class socket_t;
class context_t;
}  // namespace zmq

namespace open3d {
namespace io {
namespace rpc {

class MessageProcessorBase;

namespace messages {
struct Request;
struct SetMeshData;
struct GetMeshData;
struct SetCameraData;
struct SetProperties;
struct SetActiveCamera;
struct SetTime;
}  // namespace messages

/// Class for the server side receiving requests from a client.
class ZMQReceiver {
public:
    /// Constructs a receiver listening on the specified address.
    /// \param address  Address to listen on.
    /// \param timeout       Timeout in milliseconds for sending the reply.
    ZMQReceiver(const std::string& address = "tcp://127.0.0.1:51454",
                int timeout = 10000);

    ZMQReceiver(const ZMQReceiver&) = delete;
    ZMQReceiver& operator=(const ZMQReceiver&) = delete;

    virtual ~ZMQReceiver();

    /// Starts the receiver mainloop in a new thread.
    void Start();

    /// Stops the receiver mainloop and joins the thread.
    /// This function blocks until the mainloop is done with processing
    /// messages that have already been received.
    void Stop();

    /// Returns the last error from the mainloop thread.
    std::runtime_error GetLastError();

    /// Sets the message processor object which will process incoming messages.
    void SetMessageProcessor(std::shared_ptr<MessageProcessorBase> processor);

private:
    void Mainloop();

    const std::string address_;
    const int timeout_;
    std::shared_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    std::thread thread_;
    std::mutex mutex_;
    bool keep_running_;
    std::atomic<bool> loop_running_;
    std::atomic<int> mainloop_error_code_;
    std::runtime_error mainloop_exception_;
    std::shared_ptr<MessageProcessorBase> processor_;
};

}  // namespace rpc
}  // namespace io
}  // namespace open3d
